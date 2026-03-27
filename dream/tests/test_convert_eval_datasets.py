"""Offline tests for HumanEval/MBPP → CodeTask conversion (no evalplus required)."""

import json
from pathlib import Path

import pytest

from dream.src.eval_dataset_convert import (
    extract_assertions_from_humaneval_test,
    humaneval_to_row,
    mbpp_to_row,
    replace_candidate_with_entry_point,
)
from dream.src.task_registry import load_code_tasks


def test_replace_candidate():
    assert replace_candidate_with_entry_point(
        ["assert candidate(1) == 2"], "foo"
    ) == ["assert foo(1) == 2"]


def test_extract_assertions_line_based():
    block = """
def check(candidate):
    assert candidate([1, 2], 0.5) == False
    assert candidate([3], 1.0) == True
"""
    got = extract_assertions_from_humaneval_test(block)
    assert len(got) == 2
    assert "candidate" in got[0]


def test_extract_assertions_ast_fallback():
    # No line-starting asserts (only inside check) — AST path
    block = """
def check(candidate):
    assert candidate(1) == 1
"""
    got = extract_assertions_from_humaneval_test(block)
    assert len(got) == 1
    assert got[0].startswith("assert")


def test_humaneval_to_row_loads_via_registry(tmp_path: Path):
    item = {
        "task_id": "HumanEval/0",
        "prompt": "def f(x: int) -> int:\n    \"\"\"Double.\"\"\"\n",
        "entry_point": "f",
        "canonical_solution": "def f(x: int) -> int:\n    return x * 2\n",
        "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(0) == 0\n",
    }
    row = humaneval_to_row(item, source="test_humaneval")
    p = tmp_path / "he.jsonl"
    p.write_text(json.dumps(row) + "\n")
    tasks = load_code_tasks(p)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.split == "eval"
    assert t.entry_point == "f"
    assert t.test_format == "assertion"
    assert t.tests == ["assert f(2) == 4", "assert f(0) == 0"]


def test_mbpp_to_row_loads_via_registry(tmp_path: Path):
    item = {
        "task_id": "MBPP/0",
        "text": "Write a function that adds two numbers.",
        "code": "def add(a, b):\n    return a + b\n",
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }
    row = mbpp_to_row(item, source="test_mbpp")
    p = tmp_path / "mbpp.jsonl"
    p.write_text(json.dumps(row) + "\n")
    tasks = load_code_tasks(p)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.split == "eval"
    assert t.entry_point == "add"
    assert "def add(" in t.starter_code


def test_humaneval_missing_assertions_raises():
    item = {
        "task_id": "HumanEval/bad",
        "prompt": "def f():\n    pass\n",
        "entry_point": "f",
        "canonical_solution": "",
        "test": "print(1)",
    }
    with pytest.raises(ValueError, match="assertions"):
        humaneval_to_row(item)
