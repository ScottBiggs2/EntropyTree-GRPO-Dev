"""Phase 5: Reward function tests. Phase 8.5: ExecutionLiteReward tests."""

import json
import tempfile
from pathlib import Path

import pytest

from src.rewards import ExecutionLiteReward, SyntaxReward, ExecutionReward


def test_valid_python_high_reward():
    r = SyntaxReward()
    score = r("def foo():\n    return 1", "def foo():")
    assert score > 0.5
    assert score <= 1.0


def test_invalid_python_low_reward():
    r = SyntaxReward()
    score = r("def foo( return 1  # syntax error", "")
    assert score <= 0.5


def test_empty_string_zero():
    r = SyntaxReward()
    assert r("", "") < 0.5
    assert r("", "def x():") >= 0.0


def test_def_return_docstring():
    r = SyntaxReward()
    code = '''def foo():
    """Docstring."""
    return 42
'''
    score = r(code, "")
    assert score >= 0.5 + 0.15 + 0.15 + 0.1


def test_execution_lite_unknown_prompt_zero():
    """ExecutionLiteReward returns 0.0 when prompt is not in registry."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"prompt": "def foo():", "function_name": "foo", "tests": [[0, 0]]}], f)
        path = f.name
    try:
        r = ExecutionLiteReward(registry_path=path)
        score = r("return 42", "def other_thing():")  # prompt not in registry
        assert score == 0.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_execution_lite_registry_lookup():
    """ExecutionLiteReward with temp registry: known completion gets fraction in [0,1]."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([
            {
                "prompt": "def fibonacci(n):",
                "function_name": "fibonacci",
                "tests": [[0, 0], [1, 1], [2, 1]],
            }
        ], f)
        path = f.name
    try:
        r = ExecutionLiteReward(registry_path=path, syntax_bonus=0.0)
        completion = "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        score = r(completion, "def fibonacci(n):")
        assert 0.0 <= score <= 1.0
        assert score == 1.0  # all 3 tests should pass
    finally:
        Path(path).unlink(missing_ok=True)
