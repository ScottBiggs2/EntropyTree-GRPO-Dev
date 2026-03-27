import json
from pathlib import Path

from dream.src.task_registry import (
    build_prompt_lookup,
    export_execution_lite_rows,
    filter_code_tasks,
    infer_default_split,
    load_code_tasks,
)


ROOT = Path(__file__).resolve().parents[2]


def test_load_sample_jsonl_tasks():
    tasks = load_code_tasks(ROOT / "dream" / "data" / "code_grpo_train.sample.jsonl")
    assert len(tasks) == 3
    assert tasks[0].entry_point == "fibonacci"
    assert "```python" in tasks[0].canonical_prompt


def test_load_legacy_execution_lite_json():
    tasks = load_code_tasks(ROOT / "data" / "execution_lite.json")
    assert tasks
    assert any(task.source == "execution_lite" for task in tasks)
    assert any(task.split == "dev" for task in tasks)


def test_build_prompt_lookup_supports_canonical_and_starter_code():
    tasks = load_code_tasks(ROOT / "dream" / "data" / "code_grpo_train.sample.jsonl")
    lookup = build_prompt_lookup(tasks)
    first = tasks[0]
    assert lookup[first.canonical_prompt] == first
    assert lookup[first.starter_code] == first


def test_filter_and_infer_splits():
    tasks = load_code_tasks(ROOT / "data" / "execution_lite.json")
    train = filter_code_tasks(tasks, "train")
    dev = filter_code_tasks(tasks, "dev")
    assert train
    assert dev
    assert infer_default_split(tasks, "initial_eval") in {"dev", "eval"}


def test_export_execution_lite_rows_round_trip_shape():
    tasks = load_code_tasks(ROOT / "dream" / "data" / "code_grpo_dev.sample.jsonl")
    rows = export_execution_lite_rows(tasks)
    assert rows
    assert rows[0]["function_name"] == tasks[0].entry_point
    assert rows[0]["prompt"] == tasks[0].starter_code


def test_load_assertion_format_jsonl(tmp_path):
    p = tmp_path / "ace_assert.jsonl"
    row = {
        "task_id": "t1",
        "source": "test",
        "split": "train",
        "instruction": "Add two numbers.",
        "starter_code": "def add(a, b):",
        "entry_point": "add",
        "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        "test_format": "assertion",
        "canonical_prompt": "dummy",
    }
    p.write_text(json.dumps(row) + "\n")
    tasks = load_code_tasks(p)
    assert len(tasks) == 1
    assert tasks[0].test_format == "assertion"
    assert tasks[0].tests == ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
