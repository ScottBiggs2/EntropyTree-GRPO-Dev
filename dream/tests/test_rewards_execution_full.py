import json
from pathlib import Path

from dream.src.rewards import (
    CodeFormatReward,
    ExecutionReward,
    ExecutionShapedReward,
    build_reward_function,
)
from dream.src.task_registry import load_code_tasks


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dream" / "data" / "code_grpo_train.sample.jsonl"


def test_code_format_reward_scores_parseable_code():
    reward = CodeFormatReward()
    score = reward("def foo(x):\n    return x + 1", "unused")
    assert score >= 1.0


def test_execution_reward_scores_known_good_completion():
    task = load_code_tasks(DATASET)[0]
    reward = ExecutionReward(registry_path=str(DATASET), project_root=ROOT)
    completion = "    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"
    score = reward(completion, task.canonical_prompt)
    assert score >= 1.0


def test_execution_shaped_reward_handles_fenced_code():
    task = load_code_tasks(DATASET)[1]
    reward = ExecutionShapedReward(registry_path=str(DATASET), project_root=ROOT)
    completion = "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"
    score = reward(completion, task.canonical_prompt)
    assert score >= 1.0


def test_build_reward_function_dispatch():
    reward = build_reward_function(
        "execution_shaped",
        registry_path=str(DATASET),
        project_root=ROOT,
    )
    assert isinstance(reward, ExecutionShapedReward)


def test_execution_shaped_reward_assertion_format(tmp_path):
    row = {
        "task_id": "assert_add",
        "source": "test",
        "split": "train",
        "instruction": "Implement add.",
        "starter_code": "def add(a, b):",
        "entry_point": "add",
        "tests": ["assert add(1, 2) == 3"],
        "test_format": "assertion",
        "canonical_prompt": "dummy",
    }
    p = tmp_path / "assert.jsonl"
    p.write_text(json.dumps(row) + "\n")
    task = load_code_tasks(p)[0]
    reward = ExecutionShapedReward(registry_path=str(p), project_root=ROOT)
    completion = "    return a + b"
    score = reward(completion, task.canonical_prompt)
    assert score >= 1.0
