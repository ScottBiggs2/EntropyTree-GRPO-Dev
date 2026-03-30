"""Golden-string tests for DiffuCoder-aligned eval prompts and code extraction."""

from __future__ import annotations

import pytest

from dream.src.eval_prompts import (
    build_humaneval_prompt,
    build_mbpp_prompt,
    extract_diffucoder_completion,
    normalize_humaneval_evalplus_completion,
)


def test_humaneval_prompt_matches_paper_table5_shape() -> None:
    # Fixed stub — character-level template must stay stable for literature parity.
    stub = "def add(a, b):\n    \"\"\"Return a + b.\"\"\""
    got = build_humaneval_prompt(stub)
    expected = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Please complete the following problem:\n"
        "```\n"
        f"{stub}\n"
        "```\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Here is the code to solve this problem:\n"
        "```python\n"
    )
    assert got == expected


def test_mbpp_prompt_matches_paper_table6_shape() -> None:
    desc = "Write a function that returns the sum of two integers."
    got = build_mbpp_prompt(desc)
    expected = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "```\n"
        f"{desc}\n"
        "```\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Here is the code to solve this problem:\n"
        "```python\n"
    )
    assert got == expected


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        (
            "    return a + b\n",
            "    return a + b",
        ),
        (
            "```python\n    return a + b\n```",
            "    return a + b",
        ),
        (
            "```python\n    return a + b\n```\n<|im_end|>",
            "    return a + b",
        ),
        (
            "    return a + b\n<|dlm_pad|>",
            "    return a + b",
        ),
        (
            "```python\ndef foo():\n    pass\n```",
            "def foo():\n    pass",
        ),
        # MDLM fixed-length padding after closing fence (must not reach JSONL / EvalPlus)
        (
            "    return a + b\n```<|endoftext|><|endoftext|><|endoftext|>",
            "    return a + b",
        ),
    ],
)
def test_extract_diffucoder_completion(raw: str, want: str) -> None:
    assert extract_diffucoder_completion(raw) == want


def test_extract_empty() -> None:
    assert extract_diffucoder_completion("") == ""


def test_normalize_humaneval_strips_full_def_for_evalplus_merge() -> None:
    """EvalPlus uses prompt + completion; full-function model output must become body-only."""
    src = "def foo(x):\n    return x + 1\n"
    got = normalize_humaneval_evalplus_completion(src, "foo")
    assert "def foo" not in got
    assert "return x + 1" in got
    assert got.splitlines()[0].startswith("    ")


def test_normalize_humaneval_body_only_passthrough() -> None:
    """Already-indented body (no parseable def) is unchanged."""
    src = "    return 42\n"
    assert normalize_humaneval_evalplus_completion(src, "foo") == src
