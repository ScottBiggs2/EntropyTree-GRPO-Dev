"""Phase 5: Reward function tests."""

import pytest
from src.rewards import SyntaxReward, ExecutionReward


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
