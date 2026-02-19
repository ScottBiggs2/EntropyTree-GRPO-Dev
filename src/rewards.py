"""Reward functions for GRPO (Phase 5). D-008: heuristic for dev, execution for experiments."""

import ast
import re
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, completion: str, prompt: str) -> float:
        pass


class SyntaxReward(RewardFunction):
    """Development heuristic: AST-parseable + keywords + docstring (D-008)."""

    def __call__(self, completion: str, prompt: str) -> float:
        reward = 0.0
        if not completion.strip():
            return 0.0
        try:
            ast.parse(completion)
            reward += 0.5
        except SyntaxError:
            pass
        if "def " in completion:
            reward += 0.15
        if "return " in completion:
            reward += 0.15
        if '"""' in completion or "'''" in completion:
            reward += 0.1
        if reward >= 0.5 and "def " in completion and "return " in completion:
            reward += 0.1
        return min(1.0, reward)


class ExecutionReward(RewardFunction):
    """Placeholder for Phase 8+: run tests in sandbox, return fraction passed."""

    def __call__(self, completion: str, prompt: str) -> float:
        # TODO: sandbox execution, test cases
        return 0.0
