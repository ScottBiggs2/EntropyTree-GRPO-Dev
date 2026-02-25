"""Reward functions for GRPO (Phase 5). D-008: heuristic for dev, execution for experiments. Phase 8.5: ExecutionLiteReward."""

import ast
from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.execution import load_registry, run_tests


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


class ExecutionLiteReward(RewardFunction):
    """Phase 8.5: Sandboxed execution against a prompt→test registry; reward = fraction of tests passed.
    Optional small syntax bonus if AST-parseable and not all tests passed (gradient signal)."""

    def __init__(
        self,
        registry_path: Optional[str] = None,
        syntax_bonus: float = 0.05,
        timeout: float = 2.0,
    ):
        self.registry_path = registry_path
        self.syntax_bonus = syntax_bonus
        self.timeout = timeout
        self._registry: Optional[Dict] = None

    def _get_registry(self) -> Dict:
        if self._registry is None:
            self._registry = load_registry(self.registry_path)
        return self._registry

    def __call__(self, completion: str, prompt: str) -> float:
        if not completion.strip():
            return 0.0
        key = prompt.strip()
        registry = self._get_registry()
        if key not in registry:
            return 0.0
        entry = registry[key]
        func_name = entry["function_name"]
        tests = entry["tests"]
        frac = run_tests(
            prompt=prompt,
            completion=completion,
            function_name=func_name,
            tests=tests,
            timeout=self.timeout,
        )
        if frac >= 1.0:
            return 1.0
        try:
            ast.parse(completion)
            frac = min(1.0, frac + self.syntax_bonus)
        except SyntaxError:
            pass
        return frac


class ExecutionReward(RewardFunction):
    """Placeholder for Phase 8+: run tests in sandbox, return fraction passed."""

    def __call__(self, completion: str, prompt: str) -> float:
        # TODO: sandbox execution, test cases
        return 0.0
