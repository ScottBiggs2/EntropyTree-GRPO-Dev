"""Reward functions for GRPO (Phase 5). D-008: heuristic for dev, execution for experiments. Phase 8.5: ExecutionLiteReward."""

import ast
from abc import ABC, abstractmethod
from pathlib import Path
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
    Implemented via: run_tests() in src.execution (subprocess to scripts/run_execution_sandbox.py),
    registry in data/execution_lite.json.

    Dense reward shaping ensures GRPO gets gradient signal even before the model
    produces fully correct code.  Reward breakdown (approximate):
      - Tests passed fraction:  up to 0.80 (the primary signal)
      - AST-parseable:          +0.05
      - Contains "def " + "return ":  +0.05
      - Contains function name: +0.02
      - Contains "if " or "for ": +0.02
      - Indented code present:  +0.02
    Capped at 1.0.  All tests passed → exactly 1.0 (no shaping needed).
    """

    TESTS_WEIGHT = 0.80

    def __init__(
        self,
        registry_path: Optional[str] = None,
        timeout: float = 2.0,
        project_root: Optional[Path] = None,
        syntax_bonus: float = 0.05,
    ):
        self.registry_path = registry_path
        self.timeout = timeout
        self.project_root = Path(project_root) if project_root is not None else None
        self.syntax_bonus = syntax_bonus
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
            project_root=self.project_root,
        )

        if frac >= 1.0:
            return 1.0

        reward = frac * self.TESTS_WEIGHT
        reward += self._shaping_bonus(completion, func_name)
        return min(1.0, reward)

    def _shaping_bonus(self, completion: str, func_name: str) -> float:
        """Lightweight text-based bonuses that create reward variance across
        completions so GRPO has gradient signal even when no tests pass."""
        bonus = 0.0
        try:
            ast.parse(completion)
            bonus += 0.05
        except SyntaxError:
            pass
        if "def " in completion and "return " in completion:
            bonus += 0.05
        if func_name in completion:
            bonus += 0.02
        if "if " in completion or "for " in completion:
            bonus += 0.02
        if any(line.startswith(("    ", "\t")) for line in completion.splitlines()):
            bonus += 0.02
        return bonus


class ExecutionReward(RewardFunction):
    """Placeholder for Phase 8+: run tests in sandbox, return fraction passed."""

    def __call__(self, completion: str, prompt: str) -> float:
        # TODO: sandbox execution, test cases
        return 0.0
