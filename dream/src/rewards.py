"""Reward functions for Dream stack GRPO on code.

This mirrors the parent project's reward module but lives in the Dream
subdirectory so Dream-specific trainers can depend only on dream.src.*.
"""

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from src.execution import load_registry, run_tests  # reuse existing sandbox


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, completion: str, prompt: str) -> float:
        raise NotImplementedError


class SyntaxReward(RewardFunction):
    """Heuristic syntax/structure reward for development.

    This is identical in spirit to the parent SyntaxReward and is
    suitable for fast local debugging before wiring execution-based
    rewards or LLM-as-a-judge.
    """

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
    """Sandboxed execution against a prompt→test registry (ExecutionLite).

    Reward = fraction of tests passed (primary signal) plus lightweight
    text-based shaping bonuses to provide variance when tests fail.
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


class LLMEvalReward(RewardFunction):
    """Placeholder for LLM-as-a-judge evaluation.

    Design goal:
    - Optional wrapper that queries an external judge model for a
      scalar score in [0, 1] describing correctness / style.
    - Must always be *optional* and paired with a concrete fallback
      (SyntaxReward / ExecutionLiteReward) to support debugging and
      reproducibility without external services.

    The concrete implementation should live in a separate script or
    be injected as a callable, to avoid hard coding network access
    into the core training loop.
    """

    def __call__(self, completion: str, prompt: str) -> float:
        # For now, this acts as a no-op; real judge integration should
        # be implemented in cloud environments where an external LLM
        # is available.
        return 0.0

