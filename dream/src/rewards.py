"""Reward functions for Dream stack GRPO on code."""

import ast
from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path
from typing import Dict, Optional

from dream.src.execution_backends import ExecutionBackend, SubprocessBackend
from dream.src.formatting import normalize_completion_for_reward
from dream.src.task_registry import build_prompt_lookup, load_code_tasks

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXECUTION_PATH = _REPO_ROOT / "src" / "execution.py"
_EXECUTION_SPEC = importlib.util.spec_from_file_location(
    "entropy_tree_src_execution", _EXECUTION_PATH
)
if _EXECUTION_SPEC is None or _EXECUTION_SPEC.loader is None:
    raise ImportError(f"Unable to load execution helpers from {_EXECUTION_PATH}")
_EXECUTION_MODULE = importlib.util.module_from_spec(_EXECUTION_SPEC)
_EXECUTION_SPEC.loader.exec_module(_EXECUTION_MODULE)
load_registry = _EXECUTION_MODULE.load_registry
run_tests = _EXECUTION_MODULE.run_tests


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, completion: str, prompt: str) -> float:
        raise NotImplementedError


class CodeFormatReward(RewardFunction):
    """Reward code-like outputs that are extractable and parseable."""

    def __call__(self, completion: str, prompt: str) -> float:
        del prompt
        code = normalize_completion_for_reward(completion)
        if not code:
            return 0.0
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            if "def " in code or "return " in code:
                return 0.5
            return 0.0


class SyntaxReward(RewardFunction):
    """Heuristic syntax/structure reward for development.

    This is identical in spirit to the parent SyntaxReward and is
    suitable for fast local debugging before wiring execution-based
    rewards or LLM-as-a-judge.
    """

    def __call__(self, completion: str, prompt: str) -> float:
        del prompt
        completion = normalize_completion_for_reward(completion) or completion
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


class _TaskBackedReward(RewardFunction):
    def __init__(
        self,
        registry_path: Optional[str] = None,
        timeout: float = 2.0,
        project_root: Optional[Path] = None,
        backend: Optional[ExecutionBackend] = None,
    ):
        self.registry_path = registry_path
        self.timeout = timeout
        self.project_root = Path(project_root) if project_root is not None else None
        self.backend = backend
        self._task_lookup: Optional[Dict] = None

    def _get_task_lookup(self) -> Dict:
        if self._task_lookup is not None:
            return self._task_lookup
        if self.registry_path:
            try:
                tasks = load_code_tasks(self.registry_path)
                self._task_lookup = build_prompt_lookup(tasks)
                return self._task_lookup
            except Exception:
                pass
        registry = load_registry(self.registry_path)
        self._task_lookup = {}
        for key, value in registry.items():
            self._task_lookup[key.strip()] = {
                "entry_point": value["function_name"],
                "tests": value["tests"],
                "starter_code": key.strip(),
            }
        return self._task_lookup

    def _lookup_task(self, prompt: str):
        prompt_key = (prompt or "").strip()
        return self._get_task_lookup().get(prompt_key)

    def _normalized_code(self, completion: str, prompt: str) -> str:
        task = self._lookup_task(prompt)
        entry_point = None
        if task is not None:
            entry_point = getattr(task, "entry_point", None) or task.get("entry_point")
        return normalize_completion_for_reward(completion, entry_point=entry_point)


class ExecutionReward(_TaskBackedReward):
    """Pure execution reward: fraction of tests passed."""

    def _run(self, code: str, starter_code: str, func_name: str,
             tests: list, test_format: str) -> float:
        if self.backend is not None:
            return self.backend.run_tests(
                code=code,
                function_name=func_name,
                tests=tests,
                starter_code=starter_code,
                timeout=self.timeout,
                test_format=test_format,
            )
        return run_tests(
            prompt=starter_code,
            completion=code,
            function_name=func_name,
            tests=tests,
            timeout=self.timeout,
            project_root=self.project_root,
            test_format=test_format,
        )

    def __call__(self, completion: str, prompt: str) -> float:
        if not completion.strip():
            return 0.0
        task = self._lookup_task(prompt)
        if task is None:
            return 0.0
        code = self._normalized_code(completion, prompt)
        if not code:
            return 0.0
        starter_code = getattr(task, "starter_code", None) or task.get("starter_code", "")
        func_name = getattr(task, "entry_point", None) or task.get("entry_point")
        tests = getattr(task, "tests", None) or task.get("tests", [])
        test_format = getattr(task, "test_format", None) or task.get(
            "test_format", "args_expected"
        )
        return self._run(code, starter_code, func_name, tests, test_format)


class ExecutionShapedReward(ExecutionReward):
    """Execution reward plus lightweight shaping when tests fail."""

    TESTS_WEIGHT = 0.80

    def __init__(
        self,
        registry_path: Optional[str] = None,
        timeout: float = 2.0,
        project_root: Optional[Path] = None,
        backend: Optional[ExecutionBackend] = None,
        *,
        tie_breaker: str = "none",
    ):
        super().__init__(
            registry_path=registry_path,
            timeout=timeout,
            project_root=project_root,
            backend=backend,
        )
        self.tie_breaker = (tie_breaker or "none").strip().lower()

    def score_components(self, completion: str, prompt: str) -> Dict[str, float]:
        """Return raw execution fraction, shaping bonus, and final shaped reward."""
        task = self._lookup_task(prompt)
        if task is None:
            return {"exec_frac": 0.0, "shaping_bonus": 0.0, "reward": 0.0}
        code = self._normalized_code(completion, prompt)
        if not code:
            return {"exec_frac": 0.0, "shaping_bonus": 0.0, "reward": 0.0}
        starter_code = getattr(task, "starter_code", None) or task.get("starter_code", "")
        func_name = getattr(task, "entry_point", None) or task.get("entry_point")
        tests = getattr(task, "tests", None) or task.get("tests", [])
        test_format = getattr(task, "test_format", None) or task.get(
            "test_format", "args_expected"
        )
        frac = float(self._run(code, starter_code, func_name, tests, test_format))
        if frac >= 1.0:
            return {"exec_frac": 1.0, "shaping_bonus": 0.0, "reward": 1.0}
        bonus = float(self._shaping_bonus(code, func_name))
        reward = float(frac * self.TESTS_WEIGHT + bonus)
        # Optional: break ties among many exec_frac==0 samples without changing
        # the core reward unless explicitly enabled.
        if self.tie_breaker != "none":
            reward += float(self._tie_break_bonus(code))
        reward = min(1.0, reward)
        return {"exec_frac": frac, "shaping_bonus": bonus, "reward": float(reward)}

    def __call__(self, completion: str, prompt: str) -> float:
        return float(self.score_components(completion, prompt)["reward"])

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

    def _tie_break_bonus(self, code: str) -> float:
        """Small continuous bonus to reduce reward ties (default-off)."""
        key = self.tie_breaker
        if key == "ast_size":
            try:
                tree = ast.parse(code)
                n = sum(1 for _ in ast.walk(tree))
                # Log-scaled, tiny magnitude to avoid overpowering exec_frac.
                # Typical: 0.0–0.01 range.
                return float(min(0.01, 0.002 * (1.0 + (n / 50.0)) ** 0.5))
            except SyntaxError:
                return 0.0
        if key == "code_len":
            n = len(code)
            return float(min(0.01, 0.00002 * n))
        return 0.0


class ExecutionLiteReward(ExecutionShapedReward):
    """Backward-compatible alias for the current execution-lite style reward."""


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


def build_reward_function(
    name: str,
    *,
    registry_path: Optional[str] = None,
    timeout: float = 2.0,
    project_root: Optional[Path] = None,
    backend: Optional[ExecutionBackend] = None,
    tie_breaker: str = "none",
) -> RewardFunction:
    key = name.strip().lower()
    if key == "syntax":
        return SyntaxReward()
    if key == "code_format":
        return CodeFormatReward()
    if key == "execution":
        return ExecutionReward(
            registry_path=registry_path,
            timeout=timeout,
            project_root=project_root,
            backend=backend,
        )
    if key in ("execution_shaped", "execution_lite"):
        return ExecutionShapedReward(
            registry_path=registry_path,
            timeout=timeout,
            project_root=project_root,
            backend=backend,
            tie_breaker=tie_breaker,
        )
    raise ValueError(f"Unknown reward function: {name}")

