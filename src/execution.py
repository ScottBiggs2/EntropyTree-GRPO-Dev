"""Phase 8.5: Sandboxed execution for execution-lite reward."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default timeout for running one completion's tests (seconds)
EXECUTION_TIMEOUT = 2

# Path to the sandbox runner script (relative to repo root)
RUNNER_SCRIPT = "scripts/run_execution_sandbox.py"


def _runner_script_path() -> Path:
    """Resolve runner script path from repo root (where we run experiments)."""
    # When running as python scripts/run_experiment_2.py, cwd is typically repo root
    root = Path.cwd()
    path = root / RUNNER_SCRIPT
    if not path.exists():
        # Try relative to this file
        root = Path(__file__).resolve().parents[1]
        path = root / RUNNER_SCRIPT
    return path


def run_tests(
    prompt: str,
    completion: str,
    function_name: str,
    tests: List[Any],
    timeout: float = EXECUTION_TIMEOUT,
) -> float:
    """
    Run completion in a subprocess with the given tests.
    Returns fraction of tests passed in [0, 1]. Returns 0.0 on timeout, crash, or missing function.
    tests: list of [arg1, ..., expected] (single-arg: [arg, expected]; multi-arg: [a, b, expected]).
    """
    if not tests:
        return 0.0
    code = (prompt + "\n" + completion).strip()
    if not code.strip():
        return 0.0
    # Normalize to list of lists for JSON
    tests_serializable = [t if isinstance(t, list) else list(t) for t in tests]
    config = {"function_name": function_name, "tests": tests_serializable}
    runner = _runner_script_path()
    if not runner.exists():
        return 0.0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            config_path = f.name
        try:
            result = subprocess.run(
                [os.environ.get("PYTHON", "python"), str(runner), config_path],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd(),
            )
            out = (result.stdout or "").strip()
            if result.returncode != 0 or not out:
                return 0.0
            return float(out)
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


def load_registry(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load prompt -> { function_name, tests } registry from JSON file.
    JSON format: list of { "prompt": "...", "function_name": "...", "tests": [[arg, expected], ...] }
    Returns dict keyed by prompt (strip) for lookup.
    """
    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "execution_lite.json"
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    registry = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = (item.get("prompt") or "").strip()
        func = item.get("function_name")
        tests = item.get("tests", [])
        if prompt and func and tests:
            is_eval = bool(item.get("eval", False))
            registry[prompt] = {"function_name": func, "tests": tests, "eval": is_eval}
    return registry


def get_train_prompts_from_registry(path: Optional[str] = None) -> List[str]:
    """Return list of training prompts (entries with eval != true) in registry order."""
    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "execution_lite.json"
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item.get("prompt", "")).strip() for item in data if item.get("prompt") and not item.get("eval", False)]


def get_eval_prompts_from_registry(path: Optional[str] = None) -> List[str]:
    """Return list of eval prompts (entries with \"eval\": true) for held-out evaluation."""
    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "execution_lite.json"
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item.get("prompt", "")).strip() for item in data if item.get("prompt") and item.get("eval", False)]


def get_prompts_from_registry(path: Optional[str] = None) -> List[str]:
    """Return list of all prompts in registry order. For training use get_train_prompts_from_registry."""
    if path is None:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "execution_lite.json"
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item.get("prompt", "")).strip() for item in data if item.get("prompt")]
