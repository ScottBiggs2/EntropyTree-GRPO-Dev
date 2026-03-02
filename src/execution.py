"""Phase 8.5: Sandboxed execution for execution-lite reward."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default timeout for running one completion's tests (seconds)
EXECUTION_TIMEOUT = 2

RUNNER_SCRIPT_NAME = "run_execution_sandbox.py"


def _runner_script_path(project_root: Optional[Path] = None) -> Path:
    """Resolve path to the sandbox runner script. Prefer project_root if provided (so cwd doesn't matter)."""
    if project_root is not None:
        p = project_root / "scripts" / RUNNER_SCRIPT_NAME
        if p.exists():
            return p
    # Fallback: cwd then relative to this file
    root = Path.cwd()
    p = root / "scripts" / RUNNER_SCRIPT_NAME
    if p.exists():
        return p
    root = Path(__file__).resolve().parents[1]
    return root / "scripts" / RUNNER_SCRIPT_NAME


def run_tests(
    prompt: str,
    completion: str,
    function_name: str,
    tests: List[Any],
    timeout: float = EXECUTION_TIMEOUT,
    project_root: Optional[Path] = None,
) -> float:
    """
    Run completion in a subprocess with the given tests.
    Returns fraction of tests passed in [0, 1]. Returns 0.0 on timeout, crash, or missing function.
    project_root: if set, used to find scripts/run_execution_sandbox.py so cwd doesn't matter.
    """
    if not tests:
        return 0.0
    code = (prompt + "\n" + completion).strip()
    if not code.strip():
        return 0.0
    tests_serializable = [t if isinstance(t, list) else list(t) for t in tests]
    config = {"function_name": function_name, "tests": tests_serializable}
    runner = _runner_script_path(project_root)
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
                cwd=str(project_root) if project_root is not None else None,
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
