"""Pluggable execution backends for GRPO code rewards.

See dream/PLAN_03_ENVIRONMENT_SCALEUP.md Step 5.

Two implementations:
  SubprocessBackend  — wraps the existing src/execution.run_tests() subprocess path.
  ContainerBackend   — runs code inside Docker (local) or Apptainer (HPC).

Usage::

    from dream.src.execution_backends import SubprocessBackend, ContainerBackend

    backend = ContainerBackend(image="dream-sandbox:latest", runtime="docker")
    score = backend.run_tests(code="def add(a,b): return a+b",
                              function_name="add",
                              tests=["assert add(1,2)==3"],
                              test_format="assertion")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional


class ExecutionBackend(ABC):
    """ABC for code execution backends used by reward functions."""

    @abstractmethod
    def run_tests(
        self,
        code: str,
        function_name: str,
        tests: List[Any],
        starter_code: str = "",
        timeout: float = 5.0,
        test_format: str = "args_expected",
    ) -> float:
        """Execute *code* against *tests* and return fraction passed in [0, 1]."""
        ...


class SubprocessBackend(ExecutionBackend):
    """Delegates to the existing ``src/execution.run_tests()`` subprocess runner.

    This is a thin wrapper that preserves backward compatibility with the
    original execution path used before containerised execution was available.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self._project_root = project_root

    def run_tests(
        self,
        code: str,
        function_name: str,
        tests: List[Any],
        starter_code: str = "",
        timeout: float = 5.0,
        test_format: str = "args_expected",
    ) -> float:
        import importlib.util

        repo_root = self._project_root or Path(__file__).resolve().parents[2]
        exec_path = repo_root / "src" / "execution.py"
        spec = importlib.util.spec_from_file_location("_exec", exec_path)
        if spec is None or spec.loader is None:
            return 0.0
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_tests(
            prompt=starter_code,
            completion=code,
            function_name=function_name,
            tests=tests,
            timeout=timeout,
            project_root=repo_root,
            test_format=test_format,
        )


class ContainerBackend(ExecutionBackend):
    """Runs code inside a container (Docker or Apptainer).

    Parameters
    ----------
    image : str
        Docker image name (e.g. ``"dream-sandbox:latest"``) or path to an
        Apptainer ``.sif`` file.
    runtime : str
        ``"docker"`` or ``"apptainer"`` (auto-detected from *image* suffix
        if not given explicitly).
    timeout : float
        Default per-invocation wall-clock timeout (seconds).
    memory_limit : str
        Docker ``--memory`` value; ignored for Apptainer.
    """

    def __init__(
        self,
        image: str = "dream-sandbox:latest",
        runtime: str | None = None,
        timeout: float = 10.0,
        memory_limit: str = "256m",
    ):
        if runtime is None:
            runtime = "apptainer" if image.endswith(".sif") else "docker"
        self.image = image
        self.runtime = runtime
        self.default_timeout = timeout
        self.memory_limit = memory_limit
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        if self.runtime not in ("docker", "apptainer"):
            raise ValueError(f"Unknown runtime: {self.runtime!r}")
        if shutil.which(self.runtime) is None:
            raise FileNotFoundError(
                f"{self.runtime!r} not found on PATH. "
                f"Install it or use SubprocessBackend as a fallback."
            )

    def run_tests(
        self,
        code: str,
        function_name: str,
        tests: List[Any],
        starter_code: str = "",
        timeout: float = 5.0,
        test_format: str = "args_expected",
    ) -> float:
        if not code or not code.strip():
            return 0.0
        if not tests:
            return 0.0

        full_code = (starter_code + "\n" + code).strip() if starter_code else code.strip()

        if test_format == "assertion":
            payload: dict[str, Any] = {
                "code": full_code,
                "test_cases": [str(t) for t in tests],
                "test_format": "assertion",
            }
        else:
            payload = {
                "code": full_code,
                "function_name": function_name,
                "tests": [t if isinstance(t, list) else list(t) for t in tests],
                "test_format": "args_expected",
            }

        wall_timeout = max(timeout, self.default_timeout)

        if self.runtime == "docker":
            return self._run_docker(payload, wall_timeout)
        return self._run_apptainer(payload, wall_timeout)

    def _run_docker(self, payload: dict[str, Any], timeout: float) -> float:
        cmd = [
            "docker", "run",
            "-i", "--rm",
            f"--memory={self.memory_limit}",
            "--cpus=0.5",
            "--network=none",
            "--read-only",
            "--tmpfs", "/tmp:size=16m",
            self.image,
        ]
        return self._exec(cmd, json.dumps(payload), timeout)

    def _run_apptainer(self, payload: dict[str, Any], timeout: float) -> float:
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(
                prefix=f"dream_sandbox_r{_dist_rank()}_",
                dir=_sandbox_tmp_root(),
            )
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(payload, f)

            cmd = [
                "apptainer", "exec",
                "--contain",
                "--no-home",
                "-B", f"{tmpdir}:/task:ro",
            ]
            try:
                subprocess.run(
                    ["apptainer", "exec", "--help"],
                    capture_output=True, timeout=5,
                )
                cmd.extend(["--net", "--network", "none"])
            except Exception:
                pass

            cmd.extend([self.image, "python", "/entrypoint.py", "/task/config.json"])
            with _maybe_node_sandbox_lock():
                return self._exec(cmd, stdin_data=None, timeout=timeout)
        finally:
            if tmpdir:
                try:
                    import shutil as _sh
                    _sh.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass

    @staticmethod
    def _exec(cmd: list[str], stdin_data: str | None, timeout: float) -> float:
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (result.stdout or "").strip()
            if result.returncode != 0 or not out:
                return 0.0
            return float(out)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return 0.0


def make_backend(
    name: str = "subprocess",
    *,
    image: str = "dream-sandbox:latest",
    runtime: str | None = None,
    project_root: Optional[Path] = None,
    timeout: float = 10.0,
) -> ExecutionBackend:
    """Factory: create a backend by name.

    ``name`` is one of ``"subprocess"`` (default), ``"docker"``, ``"apptainer"``,
    or ``"container"`` (auto-detect runtime from *image*).
    """
    key = name.strip().lower()
    if key == "subprocess":
        return SubprocessBackend(project_root=project_root)
    if key in ("docker", "apptainer"):
        return ContainerBackend(image=image, runtime=key, timeout=timeout)
    if key == "container":
        return ContainerBackend(image=image, runtime=runtime, timeout=timeout)
    raise ValueError(
        f"Unknown backend {name!r}. Choose from: subprocess, docker, apptainer, container"
    )


def _dist_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0") or "0")
    except Exception:
        return 0


def _sandbox_tmp_root() -> str | None:
    """Pick a tmp root for per-rank sandbox payloads.

    Prefer scratch (fast + large) when available; fall back to default temp dir.
    """
    base = os.environ.get("DREAM_SANDBOX_TMP_ROOT") or os.environ.get("TMPDIR") or ""
    if base:
        try:
            os.makedirs(base, exist_ok=True)
            return base
        except Exception:
            return None
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    scratch_root = os.environ.get("SCRATCH") or "/scratch"
    guess = os.path.join(scratch_root, user, "dream_sandbox_tmp")
    try:
        os.makedirs(guess, exist_ok=True)
        return guess
    except Exception:
        return None


@contextmanager
def _maybe_node_sandbox_lock():
    """Optionally serialize container runs per node to avoid Apptainer/fs overload.

    Enable by setting DREAM_SANDBOX_MAX_CONCURRENT_PER_NODE=1.
    """
    if str(os.environ.get("DREAM_SANDBOX_MAX_CONCURRENT_PER_NODE", "")).strip() != "1":
        yield
        return
    # Best-effort lock file in scratch.
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    scratch_root = os.environ.get("SCRATCH") or "/scratch"
    lock_dir = os.path.join(scratch_root, user, "dream_sandbox_locks")
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except Exception:
        yield
        return
    lock_path = os.path.join(lock_dir, f"node_{os.uname().nodename}.lock")
    try:
        import fcntl

        with open(lock_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        yield
