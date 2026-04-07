"""Write `eval_run_manifest.json` for reproducibility (cluster eval jobs)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_eval_run_manifest(**fields: Any) -> Dict[str, Any]:
    """Merge caller fields with environment and library versions."""
    commit: Optional[str] = None
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass

    m: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "git_commit": commit,
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "evalplus": _pkg_version("evalplus"),
        "bigcodebench": _pkg_version("bigcodebench"),
        "datasets": _pkg_version("datasets"),
    }
    m.update({k: v for k, v in fields.items() if v is not None})
    return m


def write_eval_run_manifest(path: Path | str, **fields: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_eval_run_manifest(**fields)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_eval_run_manifest_from_env() -> None:
    """CLI helper: reads `OUT_BASE` (required) and common env vars; writes `eval_run_manifest.json`."""
    out = os.environ.get("OUT_BASE") or os.environ.get("OUT_DIR")
    if not out:
        print("ERROR: set OUT_BASE (or OUT_DIR) for eval manifest", file=sys.stderr)
        sys.exit(1)
    out_base = Path(out)
    adapter = os.environ.get("ADAPTER") or None
    fields: Dict[str, Any] = {
        "model": os.environ.get("MODEL"),
        "adapter": adapter,
        "out_base": str(out_base),
        "max_new_tokens": os.environ.get("MAX_NEW_TOKENS"),
        "steps": os.environ.get("STEPS"),
        "max_tasks": os.environ.get("MAX_TASKS"),
        "run_humaneval": os.environ.get("RUN_HUMANEVAL"),
        "run_mbpp": os.environ.get("RUN_MBPP"),
        "run_pass10": os.environ.get("RUN_PASS10"),
        "run_gsm8k": os.environ.get("RUN_GSM8K"),
        "run_bigcodebench": os.environ.get("RUN_BIGCODEBENCH"),
        "bcb_subset": os.environ.get("BCB_SUBSET"),
        "bcb_execution": os.environ.get("BCB_EXECUTION"),
        "gsm8k_max_tasks": os.environ.get("GSM8K_MAX_TASKS"),
        "evalplus_backend": os.environ.get("EVALPLUS_BACKEND"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME"),
        "temp_pass1": os.environ.get("TEMP_PASS1"),
        "temp_pass10": os.environ.get("TEMP_PASS10"),
    }
    write_eval_run_manifest(out_base / "eval_run_manifest.json", **fields)


if __name__ == "__main__":
    write_eval_run_manifest_from_env()
