"""Invoke BigCodeBench official scorer on a samples JSONL file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Union


def run_bigcodebench_evaluate(
    samples: Union[str, Path],
    *,
    split: str = "instruct",
    subset: str = "hard",
    execution: str = "local",
) -> int:
    """Run BigCodeBench's official ``evaluate`` on pre-generated samples."""
    samples = Path(samples)
    cmd_sets = [
        [
            "bigcodebench.evaluate",
            "--split",
            split,
            "--subset",
            subset,
            "--samples",
            str(samples),
            "--execution",
            execution,
        ],
        [
            sys.executable,
            "-m",
            "bigcodebench.evaluate",
            "--split",
            split,
            "--subset",
            subset,
            "--samples",
            str(samples),
            "--execution",
            execution,
        ],
    ]
    last_err: Optional[BaseException] = None
    for cmd in cmd_sets:
        try:
            proc = subprocess.run(cmd)
            return int(proc.returncode)
        except FileNotFoundError as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not run bigcodebench.evaluate (install with `pip install bigcodebench`)."
    ) from last_err
