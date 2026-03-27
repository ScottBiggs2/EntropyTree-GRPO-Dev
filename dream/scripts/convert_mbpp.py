#!/usr/bin/env python3
"""Convert MBPP+ (EvalPlus) or MBPP JSONL into Dream CodeTask JSONL.

All rows use ``split: eval``. See dream/PLAN_03_ENVIRONMENT_SCALEUP.md Step 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dream.src.eval_dataset_convert import (  # noqa: E402
    iter_jsonl,
    mbpp_to_row,
    write_jsonl,
)


def _load_tasks_evalplus() -> Dict[str, Mapping[str, Any]]:
    from evalplus.data import get_mbpp_plus
    from evalplus.data.mbpp import get_mbpp

    plus = get_mbpp_plus()

    # get_mbpp_plus() has base_input/plus_input but NOT test_list or test.
    # The original sanitized MBPP (get_mbpp()) has test_list with assertion
    # strings.  Merge them so mbpp_to_row can extract assertions.
    original = get_mbpp()
    merged = 0
    for task_id, task in plus.items():
        if task.get("test_list"):
            continue
        num = task_id.split("/")[-1] if "/" in task_id else task_id
        orig = original.get(num) or original.get(task_id)
        if orig and orig.get("test_list"):
            task["test_list"] = orig["test_list"]
            if not task.get("code") and orig.get("code"):
                task["code"] = orig["code"]
            if not task.get("text") and orig.get("text"):
                task["text"] = orig["text"]
            merged += 1

    first_key = next(iter(plus), None)
    if first_key:
        print(f"MBPP+ sample keys ({first_key}): {sorted(plus[first_key].keys())}")
    print(f"Merged test_list from sanitized MBPP for {merged}/{len(plus)} tasks")
    return plus


def _load_tasks_from_jsonl(path: Path) -> List[Mapping[str, Any]]:
    return list(iter_jsonl(path))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert MBPP / MBPP+ to Dream CodeTask JSONL"
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_repo_root / "dream" / "data" / "mbpp.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional MBPP JSONL (EvalPlus-like rows). "
        "If omitted, uses EvalPlus ``get_mbpp_plus()`` (requires evalplus).",
    )
    p.add_argument(
        "--source",
        type=str,
        default="mbpp_plus",
        help="Value for the ``source`` field on each row",
    )
    args = p.parse_args()

    rows_raw: List[Mapping[str, Any]]
    if args.input:
        rows_raw = _load_tasks_from_jsonl(Path(args.input))
    else:
        try:
            mbpp = _load_tasks_evalplus()
        except ImportError as e:
            print(
                "ERROR: evalplus is not installed and --input was not set.\n"
                "  pip install evalplus\n"
                "Or export JSONL from EvalPlus and pass --input.",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        rows_raw = [mbpp[k] for k in sorted(mbpp.keys(), key=_mbpp_sort_key)]

    out: List[Dict[str, Any]] = []
    skipped = 0
    for item in rows_raw:
        try:
            out.append(mbpp_to_row(item, source=args.source))
        except Exception as exc:
            skipped += 1
            tid = item.get("task_id", "?")
            print(f"SKIP {tid}: {exc}", file=sys.stderr)

    out_path = Path(args.output)
    n = write_jsonl(out_path, out)
    print(f"Wrote {n} tasks to {out_path} (skipped {skipped})")
    return 0 if n else 1


def _mbpp_sort_key(task_id: str) -> tuple[int, str]:
    if "/" in task_id:
        tail = task_id.split("/", 1)[1]
        if tail.isdigit():
            return (int(tail), task_id)
    return (10**9, task_id)


if __name__ == "__main__":
    raise SystemExit(main())
