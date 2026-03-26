#!/usr/bin/env python3
"""
Print canonical_prompt for one task row (same construction as training).

Use when an older single_step_dream.py has no --dataset: pass the printed line
to --prompt. For execution_shaped rewards you still need the updated reward
stack; see dream/HPC_SYNC.md.

Usage (repo root):
  python dream/scripts/extract_dataset_prompt.py dream/data/code_grpo_train.sample.jsonl 0 --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def main() -> int:
    p = argparse.ArgumentParser(description="Extract canonical_prompt from a task JSONL")
    p.add_argument("dataset", type=str, help="Path to .jsonl")
    p.add_argument("task_index", type=int, help="0-based index within the filtered split")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split filter: train, dev, or all",
    )
    args = p.parse_args()

    from dream.src.task_registry import filter_code_tasks, infer_default_split, load_code_tasks

    path = Path(args.dataset)
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    tasks = load_code_tasks(str(path))
    chosen = args.split if args.split.lower() != "all" else ""
    split_key = chosen or infer_default_split(tasks, "baseline_train")
    tasks = filter_code_tasks(tasks, split_key)

    if args.task_index < 0 or args.task_index >= len(tasks):
        print(
            f"error: task_index {args.task_index} out of range (n={len(tasks)} after split filter)",
            file=sys.stderr,
        )
        return 1

    prompt = tasks[args.task_index].canonical_prompt.strip()
    if not prompt:
        print("error: empty canonical_prompt", file=sys.stderr)
        return 1
    sys.stdout.write(prompt)
    if not prompt.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
