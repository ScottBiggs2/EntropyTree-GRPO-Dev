#!/usr/bin/env python3
"""Check Dream code-reward plumbing on a task dataset.

This is a light local/HPC sanity script. It does not load Dream weights.
It verifies that:

- tasks load,
- prompts resolve to tasks,
- reward functions can execute,
- known-good completions score above known-bad completions where we have canned solutions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dream.src.rewards import build_reward_function
from dream.src.task_registry import filter_code_tasks, infer_default_split, load_code_tasks


KNOWN_GOOD = {
    "fibonacci": "    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "factorial": "    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "sum_list": "    return sum(lst)",
    "is_prime": "    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "reverse_list": "    return list(reversed(lst))",
}

KNOWN_BAD = {
    "generic": "    return None",
}


def main() -> int:
    p = argparse.ArgumentParser(description="Check Dream reward pipeline on a task dataset")
    p.add_argument("--dataset", type=str, required=True, help="Task dataset path")
    p.add_argument(
        "--dataset-split",
        type=str,
        default="",
        help="Task split to use (train/dev/all). Default inferred.",
    )
    p.add_argument(
        "--reward",
        type=str,
        default="execution_shaped",
        choices=("syntax", "code_format", "execution", "execution_shaped", "execution_lite"),
    )
    p.add_argument("--backend", type=str, default="local", help="Reserved for future backends")
    p.add_argument("--max-tasks", type=int, default=5)
    p.add_argument("--timeout", type=float, default=2.0)
    args = p.parse_args()

    if args.backend != "local":
        print(f"WARNING: backend={args.backend!r} not yet implemented; using local execution path.")

    tasks = load_code_tasks(args.dataset)
    split = args.dataset_split or infer_default_split(tasks, "baseline_train")
    tasks = filter_code_tasks(tasks, split)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        print("ERROR: no tasks found after split filtering.")
        return 1

    reward_fn = build_reward_function(
        args.reward,
        registry_path=args.dataset,
        timeout=args.timeout,
        project_root=_repo_root,
    )

    print(f"[reward_check] dataset={args.dataset} split={split} reward={args.reward} n_tasks={len(tasks)}")
    n_good = 0
    for task in tasks:
        good = KNOWN_GOOD.get(task.entry_point)
        bad = KNOWN_BAD["generic"]
        prompt = task.canonical_prompt
        bad_score = reward_fn(bad, prompt)
        if good is not None:
            good_score = reward_fn(good, prompt)
            print(
                f"[reward_check] task={task.task_id} entry={task.entry_point} "
                f"bad={bad_score:.3f} good={good_score:.3f}"
            )
            if good_score > bad_score:
                n_good += 1
        else:
            print(
                f"[reward_check] task={task.task_id} entry={task.entry_point} "
                f"bad={bad_score:.3f} good=SKIP(no canned solution)"
            )

    print(f"[reward_check] tasks_with_good_gt_bad={n_good}")
    if n_good == 0 and args.reward in ("execution", "execution_shaped", "execution_lite"):
        print("ERROR: no canned known-good completion outscored the bad completion.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
