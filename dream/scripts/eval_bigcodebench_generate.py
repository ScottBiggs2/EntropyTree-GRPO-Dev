#!/usr/bin/env python3
"""Generate BigCodeBench-Instruct samples with Dream ``diffusion_generate``; optional official scoring."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


@dataclass(frozen=True)
class _BCBTask:
    task_id: str
    instruct: str


def main() -> int:
    from dream.src.eval_bigcodebench_runner import run_bigcodebench_evaluate
    from dream.src.eval_generate import (
        _configure_hf_cache,
        generate_completions,
        load_dream_model_for_eval,
    )
    from dream.src.eval_prompts import build_mbpp_prompt, extract_diffucoder_completion

    try:
        from bigcodebench.data import get_bigcodebench
    except ImportError as e:
        print("ERROR: pip install bigcodebench", file=sys.stderr)
        raise SystemExit(1) from e

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--output", type=Path, required=True, help="Output JSONL (BigCodeBench samples schema)")
    p.add_argument("--subset", choices=("full", "hard"), default="hard")
    p.add_argument("--max-tasks", type=int, default=0, help="0 = all tasks in subset")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--run-evaluate",
        action="store_true",
        help="After generation, run bigcodebench.evaluate on the JSONL",
    )
    p.add_argument(
        "--execution",
        default="local",
        help="BigCodeBench execution backend (local, gradio, e2b)",
    )
    args = p.parse_args()

    _configure_hf_cache()

    raw: Dict[str, Dict[str, Any]] = get_bigcodebench(subset=args.subset)
    task_ids = sorted(raw.keys(), key=lambda x: (len(x), x))
    tasks: List[_BCBTask] = []
    for tid in task_ids:
        row = raw[tid]
        ip = row.get("instruct_prompt") or ""
        tasks.append(_BCBTask(task_id=tid, instruct=ip))
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    print(f"Loading model {args.model!r} ({len(tasks)} tasks, subset={args.subset})...")
    model, tokenizer = load_dream_model_for_eval(
        args.model, device=args.device, adapter_path=args.adapter
    )

    def _post(_t: _BCBTask, completion: str) -> str:
        return extract_diffucoder_completion(completion)

    rows_list: List[Tuple[str, str]] = generate_completions(
        model,
        tokenizer,
        tasks,
        prompt_builder=build_mbpp_prompt,
        slot_from_task=lambda t: t.instruct,
        n_samples=1,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        completion_postprocess=_post,
        quiet=args.quiet,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for task_id, sol in rows_list:
            rec = {
                "task_id": task_id,
                "solution": sol,
                "raw_solution": sol,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows_list)} lines to {args.output}")

    if args.run_evaluate:
        rc = run_bigcodebench_evaluate(
            args.output,
            split="instruct",
            subset=args.subset,
            execution=args.execution,
        )
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
