#!/usr/bin/env python3
"""Generate MBPP+ completions with DiffuCoder-aligned prompts; optional EvalPlus scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def main() -> int:
    from dream.src.eval_generate import (
        _configure_hf_cache,
        generate_completions,
        load_dream_model_for_eval,
        load_tasks_from_jsonl,
        run_evalplus,
        write_evalplus_jsonl,
    )
    from dream.src.eval_prompts import build_mbpp_prompt
    from dream.src.task_registry import CodeTask

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B", help="HF model id or path")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter directory")
    parser.add_argument("--dataset", required=True, type=Path, help="mbpp.jsonl (CodeTask rows)")
    parser.add_argument("--output", required=True, type=Path, help="Output EvalPlus JSONL path")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per task (1=pass@1, 10=pass@10)")
    parser.add_argument("--temperature", type=float, default=0.2, help="0.2 pass@1; 0.4 pass@10")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all tasks")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument(
        "--run-evalplus",
        action="store_true",
        help="After generation, run evalplus.evaluate on the JSONL",
    )
    args = parser.parse_args()

    _configure_hf_cache()

    tasks = load_tasks_from_jsonl(args.dataset, max_tasks=args.max_tasks)
    if not tasks:
        print("ERROR: no tasks loaded.", file=sys.stderr)
        return 1

    print(f"Loading model {args.model!r}...")
    model, tokenizer = load_dream_model_for_eval(
        args.model,
        device=args.device,
        adapter_path=args.adapter,
    )

    def slot_from_task(t: CodeTask) -> str:
        # DiffuCoder MBPP: fenced {prompt} is the task description text
        return t.instruction

    rows = generate_completions(
        model,
        tokenizer,
        tasks,
        prompt_builder=build_mbpp_prompt,
        slot_from_task=slot_from_task,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        device=args.device,
    )

    write_evalplus_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} lines to {args.output}")

    if args.run_evalplus:
        rc = run_evalplus(args.output, dataset="mbpp")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
