#!/usr/bin/env python3
"""GSM8K (test split) eval via Dream ``diffusion_generate`` (paper uses lm-eval-harness; see docs)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def main() -> int:
    from datasets import load_dataset

    from dream.src.eval_generate import _configure_hf_cache, load_dream_model_for_eval
    from dream.src.gsm8k_eval_utils import (
        answers_match,
        extract_predicted_answer,
        reference_final_answer,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--adapter", default=None, help="Optional PEFT adapter directory")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for gsm8k_results.json")
    p.add_argument("--max-tasks", type=int, default=0, help="0 = full test set")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _configure_hf_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("gsm8k", "main", split="test")
    rows: List[Dict[str, Any]] = list(ds)
    if args.max_tasks and args.max_tasks > 0:
        rows = rows[: args.max_tasks]

    print(f"Loading model {args.model!r} ({len(rows)} tasks)...")
    model, tokenizer = load_dream_model_for_eval(
        args.model, device=args.device, adapter_path=args.adapter
    )
    import torch

    dev = torch.device(args.device) if args.device else next(model.parameters()).device
    correct = 0
    t0 = time.perf_counter()
    details: List[Dict[str, Any]] = []

    for i, ex in enumerate(rows):
        q = ex["question"]
        ref = ex["answer"]
        gold = reference_final_answer(ref)
        messages = [{"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs["input_ids"].to(dev)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=dev)
        else:
            attention_mask = attention_mask.to(dev)
        with torch.no_grad():
            out = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg="entropy",
                alg_temp=0.0,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][input_ids.shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_ans = extract_predicted_answer(text)
        ok = answers_match(pred_ans, gold)
        if ok:
            correct += 1
        details.append(
            {
                "index": i,
                "correct": ok,
                "gold": gold,
                "predicted": pred_ans,
            }
        )
        if not args.quiet and (i % 50 == 0 or i == len(rows) - 1):
            print(
                f"[gsm8k] {i + 1}/{len(rows)} acc_so_far={correct/(i+1):.4f}",
                flush=True,
            )

    wall = time.perf_counter() - t0
    acc = correct / len(rows) if rows else 0.0
    summary: Dict[str, Any] = {
        "benchmark": "gsm8k",
        "split": "test",
        "n": len(rows),
        "correct": correct,
        "accuracy": acc,
        "wall_s": wall,
        "model": args.model,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    out_json = args.output_dir / "gsm8k_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_json} accuracy={acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
