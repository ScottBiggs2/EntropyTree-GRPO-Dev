#!/usr/bin/env python3
"""
Light Dream 7B comparison runs with WandB (entropy-tree + LoRA).

Phases (run separately or via run_dream_comparison.sh):
  initial_eval     — tree build + SyntaxReward only, no optimizer (pre-train snapshot)
  baseline_train   — fixed-step MCTS-GRPO (adaptive_stepping=False); train N epochs
  adaptive_default — adaptive stepping + default branch_threshold / alpha_entropy
  adaptive_alt_hp  — adaptive + alternate HP (e.g. higher alpha_entropy) to sanity-check logging

WandB: by default logs flat keys (loss, avg_reward, wall_sec_step, …) like scripts/run_experiment_2.py
so runs in the same --wandb_group overlay on shared charts. Use --wandb_prefixed_keys for phase/* charts.

Usage (repo root):
  python dream/scripts/run_dream_comparison.py --phase initial_eval --wandb_project entropy-tree-grpo-dream
  python dream/scripts/run_dream_comparison.py --phase baseline_train --num_epochs 3 --run_name my_run
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _configure_hf_cache() -> None:
    if os.environ.get("HF_HOME"):
        return
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    scratch_root = os.environ.get("SCRATCH") or "/scratch"
    hf_home = os.path.join(scratch_root, user, "hf_home")
    try:
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ.setdefault(
            "TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers")
        )
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    except Exception:
        pass


_configure_hf_cache()

import torch

from dream.src.config import MCTSConfig
from dream.src.rewards import SyntaxReward
from dream.src.trainer import EntropyMCTSTrainer
from dream.src.utils import build_lr_scheduler, load_model_and_tokenizer


DEFAULT_PROMPTS = [
    "Write a Python function to check if a number is prime.",
    "Write a Python function that returns the factorial of n.",
    "Write a Python function to merge two sorted lists.",
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python function to compute the greatest common divisor of two integers.",
    "Write a Python function to reverse a linked list (describe the Node class briefly).",
    "Write a Python function to find the longest common prefix among a list of strings.",
    "Write a Python function to validate balanced parentheses in a string.",
    "Write a Python function to binary search a sorted list for a target value.",
    "Write a Python function to count inversions in a list using merge sort.",
]


def load_prompts(path: str | None) -> List[str]:
    if path and Path(path).exists():
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return DEFAULT_PROMPTS


PHASE_ORDER = (
    "initial_eval",
    "baseline_train",
    "adaptive_default",
    "adaptive_alt_hp",
)


def _cuda_relax_after_train_step() -> None:
    """Free fragmented GPU memory between prompts (Dream 7B + tree can hit ~80GB peak)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def config_to_jsonable(cfg: MCTSConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Dream comparison runner with WandB")
    p.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=(
            "initial_eval",
            "baseline_train",
            "adaptive_default",
            "adaptive_alt_hp",
        ),
        help="Which comparison arm to run (invoke once per arm from the shell script)",
    )
    p.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run_name", type=str, default="dream_cmp")
    p.add_argument("--wandb_project", type=str, default="entropy-tree-grpo-dream")
    p.add_argument("--wandb_group", type=str, default="", help="WandB group id (e.g. SLURM_JOB_ID)")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--prompts_file", type=str, default="")
    p.add_argument("--max_tree_nodes", type=int, default=8)
    p.add_argument("--branch_width", type=int, default=2)
    p.add_argument("--steps_per_expansion", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--min_steps_per_expansion", type=int, default=8)
    p.add_argument("--max_steps_per_expansion", type=int, default=36)
    p.add_argument("--branch_threshold", type=float, default=0.65)
    p.add_argument("--lora", action="store_true", help="PEFT LoRA (recommended on ~32GB)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every_steps", type=int, default=0, help="0 = no checkpoints")
    p.add_argument(
        "--wandb_prefixed_keys",
        action="store_true",
        help="Log metrics as phase/key (separate charts per arm). Default: flat keys like run_experiment_2.py for easy multi-run compare.",
    )
    args = p.parse_args()

    prompts = load_prompts(args.prompts_file or None)

    # Phase-specific tree / loss hyperparameters
    adaptive = False
    alpha_entropy = 0.5
    branch_threshold = args.branch_threshold
    phase_tag = args.phase
    phase_idx = float(PHASE_ORDER.index(args.phase))

    if args.phase == "initial_eval":
        adaptive = True
    elif args.phase == "baseline_train":
        adaptive = False
    elif args.phase == "adaptive_default":
        adaptive = True
    elif args.phase == "adaptive_alt_hp":
        adaptive = True
        alpha_entropy = 1.0  # arbitrary contrast vs default 0.5
        branch_threshold = 0.55  # slightly easier early-stop

    cfg = MCTSConfig(
        model_type="dream",
        model_name_or_path=args.model,
        device=args.device,
        max_tree_nodes=args.max_tree_nodes,
        branch_width=args.branch_width,
        steps_per_expansion=args.steps_per_expansion,
        max_new_tokens=args.max_new_tokens,
        total_denoising_steps=min(256, args.max_new_tokens),
        adaptive_stepping=adaptive,
        min_steps_per_expansion=args.min_steps_per_expansion,
        max_steps_per_expansion=args.max_steps_per_expansion,
        branch_threshold=branch_threshold,
        alpha_entropy=alpha_entropy,
        learning_rate=args.learning_rate,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=True,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    train = args.phase != "initial_eval"
    epochs = 0 if args.phase == "initial_eval" else args.num_epochs

    print(f"[dream_cmp] phase={args.phase} train={train} epochs={epochs} prompts={len(prompts)}")
    print(f"[dream_cmp] adaptive_stepping={adaptive} branch_threshold={branch_threshold} alpha_entropy={alpha_entropy}")

    # Init W&B before model load so a failed load still creates a run (config visible in UI).
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        group = args.wandb_group or None
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}_{args.phase}",
            group=group,
            tags=[args.phase, "dream", "lora" if args.lora else "full_ft"],
            config={
                **config_to_jsonable(cfg),
                "phase": args.phase,
                "phase_idx": phase_idx,
                "wandb_flat_keys": not args.wandb_prefixed_keys,
            },
        )

    global_step = 0

    def _log_wandb_step(metrics: Dict[str, Any], extra: Dict[str, float]) -> None:
        """Same metric names across arms (like scripts/run_experiment_2.py) unless --wandb_prefixed_keys."""
        if not use_wandb:
            return
        import wandb

        row: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[k] = float(v)
        row.update(extra)
        if args.wandb_prefixed_keys:
            row = {f"{phase_tag}/{k}": v for k, v in row.items()}
        wandb.log(row, step=global_step)

    try:
        model, tokenizer = load_model_and_tokenizer(cfg)
        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        reward_fn = SyntaxReward()
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )
        total_steps = max(1, epochs * len(prompts))
        scheduler = build_lr_scheduler(
            optimizer,
            total_steps,
            warmup_ratio=cfg.warmup_ratio,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        trainer = EntropyMCTSTrainer(model, tokenizer, cfg, reward_fn, optimizer, scheduler)

        t_run0 = time.perf_counter()

        for epoch in range(max(1, epochs) if train else 1):
            if not train and epoch > 0:
                break
            t_ep = time.perf_counter()
            epoch_means: Dict[str, float] = {}
            n_prompts = 0
            for pi, prompt in enumerate(prompts):
                t_step0 = time.perf_counter()
                if train:
                    metrics = trainer.train_step(prompt)
                else:
                    metrics = trainer.eval_step(prompt)
                metrics["epoch"] = float(epoch)
                metrics["prompt_idx"] = float(pi)
                step_wall = time.perf_counter() - t_step0
                n_prompts += 1
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k not in (
                        "epoch",
                        "prompt_idx",
                    ):
                        epoch_means[k] = epoch_means.get(k, 0.0) + float(v)
                _log_wandb_step(
                    metrics,
                    {
                        "wall_sec_step": float(step_wall),
                        "phase_idx": phase_idx,
                        "training_step": 1.0 if train else 0.0,
                    },
                )
                global_step += 1
                numeric_keys = sorted(
                    k
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                )
                detail = " ".join(f"{k}={float(metrics[k]):.4f}" for k in numeric_keys)
                print(f"[dream_cmp] epoch={epoch} step={global_step} {detail} wall_sec={step_wall:.2f}")

                if train:
                    _cuda_relax_after_train_step()

                if train and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    save_dir = Path(cfg.checkpoint_dir) / "dream_comparison" / args.run_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ckpt = save_dir / f"{args.phase}_step_{global_step}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": global_step,
                            "phase": args.phase,
                        },
                        ckpt,
                    )
                    print(f"[dream_cmp] saved {ckpt}")

            for k in list(epoch_means.keys()):
                epoch_means[k] /= max(n_prompts, 1)
            epoch_means["wall_sec_epoch"] = time.perf_counter() - t_ep
            # One summary row per epoch (flat names so all arms share charts).
            epoch_row = {f"epoch_mean_{k}": float(v) for k, v in epoch_means.items()}
            epoch_row["epoch"] = float(epoch)
            epoch_row["phase_idx"] = phase_idx
            epoch_row["training_step"] = 1.0 if train else 0.0
            if use_wandb:
                import wandb

                if args.wandb_prefixed_keys:
                    wandb.log(
                        {f"{phase_tag}/{k}": v for k, v in epoch_row.items()},
                        step=max(global_step - 1, 0),
                    )
                else:
                    wandb.log(epoch_row, step=max(global_step - 1, 0))

            print(f"[dream_cmp] epoch {epoch} mean metrics: {epoch_means}")

        if train:
            save_dir = Path(cfg.checkpoint_dir) / "dream_comparison" / args.run_name
            save_dir.mkdir(parents=True, exist_ok=True)
            final = save_dir / f"{args.phase}_final.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "phase": args.phase,
                },
                final,
            )
            with open(save_dir / f"{args.phase}_config.json", "w") as f:
                json.dump(config_to_jsonable(cfg), f, indent=2)
            print(f"[dream_cmp] wrote {final}")

        if use_wandb:
            import wandb

            final_step = global_step - 1 if global_step else 0
            summary = {
                "total_wall_sec": time.perf_counter() - t_run0,
                "total_steps": float(global_step),
                "phase_idx": phase_idx,
            }
            if args.wandb_prefixed_keys:
                wandb.log(
                    {f"{phase_tag}/{k}": v for k, v in summary.items()},
                    step=final_step,
                )
            else:
                wandb.log(summary, step=final_step)

        return 0
    finally:
        if use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
