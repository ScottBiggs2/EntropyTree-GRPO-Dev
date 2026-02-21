"""
Phase 8.2: Experiment runner — train baseline or entropy-MCTS GRPO, log to WandB + local, checkpoint clearly.
Usage:
  python scripts/run_experiment.py --method baseline --num_epochs 2 --prompts_file prompts.txt
  python scripts/run_experiment.py --method entropy_mcts --num_epochs 2
Checkpoints: <checkpoint_dir>/baseline_grpo/<run_name>/ and <checkpoint_dir>/entropy_mcts_grpo/<run_name>/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import MCTSConfig
from src.utils import load_model_and_tokenizer
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter
from src.rewards import SyntaxReward
from src.advantages import AdvantageComputer
from src.loss import WeightedGRPOLoss
from src.trainer import BaselineGRPOTrainer, EntropyMCTSTrainer


# Subdir names so checkpoints are clearly with/without entropy-tree GRPO
BASELINE_CHECKPOINT_SUBDIR = "baseline_grpo"
ENTROPY_MCTS_CHECKPOINT_SUBDIR = "entropy_mcts_grpo"


def load_prompts(prompts_file: Optional[str]) -> List[str]:
    """Load prompts from file (one per line) or return default dev list."""
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file) as f:
            return [line.strip() for line in f if line.strip()]
    return [
        "def fibonacci(n):",
        "def factorial(n):",
        "def is_prime(n):",
    ]


def run_baseline(
    config: MCTSConfig,
    prompts: list[str],
    run_name: str,
    checkpoint_dir: str,
    save_every_steps: Optional[int],
    use_wandb: bool,
) -> None:
    """Train baseline GRPO (no tree), checkpoint to checkpoint_dir/baseline_grpo/."""
    save_dir = Path(checkpoint_dir) / BASELINE_CHECKPOINT_SUBDIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[baseline] Checkpoints: {save_dir}")

    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ec = EntropyComputer()
    tw = TimeWeighter(config.total_denoising_steps)
    loss_fn = WeightedGRPOLoss(config, ec, tw, tokenizer.mask_token_id)
    trainer = BaselineGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=SyntaxReward(),
        loss_computer=loss_fn,
        optimizer=optimizer,
    )

    if use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=f"baseline_{run_name}", config=vars(config))

    global_step = 0
    for epoch in range(config.num_epochs):
        t0 = time.perf_counter()
        epoch_metrics = trainer.train_epoch(prompts)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["method"] = "baseline"
        dt = time.perf_counter() - t0
        epoch_metrics["wall_sec"] = dt
        print(f"[baseline] epoch {epoch} " + " ".join(f"{k}={v}" for k, v in epoch_metrics.items()))
        if use_wandb:
            wandb.log(epoch_metrics, step=global_step)
        global_step += len(prompts)
        if save_every_steps and global_step % save_every_steps == 0:
            ckpt_path = save_dir / f"step_{global_step}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step}, ckpt_path)
            print(f"[baseline] Saved {ckpt_path}")

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step},
        save_dir / "final.pt",
    )
    with open(save_dir / "config.json", "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in vars(config).items()}, f, indent=2)
    print(f"[baseline] Saved final.pt and config.json to {save_dir}")
    if use_wandb:
        wandb.finish()


def run_entropy_mcts(
    config: MCTSConfig,
    prompts: list[str],
    run_name: str,
    checkpoint_dir: str,
    save_every_steps: Optional[int],
    use_wandb: bool,
) -> None:
    """Train entropy-MCTS GRPO, checkpoint to checkpoint_dir/entropy_mcts_grpo/."""
    save_dir = Path(checkpoint_dir) / ENTROPY_MCTS_CHECKPOINT_SUBDIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[entropy_mcts] Checkpoints: {save_dir}")

    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ec = EntropyComputer()
    tw = TimeWeighter(config.total_denoising_steps)
    loss_fn = WeightedGRPOLoss(config, ec, tw, tokenizer.mask_token_id)
    trainer = EntropyMCTSTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=SyntaxReward(),
        advantage_computer=AdvantageComputer(),
        loss_computer=loss_fn,
        optimizer=optimizer,
    )

    if use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=f"entropy_mcts_{run_name}", config=vars(config))

    global_step = 0
    for epoch in range(config.num_epochs):
        agg = {"loss": [], "avg_reward": [], "max_reward": [], "tree_nodes": [], "tree_leaves": [], "avg_entropy": []}
        for prompt in prompts:
            m = trainer.train_step(prompt)
            for k in agg:
                if k in m:
                    agg[k].append(m[k])
            global_step += 1
            if use_wandb:
                wandb.log(m, step=global_step)
            if save_every_steps and global_step % save_every_steps == 0:
                ckpt_path = save_dir / f"step_{global_step}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step}, ckpt_path)
                print(f"[entropy_mcts] Saved {ckpt_path}")
        epoch_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}
        epoch_metrics["epoch"] = epoch
        epoch_metrics["method"] = "entropy_mcts"
        print(f"[entropy_mcts] epoch {epoch} " + " ".join(f"{k}={v}" for k, v in epoch_metrics.items()))

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step},
        save_dir / "final.pt",
    )
    with open(save_dir / "config.json", "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in vars(config).items()}, f, indent=2)
    print(f"[entropy_mcts] Saved final.pt and config.json to {save_dir}")
    if use_wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Phase 8: run baseline or entropy-MCTS GRPO experiment")
    p.add_argument("--method", choices=["baseline", "entropy_mcts"], required=True)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="entropy-tree-grpo")
    p.add_argument("--save_every_steps", type=int, default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable WandB even if available")
    p.add_argument("--max_tree_nodes", type=int, default=10)
    p.add_argument("--branch_width", type=int, default=2)
    p.add_argument("--steps_per_expansion", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_baseline_samples", type=int, default=4)
    args = p.parse_args()

    run_name = args.run_name or f"run_{int(time.time())}"
    use_wandb = not args.no_wandb and os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        print("WandB: enabled")
    else:
        print("WandB: disabled (set WANDB_API_KEY or use --no_wandb)")

    config = MCTSConfig(
        num_epochs=args.num_epochs,
        run_name=run_name,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        save_every_steps=args.save_every_steps,
        max_tree_nodes=args.max_tree_nodes,
        branch_width=args.branch_width,
        steps_per_expansion=args.steps_per_expansion,
        max_new_tokens=args.max_new_tokens,
        num_baseline_samples=args.num_baseline_samples,
    )
    prompts = load_prompts(args.prompts_file)
    print(f"Prompts: {len(prompts)}")

    if args.method == "baseline":
        run_baseline(config, prompts, run_name, args.checkpoint_dir, args.save_every_steps, use_wandb)
    else:
        run_entropy_mcts(config, prompts, run_name, args.checkpoint_dir, args.save_every_steps, use_wandb)


if __name__ == "__main__":
    main()
