"""
Phase 8.5: Experiment runner with execution-lite reward (sandbox + prompt→test registry).
Train baseline or entropy-MCTS GRPO using ExecutionLiteReward; prompts default to registry.
Usage:
  python scripts/run_experiment_2.py --method baseline --num_epochs 2
  python scripts/run_experiment_2.py --method entropy_mcts --num_epochs 2 --prompts_file my_prompts.txt
  python scripts/run_experiment_2.py --method baseline --no_wandb --max_new_tokens 128
Checkpoints: same as run_experiment (baseline_grpo/ and entropy_mcts_grpo/ under checkpoint_dir).
"""
import argparse
import json
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
from src.rewards import ExecutionLiteReward
from src.advantages import AdvantageComputer
from src.loss import WeightedGRPOLoss
from src.trainer import BaselineGRPOTrainer, EntropyMCTSTrainer
from src.execution import get_prompts_from_registry

BASELINE_CHECKPOINT_SUBDIR = "baseline_grpo"
ENTROPY_MCTS_CHECKPOINT_SUBDIR = "entropy_mcts_grpo"


def load_prompts(prompts_file: Optional[str], registry_path: Optional[str] = None) -> List[str]:
    """Load prompts: from file if given, else from execution_lite registry. Fallback to small default list."""
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file) as f:
            return [line.strip() for line in f if line.strip()]
    prompts = get_prompts_from_registry(registry_path)
    if prompts:
        return prompts
    return [
        "def fibonacci(n):",
        "def factorial(n):",
        "def is_prime(n):",
        "def sum_list(lst):",
        "def max_list(lst):",
    ]


def run_baseline(
    config: MCTSConfig,
    prompts: List[str],
    run_name: str,
    checkpoint_dir: str,
    save_every_steps: Optional[int],
    use_wandb: bool,
    reward_fn: ExecutionLiteReward,
) -> None:
    save_dir = Path(checkpoint_dir) / BASELINE_CHECKPOINT_SUBDIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[baseline] Checkpoints: {save_dir}")
    print(f"[baseline] Reward: ExecutionLiteReward (registry, timeout={reward_fn.timeout}s)")

    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ec = EntropyComputer()
    tw = TimeWeighter(config.total_denoising_steps)
    loss_fn = WeightedGRPOLoss(config, ec, tw, tokenizer.mask_token_id)
    trainer = BaselineGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=reward_fn,
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
        epoch_metrics["reward"] = "execution_lite"
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
    prompts: List[str],
    run_name: str,
    checkpoint_dir: str,
    save_every_steps: Optional[int],
    use_wandb: bool,
    reward_fn: ExecutionLiteReward,
) -> None:
    save_dir = Path(checkpoint_dir) / ENTROPY_MCTS_CHECKPOINT_SUBDIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[entropy_mcts] Checkpoints: {save_dir}")
    print(f"[entropy_mcts] Reward: ExecutionLiteReward (registry, timeout={reward_fn.timeout}s)")

    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ec = EntropyComputer()
    tw = TimeWeighter(config.total_denoising_steps)
    loss_fn = WeightedGRPOLoss(config, ec, tw, tokenizer.mask_token_id)
    trainer = EntropyMCTSTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=reward_fn,
        advantage_computer=AdvantageComputer(),
        loss_computer=loss_fn,
        optimizer=optimizer,
    )

    if use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, name=f"entropy_mcts_{run_name}", config=vars(config))

    global_step = 0
    for epoch in range(config.num_epochs):
        agg = {
            "loss": [],
            "avg_reward": [],
            "max_reward": [],
            "tree_nodes": [],
            "tree_leaves": [],
            "avg_entropy": [],
            "n_transitions": [],
            "mean_abs_adv": [],
            "mean_w_time": [],
            "mean_w_ent": [],
            "mean_weight": [],
            "mean_weighted_adv_logp": [],
        }
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
        epoch_metrics["reward"] = "execution_lite"
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
    p = argparse.ArgumentParser(description="Phase 8.5: baseline or entropy-MCTS GRPO with execution-lite reward")
    p.add_argument("--method", choices=["baseline", "entropy_mcts"], required=True)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--prompts_file", type=str, default=None, help="Override prompts; default = from data/execution_lite.json")
    p.add_argument("--registry", type=str, default=None, help="Path to execution_lite.json; default = data/execution_lite.json")
    p.add_argument("--exec_timeout", type=float, default=2.0, help="Sandbox run timeout (seconds)")
    p.add_argument("--syntax_bonus", type=float, default=0.05, help="Extra reward for AST-parseable when not all tests pass")
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
    p.add_argument("--device", type=str, default=None, help="Override device (cuda, mps, cpu)")
    args = p.parse_args()

    run_name = args.run_name or f"exec_lite_{int(time.time())}"
    use_wandb = not args.no_wandb
    if use_wandb:
        print("WandB: enabled")
    else:
        print("WandB: disabled")

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
        device=args.device,
    )
    prompts = load_prompts(args.prompts_file, args.registry)
    print(f"Prompts: {len(prompts)} (Phase 8.5 execution-lite)")

    reward_fn = ExecutionLiteReward(
        registry_path=args.registry,
        syntax_bonus=args.syntax_bonus,
        timeout=args.exec_timeout,
    )

    if args.method == "baseline":
        run_baseline(config, prompts, run_name, args.checkpoint_dir, args.save_every_steps, use_wandb, reward_fn)
    else:
        run_entropy_mcts(config, prompts, run_name, args.checkpoint_dir, args.save_every_steps, use_wandb, reward_fn)
 

if __name__ == "__main__":
    main()
