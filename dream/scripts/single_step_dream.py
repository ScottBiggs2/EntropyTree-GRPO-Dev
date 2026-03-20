#!/usr/bin/env python3
"""
Run a single Dream 7B training step (entropy-MCTS-GRPO) on a code prompt.

Use after validate_dream.py. Runs one tree build + advantage + loss + backward + step.
Uses SyntaxReward for quick iteration; switch to ExecutionLiteReward for real code RL.

Usage (from repo root):
  python dream/scripts/single_step_dream.py [--prompt "your prompt"]

Default full fine-tune needs ~40GB+ VRAM (weights+grads ~28GB bf16 + activations).
On ~32GB use --lora (PEFT). Sibling-grouped loss: see n_loss_forwards in metrics.
"""
import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os


def _configure_hf_cache() -> None:
    """Harden HuggingFace cache locations to avoid home-dir quotas."""
    if os.environ.get("HF_HOME"):
        return

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    scratch_root = os.environ.get("SCRATCH") or "/scratch"
    hf_home = os.path.join(scratch_root, user, "hf_home")
    try:
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    except Exception:
        pass


_configure_hf_cache()

import torch

from dream.src.config import MCTSConfig
from dream.src.utils import load_model_and_tokenizer
from dream.src.rewards import SyntaxReward
from dream.src.trainer import EntropyMCTSTrainer


def main():
    p = argparse.ArgumentParser(description="Single Dream 7B training step (entropy-MCTS-GRPO)")
    p.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to check if a number is prime.",
        help="Code generation prompt",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Dream-org/Dream-v0-Instruct-7B",
        help="HuggingFace model path",
    )
    p.add_argument(
        "--max-tree-nodes",
        type=int,
        default=5,
        help="Max nodes in the tree (smaller = less VRAM)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens per completion",
    )
    p.add_argument(
        "--steps-per-expansion",
        type=int,
        default=16,
        help="Denoising steps per tree expansion",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        choices=("adamw", "sgd"),
        default="adamw",
        help="adamw (default) or sgd — SGD drops Adam state (~2× optimizer VRAM) for smoke tests",
    )
    p.add_argument(
        "--profile-memory",
        action="store_true",
        help="Print torch.cuda.max_memory_allocated() after the step (GB)",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="PEFT LoRA on Dream layers (recommended on ~32GB — full FT needs ~28GB+ weights+grads alone)",
    )
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default 8)")
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default 16)",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (sometimes trades VRAM vs peak during backward)",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA. Dream 7B training step will be very slow or OOM on CPU.")

    print("Loading Dream model and tokenizer...")
    cfg = MCTSConfig(
        model_type="dream",
        model_name_or_path=args.model,
        device=device,
        max_tree_nodes=args.max_tree_nodes,
        max_new_tokens=args.max_new_tokens,
        steps_per_expansion=args.steps_per_expansion,
        total_denoising_steps=min(256, args.max_new_tokens),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model, tokenizer = load_model_and_tokenizer(cfg)
    if getattr(cfg, "gradient_checkpointing", False) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    model.train()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    reward_fn = SyntaxReward()
    trainer = EntropyMCTSTrainer(model, tokenizer, cfg, reward_fn, optimizer)

    print("Running one train_step...")
    if args.profile_memory and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = trainer.train_step(args.prompt)
    print("Metrics:", metrics)
    if args.profile_memory and device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[memory] peak CUDA allocated (GB, approximate): {peak:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
