#!/usr/bin/env python3
"""
Run a single Dream 7B training step (entropy-MCTS-GRPO) on a code prompt.

Use after validate_dream.py. Runs one tree build + advantage + loss + backward + step.
Uses SyntaxReward for quick iteration; switch to ExecutionLiteReward for real code RL.

Usage (from repo root):
  python dream/scripts/single_step_dream.py [--prompt "your prompt"]

Requires GPU with enough VRAM (~24GB+ with gradient checkpointing and small tree).
"""
import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

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
        gradient_checkpointing=True,
    )
    model, tokenizer = load_model_and_tokenizer(cfg)
    if getattr(cfg, "gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    reward_fn = SyntaxReward()
    trainer = EntropyMCTSTrainer(model, tokenizer, cfg, reward_fn, optimizer)

    print("Running one train_step...")
    metrics = trainer.train_step(args.prompt)
    print("Metrics:", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
