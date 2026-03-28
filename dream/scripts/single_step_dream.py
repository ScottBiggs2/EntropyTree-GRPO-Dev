#!/usr/bin/env python3
"""
Run a single Dream 7B training step (entropy-MCTS-GRPO) on a code prompt or task dataset.

Use after validate_dream.py. Runs one tree build + advantage + loss + backward + step.
Supports syntax-only smoke tests or dataset-backed execution rewards for real code RL.

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
from dream.src.execution_backends import make_backend
from dream.src.rewards import build_reward_function
from dream.src.task_registry import filter_code_tasks, infer_default_split, load_code_tasks
from dream.src.utils import load_model_and_tokenizer
from dream.src.trainer import EntropyMCTSTrainer


def main():
    p = argparse.ArgumentParser(description="Single Dream 7B training step (entropy-MCTS-GRPO)")
    p.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to check if a number is prime.",
        help="Code generation prompt (ignored if --dataset is provided unless --task-index is invalid)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Task dataset path (.jsonl preferred; .json legacy execution-lite supported).",
    )
    p.add_argument(
        "--dataset-split",
        type=str,
        default="",
        help="Task split to use (train/dev/all). Default: inferred from phase/context.",
    )
    p.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index within the selected dataset split (default 0).",
    )
    p.add_argument(
        "--reward",
        type=str,
        default="syntax",
        choices=("syntax", "code_format", "execution", "execution_shaped", "execution_lite"),
        help="Reward function to use.",
    )
    p.add_argument(
        "--reward-timeout",
        type=float,
        default=2.0,
        help="Timeout in seconds for execution-backed rewards.",
    )
    p.add_argument(
        "--execution-backend",
        type=str,
        default="subprocess",
        choices=("subprocess", "docker", "apptainer"),
        help="Execution backend for code rewards (default: subprocess).",
    )
    p.add_argument(
        "--sandbox-image",
        type=str,
        default="dream-sandbox:latest",
        help="Docker image or Apptainer .sif path for container backend.",
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
        help="LoRA alpha (scaling; effective scale ≈ alpha/r). Default 16 with r=8.",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default 0)",
    )
    p.add_argument(
        "--adaptive-stepping",
        action="store_true",
        help="Entropy-threshold adaptive steps per expansion (see MCTSConfig.branch_threshold)",
    )
    p.add_argument(
        "--branch-threshold",
        type=float,
        default=0.65,
        help="Early-stop micro-steps when H_masked_mean/log(V) exceeds this (adaptive; typical 0.5–0.8)",
    )
    p.add_argument(
        "--min-steps-per-expansion",
        type=int,
        default=8,
        help="Min denoise micro-steps before adaptive can stop (adaptive mode)",
    )
    p.add_argument(
        "--max-steps-per-expansion",
        type=int,
        default=48,
        help="Cap on denoise micro-steps per expansion (adaptive mode)",
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
        lora_dropout=args.lora_dropout,
        adaptive_stepping=args.adaptive_stepping,
        branch_threshold=args.branch_threshold,
        min_steps_per_expansion=args.min_steps_per_expansion,
        max_steps_per_expansion=args.max_steps_per_expansion,
    )

    selected_prompt = args.prompt
    dataset_meta = None
    if args.dataset:
        tasks = load_code_tasks(args.dataset)
        chosen_split = args.dataset_split or infer_default_split(tasks, "baseline_train")
        tasks = filter_code_tasks(tasks, chosen_split)
        if not tasks:
            raise ValueError(f"No tasks found for split={chosen_split!r} in dataset {args.dataset}")
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise IndexError(
                f"task-index {args.task_index} out of range for split={chosen_split!r} "
                f"(n_tasks={len(tasks)})"
            )
        task = tasks[args.task_index]
        selected_prompt = task.canonical_prompt
        dataset_meta = {
            "dataset": args.dataset,
            "split": chosen_split,
            "task_id": task.task_id,
            "entry_point": task.entry_point,
        }

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
    backend = None
    if args.execution_backend != "subprocess":
        backend = make_backend(
            args.execution_backend,
            image=args.sandbox_image,
            project_root=_repo_root,
        )
    reward_fn = build_reward_function(
        args.reward,
        registry_path=args.dataset or None,
        timeout=args.reward_timeout,
        project_root=_repo_root,
        backend=backend,
    )
    trainer = EntropyMCTSTrainer(model, tokenizer, cfg, reward_fn, optimizer)

    if dataset_meta is not None:
        print(f"[single_step] dataset task: {dataset_meta}")
    print(f"[single_step] reward={args.reward} device={device} lora={args.lora}")
    print("Running one train_step...")
    if args.profile_memory and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = trainer.train_step(selected_prompt)
    print("Metrics:", metrics)
    if args.profile_memory and device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[memory] peak CUDA allocated (GB, approximate): {peak:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
