#!/usr/bin/env python3
"""
Light Dream 7B comparison runs with WandB.

Phases (run separately or via run_dream_comparison.sh):
  initial_eval          — eval only, no optimizer
  baseline_train        — **MCTS / tree** GRPO, fixed steps (NOT dense; use grpo_* for flat GRPO)
  grpo_lora_baseline    — flat trajectory GRPO (no tree), LoRA — same r/α as tree when --lora
  grpo_dense_baseline   — flat trajectory GRPO, **full fine-tune** (ignores --lora); needs big GPU
  adaptive_default      — adaptive stepping + default branch_threshold / alpha_entropy
  adaptive_alt_hp       — adaptive + alternate HP (e.g. higher alpha_entropy) to sanity-check logging

WandB: by default logs flat keys (loss, avg_reward, wall_sec_step, …) like scripts/run_experiment_2.py
so runs in the same --wandb_group overlay on shared charts. Use --wandb_prefixed_keys for phase/* charts.

Usage (repo root):
  python dream/scripts/run_dream_comparison.py --phase initial_eval --wandb_project entropy-tree-grpo-dream
  python dream/scripts/run_dream_comparison.py --phase baseline_train --num_epochs 3 --run_name my_run
  python dream/scripts/run_dream_comparison.py --phase grpo_lora_baseline --lora --lora-r 8 --lora-alpha 16
  python dream/scripts/run_dream_comparison.py --phase grpo_dense_baseline   # full FT; omit --lora (forced off)
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
from typing import Any, Dict, List, Tuple

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

# Cluster scratch (not repo disk). Used when --save-checkpoints is set.
DEFAULT_CHECKPOINT_ROOT = "/scratch/biggs.s/entropy_tree_grpo_dream"

from dream.src.config import MCTSConfig
from dream.src.execution_backends import make_backend
from dream.src.rewards import build_reward_function
from dream.src.task_registry import filter_code_tasks, infer_default_split, load_code_tasks
from dream.src.trainer import BaselineGRPOTrainer, EntropyMCTSTrainer
from dream.src.utils import build_lr_scheduler, load_model_and_tokenizer


DEFAULT_LEGACY_DATASET = str(_repo_root / "data" / "execution_lite.json")
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


def load_workload(
    *,
    dataset_path: str | None,
    prompts_file: str | None,
    dataset_split: str,
    max_tasks: int,
    phase: str,
) -> Tuple[List[str], Dict[str, Any]]:
    if dataset_path:
        tasks = load_code_tasks(dataset_path)
        chosen_split = dataset_split or infer_default_split(tasks, phase)
        tasks = filter_code_tasks(tasks, chosen_split)
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        prompts = [task.canonical_prompt for task in tasks]
        return prompts, {
            "source": "dataset",
            "dataset_path": dataset_path,
            "dataset_split": chosen_split,
            "n_tasks": len(tasks),
        }

    prompts = load_prompts(prompts_file or None)
    if max_tasks > 0:
        prompts = prompts[:max_tasks]
    return prompts, {
        "source": "prompts_file" if prompts_file else "default_prompts",
        "dataset_path": "",
        "dataset_split": "",
        "n_tasks": len(prompts),
    }


PHASE_ORDER = (
    "initial_eval",
    "baseline_train",
    "grpo_lora_baseline",
    "grpo_dense_baseline",
    "adaptive_default",
    "adaptive_alt_hp",
)


def _dist_env() -> tuple[bool, int, int, int]:
    """Return (enabled, rank, local_rank, world_size) from torchrun-style env."""
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    except Exception:
        world_size = 1
    enabled = world_size > 1
    if not enabled:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    return True, rank, local_rank, world_size


def _maybe_init_distributed(*, device: str) -> tuple[bool, int, int, int]:
    enabled, rank, local_rank, world_size = _dist_env()
    if not enabled:
        return False, 0, 0, 1
    want_cuda = str(device).startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "DDP/torchrun with --device cuda needs visible GPUs, but torch.cuda.is_available() is False. "
            "Do not run multi-GPU training on a login node; request a GPU node first, e.g. "
            "`srun --partition=gpu --gres=gpu:2 --pty bash` then rerun, or use `sbatch` with your multigpu job."
        )
    if want_cuda and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    torch.distributed.init_process_group(backend=backend)
    torch.distributed.barrier()
    return True, rank, local_rank, world_size


def _shard_prompts_for_ddp(
    prompts: list[str], *, rank: int, world_size: int, pad_to_even: bool = True
) -> list[str]:
    """Shard prompts across ranks; optionally pad so each rank has equal steps.

    DDP requires all ranks to participate in the same number of backward calls.
    We pad by repeating within each rank's shard so every rank executes the same
    number of train steps per epoch.
    """
    shard = prompts[rank::world_size]
    if not pad_to_even:
        return shard
    max_len = (len(prompts) + world_size - 1) // world_size
    if not shard:
        return shard
    if len(shard) < max_len:
        need = max_len - len(shard)
        shard = shard + [shard[i % len(shard)] for i in range(need)]
    return shard


def _reduce_numeric_metrics(metrics: dict[str, Any], *, world_size: int) -> dict[str, Any]:
    if world_size <= 1 or not torch.distributed.is_initialized():
        return metrics
    out: dict[str, Any] = dict(metrics)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k, v in list(metrics.items()):
        if not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            continue
        t = torch.tensor(float(v), device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        out[k] = float((t / float(world_size)).item())
    return out


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
            "grpo_lora_baseline",
            "grpo_dense_baseline",
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
    p.add_argument(
        "--num_epochs",
        type=int,
        default=6,
        help="Epochs per phase (default 6; use shell NUM_EPOCHS for long runs).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Task dataset path (.jsonl preferred; .json legacy execution-lite also supported).",
    )
    p.add_argument(
        "--dataset-split",
        type=str,
        default="",
        help="Task split to use (train/dev/all). Default: inferred from phase.",
    )
    p.add_argument("--prompts_file", type=str, default="")
    p.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit dataset tasks or prompts used in this run (0 = all).",
    )
    p.add_argument(
        "--reward",
        type=str,
        default="syntax",
        choices=("syntax", "code_format", "execution", "execution_shaped", "execution_lite"),
        help="Reward function to use. Execution rewards require prompts that exist in the dataset/registry.",
    )
    p.add_argument(
        "--reward-tie-breaker",
        type=str,
        default="none",
        choices=("none", "ast_size", "code_len"),
        dest="reward_tie_breaker",
        help="Optional tiny continuous bonus to reduce reward ties (default: none).",
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
    p.add_argument("--max_tree_nodes", type=int, default=8)
    p.add_argument("--branch_width", type=int, default=2)
    p.add_argument("--steps_per_expansion", type=int, default=12)
    p.add_argument(
        "--total_denoising_steps",
        type=int,
        default=128,
        help="Global diffusion denoising schedule (MCTSConfig.total_denoising_steps). "
        "Same role as diffusion_generate(..., steps=T); validate_dream.py uses steps=128.",
    )
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        dest="max_prompt_tokens",
        help="Cap prompt length in tokens for tree/baseline rollouts (0 = no cap). "
        "Helps avoid OOM on very long AceCode stubs.",
    )
    p.add_argument("--min_steps_per_expansion", type=int, default=8)
    p.add_argument("--max_steps_per_expansion", type=int, default=36)
    p.add_argument("--branch_threshold", type=float, default=0.65)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for eval / tree completion (default 0.2).",
    )
    p.add_argument(
        "--train-sampling-temperature",
        type=float,
        default=0.8,
        dest="train_sampling_temperature",
        help="Training-time generation temperature for flat GRPO and tree expansion "
        "(default 0.8; higher ⇒ more diversity across K samples / siblings). "
        "Set to 0 to fall back to --temperature.",
    )
    p.add_argument("--lora", action="store_true", help="PEFT LoRA (recommended on ~32GB)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument(
        "--num-baseline-samples",
        type=int,
        default=4,
        help="K trajectories per prompt for grpo_lora_baseline / grpo_dense_baseline (flat GRPO; default 4).",
    )
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument(
        "--entropy-weight-min",
        type=float,
        default=0.08,
        help="Clamp floor for loss entropy weight (default 0.08; lower ⇒ fewer edges floored, stronger entropy signal).",
    )
    p.add_argument(
        "--entropy-weight-max",
        type=float,
        default=2.5,
        help="Clamp ceiling for loss entropy weight.",
    )
    p.add_argument(
        "--save-checkpoints",
        "--save_checkpoints",
        action="store_true",
        help="If set, write final (and optional periodic) checkpoints under --checkpoint-dir. "
        "Default: no checkpoint files (W&B metrics only).",
    )
    p.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"Root directory for dream_comparison/<run_name>/ (default: {DEFAULT_CHECKPOINT_ROOT}).",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Path to .pt checkpoint to resume training from.",
    )
    p.add_argument(
        "--save-every-steps",
        "--save_every_steps",
        type=int,
        default=0,
        dest="save_every_steps",
        help="With --save-checkpoints: also save every N global steps (0 = only final.pt per phase).",
    )
    p.add_argument(
        "--wandb_prefixed_keys",
        action="store_true",
        help="Log metrics as phase/key (separate charts per arm). Default: flat keys like run_experiment_2.py for easy multi-run compare.",
    )
    p.add_argument(
        "--trace-every-steps",
        type=int,
        default=0,
        dest="trace_every_steps",
        help="If >0, write a JSON tree trace every N steps for tree phases (baseline_train/initial_eval/adaptive_*).",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default="traces",
        dest="trace_dir",
        help="Directory (relative to repo root) to write trace JSON files.",
    )
    p.add_argument("--trace-max-nodes", type=int, default=0, dest="trace_max_nodes")
    p.add_argument("--trace-max-leaves", type=int, default=0, dest="trace_max_leaves")
    p.add_argument("--trace-decode-chars", type=int, default=240, dest="trace_decode_chars")
    args = p.parse_args()

    ddp_enabled, rank, local_rank, world_size = _maybe_init_distributed(
        device=str(args.device)
    )
    is_rank0 = rank == 0

    dataset_path = args.dataset or ""
    if args.reward in ("execution", "execution_shaped", "execution_lite") and not dataset_path:
        dataset_path = DEFAULT_LEGACY_DATASET

    prompts, workload = load_workload(
        dataset_path=dataset_path or None,
        prompts_file=args.prompts_file or None,
        dataset_split=args.dataset_split,
        max_tasks=args.max_tasks,
        phase=args.phase,
    )

    if ddp_enabled:
        prompts = _shard_prompts_for_ddp(
            prompts, rank=rank, world_size=world_size, pad_to_even=True
        )

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
    elif args.phase == "grpo_lora_baseline":
        adaptive = False
    elif args.phase == "grpo_dense_baseline":
        adaptive = False
    elif args.phase == "adaptive_default":
        adaptive = True
    elif args.phase == "adaptive_alt_hp":
        adaptive = True
        alpha_entropy = 1.0  # arbitrary contrast vs default 0.5
        branch_threshold = 0.55  # slightly easier early-stop

    # Dense flat GRPO: always full fine-tune (no LoRA), even if the shell passes --lora.
    use_lora = False if args.phase == "grpo_dense_baseline" else args.lora
    if args.phase == "grpo_dense_baseline" and args.lora:
        print(
            "[dream_cmp] grpo_dense_baseline: ignoring --lora (this phase is full dense fine-tune)."
        )

    cfg = MCTSConfig(
        model_type="dream",
        model_name_or_path=args.model,
        device=(
            f"cuda:{local_rank}"
            if ddp_enabled and str(args.device).startswith("cuda")
            else args.device
        ),
        max_tree_nodes=args.max_tree_nodes,
        branch_width=args.branch_width,
        steps_per_expansion=args.steps_per_expansion,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        total_denoising_steps=args.total_denoising_steps,
        adaptive_stepping=adaptive,
        min_steps_per_expansion=args.min_steps_per_expansion,
        max_steps_per_expansion=args.max_steps_per_expansion,
        branch_threshold=branch_threshold,
        alpha_entropy=alpha_entropy,
        entropy_weight_min=args.entropy_weight_min,
        entropy_weight_max=args.entropy_weight_max,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        train_sampling_temperature=args.train_sampling_temperature,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=True,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        num_baseline_samples=args.num_baseline_samples,
        trace_every_steps=args.trace_every_steps,
        trace_dir=args.trace_dir,
        trace_max_nodes=args.trace_max_nodes,
        trace_max_leaves=args.trace_max_leaves,
        trace_decode_chars=args.trace_decode_chars,
    )

    train = args.phase != "initial_eval"
    epochs = 0 if args.phase == "initial_eval" else args.num_epochs

    print(
        f"[dream_cmp] phase={args.phase} train={train} epochs={epochs} prompts={len(prompts)}"
        + (
            f" num_baseline_samples={args.num_baseline_samples}"
            if args.phase in ("grpo_lora_baseline", "grpo_dense_baseline")
            else ""
        )
    )
    if ddp_enabled:
        print(
            f"[dream_cmp] ddp=1 rank={rank} local_rank={local_rank} world_size={world_size} device={cfg.device}"
        )
    print(
        f"[dream_cmp] adaptive_stepping={adaptive} branch_threshold={branch_threshold} "
        f"alpha_entropy={alpha_entropy} entropy_weight_min={args.entropy_weight_min}"
    )
    print(
        f"[dream_cmp] total_denoising_steps={args.total_denoising_steps} "
        f"max_new_tokens={args.max_new_tokens} "
        f"max_prompt_tokens={args.max_prompt_tokens} (0=no cap)"
    )
    print(
        f"[dream_cmp] reward={args.reward} workload_source={workload['source']} "
        f"dataset_split={workload['dataset_split'] or 'n/a'}"
    )
    if args.phase == "baseline_train":
        print(
            "[dream_cmp] NOTE: baseline_train is **MCTS (tree)** GRPO, not flat/dense GRPO."
        )
    if args.save_checkpoints:
        periodic = (
            f", also every {args.save_every_steps} global steps"
            if args.save_every_steps > 0
            else " (final .pt per phase only)"
        )
        print(
            f"[dream_cmp] checkpoints: saving under "
            f"{args.checkpoint_dir}/dream_comparison/{args.run_name}{periodic}"
        )
    else:
        print("[dream_cmp] checkpoints: disabled (default; no .pt files)")

    # Init W&B before model load so a failed load still creates a run (config visible in UI).
    # In DDP, only rank 0 logs (avoid duplicated runs + rate-limit issues).
    use_wandb = (not args.no_wandb) and (not ddp_enabled or is_rank0)
    if use_wandb:
        import wandb

        group = args.wandb_group or None
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}_{args.phase}",
            group=group,
            tags=[
                args.phase,
                "dream",
                "lora" if use_lora else "full_ft",
                *(
                    ["grpo_flat", f"lora_r{args.lora_r}"]
                    if args.phase == "grpo_lora_baseline"
                    else []
                ),
                *(
                    ["grpo_flat", "dense_grpo", "full_ft"]
                    if args.phase == "grpo_dense_baseline"
                    else []
                ),
            ],
            config={
                **config_to_jsonable(cfg),
                "phase": args.phase,
                "phase_idx": phase_idx,
                "wandb_flat_keys": not args.wandb_prefixed_keys,
                "reward_name": args.reward,
                "reward_timeout": args.reward_timeout,
                "dataset_path": workload["dataset_path"],
                "dataset_split": workload["dataset_split"],
                "workload_source": workload["source"],
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

        ddp_model = None
        if ddp_enabled and train and str(cfg.device).startswith("cuda"):
            # Tree-GRPO graphs differ by prompt; some LoRA params may be unused in a given
            # backward. find_unused_parameters=False often raises DDP reduction errors on
            # one rank (looks like random rank in torchrun logs).
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
            model_for_train = ddp_model
        else:
            model_for_train = model

        backend = None
        if args.execution_backend != "subprocess":
            backend = make_backend(
                args.execution_backend,
                image=args.sandbox_image,
                project_root=_repo_root,
            )
        reward_fn = build_reward_function(
            args.reward,
            registry_path=dataset_path or None,
            timeout=args.reward_timeout,
            project_root=_repo_root,
            backend=backend,
            tie_breaker=args.reward_tie_breaker,
        )
        optimizer = torch.optim.AdamW(
            (p for p in model_for_train.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )
        total_steps = max(1, epochs * len(prompts))
        scheduler = build_lr_scheduler(
            optimizer,
            total_steps,
            warmup_ratio=cfg.warmup_ratio,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        if args.phase in ("grpo_lora_baseline", "grpo_dense_baseline"):
            trainer = BaselineGRPOTrainer(
                model_for_train, tokenizer, cfg, reward_fn, optimizer, scheduler
            )
        else:
            trainer = EntropyMCTSTrainer(
                model_for_train, tokenizer, cfg, reward_fn, optimizer, scheduler
            )

        # Log these every step so W&B charts show which loss knobs apply (see dream/docs/WANDB_METRICS.md).
        wb_cfg_diag = {
            "cfg_alpha_time": float(cfg.alpha_time),
            "cfg_alpha_entropy": float(cfg.alpha_entropy),
            "cfg_entropy_weight_min": float(cfg.entropy_weight_min),
            "cfg_entropy_weight_max": float(cfg.entropy_weight_max),
            "cfg_adaptive_stepping": 1.0 if cfg.adaptive_stepping else 0.0,
            "cfg_branch_threshold": float(cfg.branch_threshold),
            "cfg_min_steps_per_expansion": float(cfg.min_steps_per_expansion),
            "cfg_max_steps_per_expansion": float(cfg.max_steps_per_expansion),
            "cfg_temperature": float(cfg.temperature),
            "cfg_train_sampling_temperature": float(cfg.train_sampling_temperature),
            "cfg_num_baseline_samples": float(cfg.num_baseline_samples),
            "cfg_use_lora": 1.0 if cfg.use_lora else 0.0,
            "cfg_reward_timeout": float(args.reward_timeout),
            "cfg_max_tasks": float(args.max_tasks),
        }

        global_step = 0
        resume_path = getattr(args, "resume_from", None)
        if resume_path and os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            if "model" in ckpt:
                if ddp_model is not None:
                    ddp_model.module.load_state_dict(ckpt["model"], strict=False)
                else:
                    model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "step" in ckpt:
                global_step = ckpt["step"]
                if scheduler is not None and hasattr(scheduler, "step"):
                    for _ in range(global_step):
                        scheduler.step()
            if not ddp_enabled or is_rank0:
                print(f"[dream_cmp] Resumed from {resume_path} at step {global_step}")

        start_epoch = 0
        start_prompt_idx = 0
        if train and global_step > 0 and len(prompts) > 0:
            start_epoch = global_step // len(prompts)
            start_prompt_idx = global_step % len(prompts)

        t_run0 = time.perf_counter()

        for epoch in range(start_epoch if train else 0, max(1, epochs) if train else 1):
            if not train and epoch > 0:
                break
            t_ep = time.perf_counter()
            epoch_means: Dict[str, float] = {}
            n_prompts = 0
            for pi, prompt in enumerate(prompts):
                if train and epoch == start_epoch and pi < start_prompt_idx:
                    continue
                t_step0 = time.perf_counter()
                if train:
                    metrics = trainer.train_step(
                        prompt,
                        step_id=global_step,
                        epoch=epoch,
                        prompt_idx=pi,
                        phase=args.phase,
                    )
                else:
                    metrics = trainer.eval_step(
                        prompt,
                        step_id=global_step,
                        epoch=epoch,
                        prompt_idx=pi,
                        phase=args.phase,
                    )
                metrics = _reduce_numeric_metrics(metrics, world_size=world_size)
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
                        **wb_cfg_diag,
                    },
                )
                global_step += 1
                numeric_keys = sorted(
                    k
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                )
                detail = " ".join(f"{k}={float(metrics[k]):.4f}" for k in numeric_keys)
                if not ddp_enabled or is_rank0:
                    print(
                        f"[dream_cmp] epoch={epoch} step={global_step} {detail} wall_sec={step_wall:.2f}"
                    )

                if train:
                    _cuda_relax_after_train_step()

                if (
                    train
                    and args.save_checkpoints
                    and args.save_every_steps > 0
                    and global_step % args.save_every_steps == 0
                ):
                    if not ddp_enabled or is_rank0:
                        save_dir = (
                            Path(args.checkpoint_dir)
                            / "dream_comparison"
                            / args.run_name
                        )
                        save_dir.mkdir(parents=True, exist_ok=True)
                        ckpt = save_dir / f"{args.phase}_step_{global_step}.pt"
                        model_state = (
                            ddp_model.module.state_dict()
                            if ddp_model is not None
                            else model.state_dict()
                        )
                        torch.save(
                            {
                                "model": model_state,
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
            epoch_row.update(wb_cfg_diag)
            if use_wandb:
                import wandb

                if args.wandb_prefixed_keys:
                    wandb.log(
                        {f"{phase_tag}/{k}": v for k, v in epoch_row.items()},
                        step=max(global_step - 1, 0),
                    )
                else:
                    wandb.log(epoch_row, step=max(global_step - 1, 0))

            if not ddp_enabled or is_rank0:
                print(f"[dream_cmp] epoch {epoch} mean metrics: {epoch_means}")

        if train and args.save_checkpoints and (not ddp_enabled or is_rank0):
            save_dir = Path(args.checkpoint_dir) / "dream_comparison" / args.run_name
            save_dir.mkdir(parents=True, exist_ok=True)
            final = save_dir / f"{args.phase}_final.pt"
            model_state = (
                ddp_model.module.state_dict() if ddp_model is not None else model.state_dict()
            )
            torch.save(
                {
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "phase": args.phase,
                },
                final,
            )
            with open(save_dir / f"{args.phase}_config.json", "w") as f:
                json.dump(config_to_jsonable(cfg), f, indent=2)
            print(f"[dream_cmp] wrote {final}")
            # If LoRA/PEFT is enabled, also save a PEFT adapter directory for eval scripts
            # (they accept --adapter and load via PeftModel.from_pretrained).
            save_model = ddp_model.module if ddp_model is not None else model
            if use_lora and hasattr(save_model, "save_pretrained"):
                adapter_dir = save_dir / "adapter"
                try:
                    save_model.save_pretrained(str(adapter_dir))
                    print(f"[dream_cmp] wrote adapter dir {adapter_dir}")
                except Exception as e:
                    print(f"[dream_cmp WARN] failed to save adapter dir: {e}")

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

        if ddp_enabled and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return 0
    finally:
        if use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        if ddp_enabled and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
