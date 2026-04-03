"""Optional CUDA / OOM diagnostics for Dream training (opt-in via env)."""

from __future__ import annotations

import os
from typing import Optional

import torch

_loss_diag_once = False


def dream_oom_diag_enabled() -> bool:
    return os.environ.get("DREAM_OOM_DIAG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )


def log_cuda_mem(tag: str, *, device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        return
    dev = device if device is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / 2**30
    reserved = torch.cuda.memory_reserved(dev) / 2**30
    print(f"[dream-cuda] {tag} alloc_GiB={alloc:.3f} reserved_GiB={reserved:.3f}")


def log_cuda_mem_if_diag(tag: str, *, device: Optional[torch.device] = None) -> None:
    if dream_oom_diag_enabled():
        log_cuda_mem(tag, device=device)


def log_loss_diag_once(
    *,
    n_transitions: int,
    n_parent_groups: int,
    first_parent_seq_len: int,
    group_by_parent: bool,
    backward_per_transition: bool,
) -> None:
    """Print once per process: loss graph shape (helps locate OOM in training vs tree)."""
    global _loss_diag_once
    if _loss_diag_once:
        return
    _loss_diag_once = True
    print(
        "[dream-loss] first compute_loss: "
        f"n_transitions={n_transitions} "
        f"n_loss_forwards≈{n_parent_groups if group_by_parent else n_transitions} "
        f"group_by_parent={group_by_parent} "
        f"backward_per_transition={backward_per_transition} "
        f"first_parent_seq_len={first_parent_seq_len}"
    )


def format_cuda_oom_context(
    *,
    device: Optional[torch.device] = None,
    extra_lines: Optional[list[str]] = None,
) -> str:
    lines: list[str] = []
    if extra_lines:
        lines.extend(extra_lines)
    if torch.cuda.is_available():
        dev = device if device is not None else torch.cuda.current_device()
        lines.append(torch.cuda.memory_summary(device=dev, abbreviated=True))
    return "\n".join(lines)


def print_cuda_oom_context(
    *,
    device: Optional[torch.device] = None,
    extra_lines: Optional[list[str]] = None,
) -> None:
    print(format_cuda_oom_context(device=device, extra_lines=extra_lines))
