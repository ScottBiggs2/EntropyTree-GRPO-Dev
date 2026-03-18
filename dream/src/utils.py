"""Shared utilities for Dream substack: device, model loading, sampling."""

from typing import Any, Tuple, Optional
import math
import os

import torch

from dream.src.config import MCTSConfig


def get_device() -> str:
    """CUDA if available (cloud GPU), else MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(config: MCTSConfig) -> Tuple[Any, Any]:
    """Load model and tokenizer for MDLM or Dream.

    On the local laptop this will typically not be executed (no heavy
    model loads); the function exists so the same code can run on a
    cloud GPU environment later.
    """
    device = config.device or get_device()

    # Optional override for local, code-modded models (mirrors parent project).
    model_name = (
        os.environ.get("LOCAL_MODEL_PATH", "model_cache")
        if os.environ.get("USE_LOCAL_MODEL_CODE") == "1"
        else config.model_name_or_path
    )

    if config.model_type == "dream":
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    else:
        # MDLM-style masked LM.
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore

        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    return model, tokenizer


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-Max trick for stochastic sampling (MDLM style).

    For Dream we use categorical sampling; this remains for MDLM mode.
    """
    if temperature == 0:
        return logits
    # MPS (Apple Silicon) does not support float64; use float32 there.
    dtype = torch.float32 if logits.device.type == "mps" else torch.float64
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype, device=logits.device)
    # Classic Gumbel(0,1)
    gumbel = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
    return (logits + gumbel) / max(temperature, 1e-6)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup + cosine decay (mirrors parent project)."""

    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(min_lr_ratio, current_step / warmup_steps)
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

