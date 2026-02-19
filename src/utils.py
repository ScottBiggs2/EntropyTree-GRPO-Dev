"""Shared utilities: device, model loading, masked canvas, sampling helpers (Phase 1)."""

from typing import Tuple, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.config import MCTSConfig


def get_device() -> str:
    """M1 MacBook: MPS if available, else CPU. No CUDA (D-001)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    config: MCTSConfig,
) -> Tuple[Any, Any]:
    """Load model and tokenizer: trust_remote_code=True, move to device, float32 (D-001)."""
    import os

    device = config.device or get_device()
    model_name = (
        os.environ.get("LOCAL_MODEL_PATH", "model_cache")
        if os.environ.get("USE_LOCAL_MODEL_CODE") == "1"
        else config.model_name_or_path
    )
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


def create_masked_response(
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Create canvas: prompt tokens + max_new_tokens mask tokens.
    Returns (input_ids, attention_mask, prompt_len).
    """
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_len = prompt_ids.size(1)
    pad_id = (
        tokenizer.pad_token_id
        or tokenizer.eos_token_id
        or tokenizer.mask_token_id
    )
    mask_id = tokenizer.mask_token_id
    total_len = prompt_len + max_new_tokens

    input_ids = torch.full(
        (1, total_len), pad_id, dtype=torch.long, device=device
    )
    input_ids[0, :prompt_len] = prompt_ids[0].to(device)
    input_ids[0, prompt_len:prompt_len + max_new_tokens] = mask_id

    attention_mask = torch.zeros_like(input_ids, device=device)
    attention_mask[0, :total_len] = 1

    return input_ids, attention_mask, prompt_len


# ----- Chat template -> token ids (same normalization as validate_model/chat for apply_chat_template) -----


def chat_template_to_token_ids(tokenizer: Any, messages: list) -> list:
    """
    Apply chat template and return a single list of token ids.
    Handles List[int], str, BatchEncoding, or tokenizers.Encoding (tokenize=True can return any of these).
    """
    raw = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return _normalize_to_token_ids(tokenizer, raw)


def _normalize_to_token_ids(tokenizer: Any, e: Any) -> list:
    """Normalize apply_chat_template output to list of ints."""
    if isinstance(e, list) and (not e or isinstance(e[0], int)):
        return e
    if isinstance(e, str):
        return tokenizer.encode(e, add_special_tokens=False)
    # tokenizers.Encoding (has .ids)
    if hasattr(e, "ids"):
        return list(e.ids)
    # BatchEncoding (input_ids)
    ids = getattr(e, "input_ids", None) or (e.get("input_ids") if isinstance(e, dict) else None)
    if ids is not None:
        if hasattr(ids, "tolist"):
            ids = ids.squeeze().tolist()
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            return list(ids[0])
        return list(ids) if not isinstance(ids, list) else ids
    if isinstance(e, (list, tuple)) and e:
        return list(e)
    raise TypeError(f"apply_chat_template returned unexpected type: {type(e)}")


# ----- Sampling helpers (D-003: Gumbel-Max, used by tree_builder and validate_model) -----

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-Max trick for stochastic sampling (dLLM / model card)."""
    if temperature == 0:
        return logits
    # MPS (Apple Silicon) does not support float64; use float32 there
    dtype = torch.float32 if logits.device.type == "mps" else torch.float64
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype, device=logits.device)
    gumbel_noise = (-torch.log(noise + 1e-10)) ** temperature
    return (logits.exp() + 1e-10) / (gumbel_noise + 1e-10)


def get_num_transfer_tokens(
    mask_index: torch.Tensor, steps: int, device: torch.device
) -> torch.Tensor:
    """Number of masked tokens to unmask per step. Returns [batch, steps]."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = (mask_num // steps).squeeze(1)
    remainder = (mask_num % steps).squeeze(1)
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        num_transfer[i, :] = base[i]
        r = int(remainder[i].item())
        if r > 0:
            num_transfer[i, :r] += 1
    return num_transfer


