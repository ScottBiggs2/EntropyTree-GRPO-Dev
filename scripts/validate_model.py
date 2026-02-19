"""
Phase 0: Validate model loads and generates on M1.
Uses model card reference logic: block-based denoising with Gumbel noise.
"""
import sys
import warnings
from pathlib import Path

# Avoid "Logging error" from transformers' deprecated attention-mask API
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")

# Add project root for imports when running as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM


def get_device() -> str:
    """M1 MacBook: MPS if available, else CPU. No CUDA."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    """How many masked tokens to unmask per step. [batch, steps]."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = (mask_num // steps).squeeze(1)
    remainder = (mask_num % steps).squeeze(1)
    num_transfer = torch.zeros(mask_num.size(0), steps, device=device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        num_transfer[i, :] = base[i]
        r = int(remainder[i].item())
        if r > 0:
            num_transfer[i, :r] += 1
    return num_transfer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_tensor: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
    mask_id: int,
    device: str,
    steps: int = 128,
    max_new_tokens: int = 128,
    block_size: int = 64,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """Block-based denoising generation (model card style). Returns final token tensor [B, T]."""
    batch_size = prompt_tensor.size(0)
    total_length = int(prompt_lens.max().item() + max_new_tokens)
    positions = torch.arange(total_length, device=device)
    prompt_index = (
        torch.arange(total_length, device=device).unsqueeze(0)
        < prompt_lens.unsqueeze(1)
    )

    x = torch.full(
        (batch_size, total_length), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt_tensor[i, :length]
        x[i, length : length + max_new_tokens] = mask_id
        valid_end = min(length + max_new_tokens, total_length)
        attention_mask[i, :valid_end] = 1

    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size
        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer_tokens = get_num_transfer_tokens(
            init_block_mask, steps_per_block, device
        )

        for i in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float32, device=device)
            else:
                raise ValueError(remasking)

            confidence = torch.full_like(x0_p, -np.inf)
            confidence = torch.where(block_mask, x0_p, confidence)
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(batch_size):
                k = int(num_transfer_tokens[j, i].item())
                if k == 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=min(k, confidence.size(1)))
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    import os

    device = get_device()
    print(f"Device: {device}")

    # Option B1: load from local dir with dllm-free modeling file to avoid circular import
    if os.environ.get("USE_LOCAL_MODEL_CODE") == "1":
        model_name = os.environ.get("LOCAL_MODEL_PATH", "model_cache")
        print(f"Loading from local path (no dllm package): {model_name}")
    else:
        model_name = "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
        print(f"Loading model and tokenizer: {model_name}")

    # D-001: float32 for correctness on M1
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {n_params:,} (~{n_params/1e6:.2f}M)")

    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer.mask_token_id is None")
    print(f"mask_token_id: {tokenizer.mask_token_id}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.mask_token_id

    # Single forward pass with masked input
    # Match HuggingFace model card: list of messages, encode each, then build tensors
    # https://huggingface.co/dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1
    # apply_chat_template(tokenize=True) can return List[int], str, or BatchEncoding (has input_ids)
    def _to_token_ids(e):
        if isinstance(e, list) and (not e or isinstance(e[0], int)):
            return e
        if isinstance(e, str):
            return tokenizer.encode(e, add_special_tokens=False)
        # BatchEncoding (dict-like or has input_ids)
        ids = getattr(e, "input_ids", None) or (e.get("input_ids") if isinstance(e, dict) else None)
        if ids is not None:
            if hasattr(ids, "tolist"):
                ids = ids.squeeze().tolist()
            return ids if isinstance(ids, list) else list(ids)
        if isinstance(e, (list, tuple)) and e:
            return list(e)
        raise TypeError(f"apply_chat_template returned unexpected type: {type(e)}")

    messages = [[{"role": "user", "content": "def fibonacci(n):"}]]
    encoded_list = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
        for m in messages
    ]
    encoded_list = [_to_token_ids(e) for e in encoded_list]
    prompt_lens = torch.tensor([len(e) for e in encoded_list], dtype=torch.long)
    max_prompt_len = prompt_lens.max().item()
    prompt_ids = torch.full((len(encoded_list), max_prompt_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(encoded_list):
        prompt_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    prompt_ids = prompt_ids.to(device)
    prompt_lens = prompt_lens.to(device)
    encoded = encoded_list[0]  # single batch for rest of script
    max_new_tokens = 64
    total_len = prompt_lens.item() + max_new_tokens
    input_ids = torch.full(
        (1, total_len), pad_id, dtype=torch.long, device=device
    )
    input_ids[0, : prompt_lens.item()] = prompt_ids[0]
    input_ids[0, prompt_lens.item() : prompt_lens.item() + max_new_tokens] = (
        tokenizer.mask_token_id
    )
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[0, : prompt_lens.item() + max_new_tokens] = 1

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    print(f"Forward pass logits shape: {logits.shape}")
    assert logits.dim() == 3 and logits.shape[2] > 1000, "Expected [1, seq_len, vocab_size]"

    # Generate short completion (reuse prompt_ids / prompt_lens from above)
    print("Generating 64 new tokens (block_size=64, steps=64)...")
    generated = generate(
        model,
        tokenizer,
        prompt_ids,
        prompt_lens,
        pad_id=pad_id,
        mask_id=tokenizer.mask_token_id,
        device=device,
        steps=64,
        max_new_tokens=64,
        block_size=64,
        temperature=0.0,
        remasking="low_confidence",
    )

    response_tokens = generated[
        0, prompt_lens[0].item() : prompt_lens[0].item() + max_new_tokens
    ].tolist()
    decoded = tokenizer.decode(response_tokens, skip_special_tokens=True)
    print("Generated completion (first 64 tokens):")
    print(decoded[:500])
    print("\nDone. Model validation passed.")


if __name__ == "__main__":
    main()
