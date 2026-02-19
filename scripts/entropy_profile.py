"""
Phase 2: Profile entropy across denoising steps.
Loads model, builds fully masked prompt, runs generate and computes entropy
at step 0 and after 32/64/128 steps (or full 256). Requires model download.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.utils import get_device, add_gumbel_noise
from src.entropy import EntropyComputer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    import os

    device = get_device()
    if os.environ.get("USE_LOCAL_MODEL_CODE") == "1":
        model_name = os.environ.get("LOCAL_MODEL_PATH", "model_cache")
    else:
        model_name = "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"

    print("Loading model (this may download)...")
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.mask_token_id
    mask_id = tokenizer.mask_token_id

    messages = [[{"role": "user", "content": "def fibonacci(n):"}]]
    encoded = tokenizer.apply_chat_template(
        messages[0], add_generation_prompt=True, tokenize=True
    )
    prompt_len = len(encoded)
    max_new_tokens = 128
    total_len = prompt_len + max_new_tokens

    # Build fully masked state (step 0)
    x0 = torch.full((1, total_len), pad_id, dtype=torch.long, device=device)
    x0[0, :prompt_len] = torch.tensor(encoded, device=device)
    x0[0, prompt_len:prompt_len + max_new_tokens] = mask_id
    attn = torch.zeros_like(x0)
    attn[0, :total_len] = 1

    # Entropy at step 0 (fully masked response region)
    with torch.no_grad():
        h0 = EntropyComputer.compute_token_entropy(model, x0, attn)
    mask_pos = (x0[0] == mask_id) & (attn[0].bool())
    agg0 = EntropyComputer.aggregate_entropy(h0[0:1], mask_pos.unsqueeze(0), "mean")
    print(f"Step 0 (fully masked): mean entropy over masked pos = {agg0:.4f}")

    # Full-sequence style: each step unmask (num_masked // steps_remaining) tokens
    for target_step in [32, 64, 128]:
        x = x0.clone()
        response_region = torch.zeros(total_len, dtype=torch.bool, device=device)
        response_region[prompt_len:prompt_len + max_new_tokens] = True

        for step in range(target_step):
            mask_now = (x[0] == mask_id) & response_region
            n_masked = mask_now.sum().item()
            if n_masked == 0:
                break
            k = max(1, n_masked // (target_step - step))
            with torch.no_grad():
                logits = model(x, attention_mask=attn).logits
            logits_n = add_gumbel_noise(logits, temperature=0.0)
            x0_pred = torch.argmax(logits_n, dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0_pred.unsqueeze(-1)).squeeze(-1)
            confidence = torch.where(
                mask_now.unsqueeze(0), x0_p, torch.full_like(x0_p, -1e9)
            )
            x0_pred = torch.where(mask_now.unsqueeze(0), x0_pred, x)
            _, sel = torch.topk(confidence[0], k=min(k, n_masked))
            x[0, sel] = x0_pred[0, sel]

        with torch.no_grad():
            h = EntropyComputer.compute_token_entropy(model, x, attn)
        mask_pos_now = (x[0] == mask_id) & (attn[0].bool())
        if mask_pos_now.any():
            agg = EntropyComputer.aggregate_entropy(
                h[0:1], mask_pos_now.unsqueeze(0), "mean"
            )
            print(f"After {target_step} steps: mean entropy over remaining masked = {agg:.4f}")
        else:
            print(f"After {target_step} steps: no masked positions left")

    print("Done. Entropy should decrease as more tokens are unmasked.")
