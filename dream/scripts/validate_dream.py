#!/usr/bin/env python3
"""
Phase 0: Validate Dream 7B forward pass and adapter stack.

Run on a GPU machine with the Dream env active. Confirms:
- Model loads (Dream-org/Dream-v0-Instruct-7B)
- diffusion_generate produces coherent output
- ModelAdapter forward_logits + right-shift gives correct logits shape
- Entropy over masked positions is in [0, log(V)]
- Optional: entropy decreases after partial denoising

Usage (from repo root):
  python dream/scripts/validate_dream.py

Or from dream/:
  python scripts/validate_dream.py
"""
import math
import sys
from pathlib import Path

# Ensure repo root is on path when running from dream/scripts
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch


def main():
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA. Dream 7B is large; validation may be slow or OOM on CPU.")

    print("Loading model and tokenizer...")
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    V = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    log_V = math.log(V)
    mask_id = tokenizer.mask_token_id
    print(f"Vocab size V = {V}, log(V) = {log_V:.4f}, mask_id = {mask_id}")

    # ---- Test 1: diffusion_generate ----
    print("\n--- Test 1: diffusion_generate ---")
    messages = [{"role": "user", "content": "Write a Python function to compute fibonacci numbers."}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            steps=128,
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
            return_dict_in_generate=True,
        )
    gen_ids = output.sequences[0][input_ids.shape[1] :]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("Generated (first 400 chars):")
    print(generated[:400])
    print("  ... [ok] diffusion_generate ran")

    # ---- Test 2: Raw forward + right-shift + entropy (adapter-style) ----
    print("\n--- Test 2: Forward + right-shift + entropy ---")
    prompt_len = input_ids.shape[1]
    L = prompt_len + 64
    x = torch.full((1, L), mask_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = input_ids[0]

    with torch.no_grad():
        # Dream raw forward (no right-shift in model). Pass tok_idx like ModelAdapter.
        B, L = x.shape
        tok_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        out = model(x, attention_mask="full", tok_idx=tok_idx)
        logits_raw = out.logits
        # Apply right-shift as in ModelAdapter
        logits = torch.cat([logits_raw[:, :1], logits_raw[:, :-1]], dim=1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        mask_positions = x[0] == mask_id
        n_masked = mask_positions.sum().item()
        mean_entropy = entropy[0][mask_positions].mean().item()

    print(f"  logits shape: {logits.shape} (expected [1, {L}, V])")
    print(f"  Masked positions: {n_masked}")
    print(f"  Mean entropy (masked): {mean_entropy:.4f}")
    print(f"  Expected range: [0, {log_V:.4f}]")

    checks = []
    checks.append(("logits shape [1, L, V]", logits.shape[0] == 1 and logits.shape[1] == L and logits.shape[2] == V))
    checks.append(("entropy in range", 0 <= mean_entropy <= log_V * 1.01))

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # ---- Test 3: ModelAdapter + EntropyComputer (our stack) ----
    print("\n--- Test 3: ModelAdapter + EntropyComputer ---")
    from dream.src.config import MCTSConfig
    from dream.src.model_adapter import ModelAdapter
    from dream.src.entropy import EntropyComputer

    cfg = MCTSConfig(model_type="dream", model_name_or_path=model_path, device=device)
    adapter = ModelAdapter(model, tokenizer, model_type="dream")
    entropy_computer = EntropyComputer()

    with torch.no_grad():
        logits_adapter = adapter.forward_logits(x, torch.ones_like(x))
        token_entropy = entropy_computer.compute_token_entropy_from_logits(logits_adapter)
    mean_entropy_adapter = token_entropy[0][mask_positions].mean().item()
    print(f"  Adapter mean entropy (masked): {mean_entropy_adapter:.4f}")
    checks.append(("adapter entropy in range", 0 <= mean_entropy_adapter <= log_V * 1.01))

    all_ok = all(ok for _, ok in checks)
    print("\n" + ("All checks PASSED." if all_ok else "Some checks FAILED."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
