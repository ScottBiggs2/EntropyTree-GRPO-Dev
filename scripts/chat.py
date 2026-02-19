"""
Simple terminal chat with the diffusion model (Option 1: use with USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=./model_cache).

Usage:
  USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=./model_cache python scripts/chat.py

Type your message and press Enter. "quit" or "exit" to stop.
"""
import os
import sys
import warnings
from pathlib import Path

# Avoid "Logging error" from transformers' deprecated attention-mask API (they pass FutureWarning into a %-format string)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from validate_model import get_device, generate


def _to_token_ids(tokenizer, e):
    """Normalize apply_chat_template output to list of token ids."""
    if isinstance(e, list) and (not e or isinstance(e[0], int)):
        return e
    if isinstance(e, str):
        return tokenizer.encode(e, add_special_tokens=False)
    ids = getattr(e, "input_ids", None) or (e.get("input_ids") if isinstance(e, dict) else None)
    if ids is not None:
        if hasattr(ids, "tolist"):
            ids = ids.squeeze().tolist()
        return ids if isinstance(ids, list) else list(ids)
    if isinstance(e, (list, tuple)) and e:
        return list(e)
    raise TypeError(f"apply_chat_template returned unexpected type: {type(e)}")


def load_model_and_tokenizer():
    """Same env and logic as validate_model.py."""
    device = get_device()
    if os.environ.get("USE_LOCAL_MODEL_CODE") == "1":
        model_name = os.environ.get("LOCAL_MODEL_PATH", "model_cache")
    else:
        model_name = "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"

    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer.mask_token_id is None")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.mask_token_id
    return model, tokenizer, device, pad_id


def run_turn(model, tokenizer, messages, device, pad_id, max_new_tokens=256, steps=256, block_size=64, temperature=0.3):
    """Encode messages, generate one reply, return (response_text, messages_with_assistant)."""
    encoded_list = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
        for m in messages
    ]
    encoded_list = [_to_token_ids(tokenizer, e) for e in encoded_list]
    prompt_lens = torch.tensor([len(e) for e in encoded_list], dtype=torch.long)
    max_prompt_len = prompt_lens.max().item()
    prompt_ids = torch.full((len(encoded_list), max_prompt_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(encoded_list):
        prompt_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    prompt_ids = prompt_ids.to(device)
    prompt_lens = prompt_lens.to(device)

    generated = generate(
        model,
        tokenizer,
        prompt_ids,
        prompt_lens,
        pad_id=pad_id,
        mask_id=tokenizer.mask_token_id,
        device=device,
        steps=steps,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        remasking="low_confidence",
    )
    response_tokens = generated[
        0, prompt_lens[0].item() : prompt_lens[0].item() + max_new_tokens
    ].tolist()
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    return response_text, messages + [{"role": "assistant", "content": response_text}]


def main():
    print("Loading model (same as validate_model)...")
    model, tokenizer, device, pad_id = load_model_and_tokenizer()
    print(f"Device: {device}. Type your message and press Enter. 'quit' or 'exit' to stop.\n")

    messages = []
    max_new_tokens = 256
    steps = 256
    block_size = 64
    temperature = 0.3

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        messages = messages + [{"role": "user", "content": user_input}]
        # Single batch of one conversation
        turn_messages = [messages]

        print("Model: ", end="", flush=True)
        response_text, _ = run_turn(
            model, tokenizer, turn_messages, device, pad_id,
            max_new_tokens=max_new_tokens, steps=steps, block_size=block_size, temperature=temperature,
        )
        print(response_text)
        messages = messages + [{"role": "assistant", "content": response_text}]


if __name__ == "__main__":
    main()
