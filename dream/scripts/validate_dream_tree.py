#!/usr/bin/env python3
"""
Phase 1/2 Dream integration validation: build a small entropy-guided tree
using the real Dream model (Dream-v0-Instruct-7B).

Confirms:
- Model loads on GPU
- ModelAdapter forward_logits + right-shift works end-to-end inside tree
- Entropy is computed for root and intermediate nodes
- Tree can expand with a small budget without crashing

Run from repo root:
  python dream/scripts/validate_dream_tree.py
"""

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


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
        print(f"Using HF_HOME={hf_home}")
    except Exception:
        pass


_configure_hf_cache()

import math
import torch


def count_nodes(node) -> int:
    return 1 + sum(count_nodes(c) for c in node.children)


def main():
    p = argparse.ArgumentParser(description="Validate Dream tree building (small tree).")
    p.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--prompt", type=str, default="Write a Python function to check if a number is prime.")
    p.add_argument("--max-tree-nodes", type=int, default=5)
    p.add_argument("--branch-width", type=int, default=2)
    p.add_argument("--steps-per-expansion", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--adaptive-stepping", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available; this validation is intended for GPU.")

    print("Loading Dream model/tokenizer...")
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    from dream.src.config import MCTSConfig
    from dream.src.model_adapter import ModelAdapter
    from dream.src.entropy import EntropyComputer
    from dream.src.tree_builder import EntropyGuidedTreeBuilder

    cfg = MCTSConfig(
        model_type="dream",
        model_name_or_path=args.model,
        device=device,
        max_tree_nodes=args.max_tree_nodes,
        branch_width=args.branch_width,
        steps_per_expansion=args.steps_per_expansion,
        max_new_tokens=args.max_new_tokens,
        total_denoising_steps=min(256, args.max_new_tokens),
        adaptive_stepping=bool(args.adaptive_stepping),
    )

    adapter = ModelAdapter(model, tokenizer, model_type="dream")
    entropy_computer = EntropyComputer()
    builder = EntropyGuidedTreeBuilder(adapter, tokenizer, cfg, entropy_computer)

    print("Building tree...")
    root, leaves = builder.build_tree(args.prompt)

    n_nodes = count_nodes(root)
    n_leaves = len(leaves)
    entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]

    print(f"Tree summary: nodes={n_nodes}, leaves={n_leaves}")
    if entropies:
        print(
            f"Entropy summary: mean={sum(entropies)/len(entropies):.4f}, "
            f"min={min(entropies):.4f}, max={max(entropies):.4f}"
        )

    # Minimal functional checks (no strict thresholds; real values depend on prompt).
    ok = root.entropy is not None and n_leaves >= 1 and all(l.entropy is not None for l in leaves)
    if not ok:
        print("WARNING: Some entropy/node checks failed (see printed summaries).")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

