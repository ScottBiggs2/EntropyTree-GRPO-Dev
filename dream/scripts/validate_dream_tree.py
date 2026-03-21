#!/usr/bin/env python3
"""
Phase 1/2 Dream integration validation: build a small entropy-guided tree
using the real Dream model (Dream-v0-Instruct-7B).

Confirms:
- Model loads on GPU
- ModelAdapter forward_logits + right-shift works end-to-end inside tree
- Entropy is computed for root and intermediate nodes
- Tree can expand with a small budget without crashing
- Optional: adaptive stepping (variable steps_in_edge) vs fixed steps_per_expansion
- Optional: LoRA-wrapped model (same tree code path as training)

Run from repo root:
  python dream/scripts/validate_dream_tree.py
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

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

import torch


def count_nodes(node) -> int:
    return 1 + sum(count_nodes(c) for c in node.children)


def collect_steps_in_edges(node) -> List[int]:
    """Gather steps_in_edge from every child link (adaptive stepping diagnostic)."""
    out: List[int] = []
    for c in node.children:
        if getattr(c, "steps_in_edge", None) is not None:
            out.append(int(c.steps_in_edge))
        out.extend(collect_steps_in_edges(c))
    return out


def main():
    p = argparse.ArgumentParser(description="Validate Dream tree building (small tree).")
    p.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to check if a number is prime.",
    )
    p.add_argument("--max-tree-nodes", type=int, default=5)
    p.add_argument("--branch-width", type=int, default=2)
    p.add_argument("--steps-per-expansion", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--adaptive-stepping", action="store_true")
    p.add_argument(
        "--branch-threshold",
        type=float,
        default=0.65,
        help="Adaptive: early-stop when H_masked_mean/log(V) > this after min_steps (typical 0.5–0.8)",
    )
    p.add_argument(
        "--min-steps-per-expansion",
        type=int,
        default=8,
        help="Adaptive: minimum micro-steps per expansion",
    )
    p.add_argument(
        "--max-steps-per-expansion",
        type=int,
        default=48,
        help="Adaptive: maximum micro-steps per expansion",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="Load model with PEFT LoRA (same as training smoke test)",
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available; this validation is intended for GPU.")

    print("Loading Dream model/tokenizer...")
    from dream.src.config import MCTSConfig
    from dream.src.model_adapter import ModelAdapter
    from dream.src.entropy import EntropyComputer
    from dream.src.tree_builder import EntropyGuidedTreeBuilder
    from dream.src.utils import load_model_and_tokenizer

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
        branch_threshold=args.branch_threshold,
        min_steps_per_expansion=args.min_steps_per_expansion,
        max_steps_per_expansion=args.max_steps_per_expansion,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model, tokenizer = load_model_and_tokenizer(cfg)
    model.eval()

    adapter = ModelAdapter(model, tokenizer, model_type="dream")
    entropy_computer = EntropyComputer()
    builder = EntropyGuidedTreeBuilder(adapter, tokenizer, cfg, entropy_computer)

    mode = "adaptive" if cfg.adaptive_stepping else "fixed"
    print(f"Building tree (stepping={mode}, branch_threshold={cfg.branch_threshold})...")
    root, leaves = builder.build_tree(args.prompt)

    n_nodes = count_nodes(root)
    n_leaves = len(leaves)
    entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]
    edge_steps = collect_steps_in_edges(root)

    print(f"Tree summary: nodes={n_nodes}, leaves={n_leaves}")
    if entropies:
        print(
            f"Entropy summary: mean={sum(entropies)/len(entropies):.4f}, "
            f"min={min(entropies):.4f}, max={max(entropies):.4f}"
        )
    if edge_steps:
        uniq = sorted(set(edge_steps))
        expansion_edges = [s for s in edge_steps if s > 0]  # exclude 0 (completion or early exit)
        if args.adaptive_stepping:
            expected_range = f"[{cfg.min_steps_per_expansion}, {cfg.max_steps_per_expansion}]"
            in_range = [s for s in expansion_edges if cfg.min_steps_per_expansion <= s <= cfg.max_steps_per_expansion]
            print(
                f"steps_in_edge (all child links): n={len(edge_steps)}, "
                f"unique={uniq}, sample={edge_steps[:12]}"
            )
            if expansion_edges:
                print(
                    f"  expansion edges (excl. 0): {len(expansion_edges)} edges, "
                    f"range {min(expansion_edges)}-{max(expansion_edges)}, "
                    f"expected {expected_range}"
                )
                if len(uniq) > 1:
                    print(
                        f"  ✓ Adaptive variation detected: {len(uniq)} distinct step counts "
                        f"(fixed mode would show 1 unique value = steps_per_expansion)"
                    )
                else:
                    print(
                        f"  ⚠ All expansion edges have same step count — threshold may not be firing "
                        f"(try lower --branch-threshold or check entropy profile)"
                    )
        else:
            expected = cfg.steps_per_expansion
            if len(uniq) == 1 and uniq[0] == expected:
                print(
                    f"steps_in_edge: all {len(edge_steps)} edges = {expected} (fixed stepping ✓)"
                )
            else:
                print(
                    f"steps_in_edge: n={len(edge_steps)}, unique={uniq} "
                    f"(expected all={expected} for fixed; 0s are completion edges or early exits)"
                )
    elif args.adaptive_stepping:
        print("NOTE: no steps_in_edge recorded (unexpected for adaptive expansions).")

    ok = root.entropy is not None and n_leaves >= 1 and all(
        l.entropy is not None for l in leaves
    )
    if not ok:
        print("WARNING: Some entropy/node checks failed (see printed summaries).")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
