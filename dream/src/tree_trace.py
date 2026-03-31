"""Tree trace exporter for paper/debug artifacts.

Goal: persist enough information to reconstruct and visualize a single tree:
- node/edge structure
- adaptive step intervals (parent/child step_index)
- intermediate decodes
- entropy/time weights used by the loss
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dream.src.config import MCTSConfig
from dream.src.entropy import EntropyComputer
from dream.src.time_weight import TimeWeighter
from dream.src.tree_node import MCTSNode


def _edge_change_positions(parent: MCTSNode, child: MCTSNode) -> List[int]:
    changed = ((parent.state != child.state) & (parent.state == parent.mask_id)).nonzero(
        as_tuple=False
    )
    return [int(i.item()) for i in changed.view(-1)]


def _edge_change_tokens(parent: MCTSNode, child: MCTSNode) -> List[int]:
    positions = _edge_change_positions(parent, child)
    return [int(child.state[pos].item()) for pos in positions]


def _decode_node_response(tokenizer, node: MCTSNode, *, max_new_tokens: int, max_chars: int) -> str:
    start = node.prompt_len
    end = start + max_new_tokens
    toks = node.state[start:end].detach().cpu().tolist()
    text = tokenizer.decode(toks, skip_special_tokens=True)
    if max_chars > 0:
        return text[:max_chars]
    return text


def iter_nodes(root: MCTSNode) -> Iterable[MCTSNode]:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        # deterministic-ish ordering for stable traces
        for c in reversed(n.children):
            stack.append(c)


def build_tree_trace(
    *,
    root: MCTSNode,
    leaves: List[MCTSNode],
    tokenizer,
    cfg: MCTSConfig,
    vocab_size: int,
    prompt: str,
    phase: str,
    step_id: int,
    prompt_idx: int,
    epoch: int,
    entropy_computer: Optional[EntropyComputer] = None,
    time_weighter: Optional[TimeWeighter] = None,
    max_nodes: int = 0,
    max_leaves: int = 0,
    decode_chars: int = 240,
) -> Dict[str, Any]:
    """Return a JSON-serializable dict describing the tree."""
    entropy_computer = entropy_computer or EntropyComputer()
    time_weighter = time_weighter or TimeWeighter(cfg.total_denoising_steps, norm_mode=cfg.time_weight_norm)  # type: ignore[arg-type]

    node_ids: Dict[int, str] = {}

    def node_key(n: MCTSNode) -> str:
        k = id(n)
        if k not in node_ids:
            node_ids[k] = f"n{len(node_ids)}"
        return node_ids[k]

    nodes_out: List[Dict[str, Any]] = []
    edges_out: List[Dict[str, Any]] = []

    # Optionally downsample leaves (but keep them as nodes if present in tree).
    leaves_use = leaves
    if max_leaves and len(leaves_use) > max_leaves:
        leaves_use = leaves_use[:max_leaves]

    # Traverse nodes, optionally cap count.
    for i, n in enumerate(iter_nodes(root)):
        if max_nodes and i >= max_nodes:
            break
        nk = node_key(n)
        n_masked = int(n.num_masked_tokens())
        nodes_out.append(
            {
                "id": nk,
                "parent": node_key(n.parent) if n.parent is not None else None,
                "depth": int(getattr(n, "depth", 0)),
                "step_index": int(n.step_index),
                "steps_in_edge": int(n.steps_in_edge) if n.steps_in_edge is not None else None,
                "masking_ratio": float(n.masking_ratio()),
                "n_masked": int(n_masked),
                "entropy": float(n.entropy) if n.entropy is not None else None,
                "reward": float(n.reward) if n.reward is not None else None,
                "fused_reward": float(n.fused_reward) if n.fused_reward is not None else None,
                "advantage": float(n.advantage) if n.advantage is not None else None,
                "is_completed": bool(getattr(n, "is_completed", False)),
                "decode": _decode_node_response(
                    tokenizer, n, max_new_tokens=cfg.max_new_tokens, max_chars=decode_chars
                ),
            }
        )

        if n.parent is not None:
            parent = n.parent
            start = int(parent.step_index)
            end = int(n.step_index)
            w_time = float(time_weighter.get_interval_weight(start, end))
            ew_raw = float(
                entropy_computer.compute_entropy_weight(
                    measured_masked_mean=float(parent.entropy or 0.0),
                    vocab_size=int(vocab_size),
                    masking_ratio=float(parent.masking_ratio()),
                    mode=str(cfg.entropy_norm_mode),
                )
            )
            ew = max(float(cfg.entropy_weight_min), min(float(cfg.entropy_weight_max), ew_raw))
            w = float(cfg.alpha_time) * w_time + float(cfg.alpha_entropy) * ew
            edges_out.append(
                {
                    "parent": node_key(parent),
                    "child": nk,
                    "step_start": start,
                    "step_end": end,
                    "time_weight": w_time,
                    "entropy_weight_raw": ew_raw,
                    "entropy_weight": ew,
                    "combined_weight": w,
                    "changed_positions": _edge_change_positions(parent, n),
                    "changed_tokens": _edge_change_tokens(parent, n),
                }
            )

    leaf_ids = [node_key(l) for l in leaves_use]
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "step_id": int(step_id),
        "epoch": int(epoch),
        "prompt_idx": int(prompt_idx),
        "prompt_text": prompt,
        "cfg": {k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, bool, str)) or v is None},
        "vocab_size": int(vocab_size),
        "root_id": node_key(root),
        "leaf_ids": leaf_ids,
        "nodes": nodes_out,
        "edges": edges_out,
    }
    return payload


def write_tree_trace_json(path: str | Path, payload: Dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(payload, f, indent=2)
    return str(p)

