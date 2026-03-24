"""Observability helpers for Dream diffusion-tree diversity.

Key distinction:
- output diversity: do leaves decode to different responses?
- trajectory diversity: did the denoising paths differ, even if the final text matched?
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from dream.src.tree_node import MCTSNode


def _response_tokens(node: MCTSNode) -> Tuple[int, ...]:
    valid = int(node.attention_mask.sum().item())
    toks = node.state[node.prompt_len:valid].detach().cpu().tolist()
    return tuple(int(x) for x in toks)


def _edge_change_positions(parent: MCTSNode, child: MCTSNode) -> Tuple[int, ...]:
    changed = ((parent.state != child.state) & (parent.state == parent.mask_id)).nonzero(
        as_tuple=False
    )
    return tuple(int(i.item()) for i in changed.view(-1))


def _edge_change_tokens(parent: MCTSNode, child: MCTSNode) -> Tuple[int, ...]:
    positions = _edge_change_positions(parent, child)
    return tuple(int(child.state[pos].item()) for pos in positions)


def _leaf_path_nodes(leaf: MCTSNode) -> List[MCTSNode]:
    path: List[MCTSNode] = []
    cur = leaf
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    return path


def leaf_path_signature(leaf: MCTSNode) -> Tuple[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]], ...]:
    path = _leaf_path_nodes(leaf)
    sig = []
    for parent, child in zip(path[:-1], path[1:]):
        sig.append(
            (
                int(parent.step_index),
                int(child.step_index),
                _edge_change_positions(parent, child),
                _edge_change_tokens(parent, child),
            )
        )
    return tuple(sig)


def leaf_schedule_signature(leaf: MCTSNode) -> Tuple[Tuple[int, int, int], ...]:
    """Per-position token-commit schedule.

    Returns tuples of:
    (response_local_position, first_child_step_index_where_committed, final_token_id)
    """
    path = _leaf_path_nodes(leaf)
    schedule: List[Tuple[int, int, int]] = []
    first_commit: Dict[int, int] = {}
    final_tokens = _response_tokens(leaf)
    for parent, child in zip(path[:-1], path[1:]):
        for pos in _edge_change_positions(parent, child):
            if pos >= leaf.prompt_len and pos not in first_commit:
                first_commit[pos] = int(child.step_index)
    for local_pos, token in enumerate(final_tokens):
        global_pos = leaf.prompt_len + local_pos
        schedule.append((local_pos, int(first_commit.get(global_pos, -1)), int(token)))
    return tuple(schedule)


def _pairwise_hamming(seqs: Sequence[Tuple[int, ...]]) -> float:
    pairs = list(combinations(seqs, 2))
    if not pairs:
        return 0.0
    total = 0.0
    for a, b in pairs:
        n = max(len(a), len(b))
        if n == 0:
            continue
        diff = sum(1 for i in range(n) if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None))
        total += diff / n
    return total / len(pairs)


def _pairwise_schedule_distance(schedules: Sequence[Tuple[Tuple[int, int, int], ...]]) -> float:
    pairs = list(combinations(schedules, 2))
    if not pairs:
        return 0.0
    total = 0.0
    for a, b in pairs:
        n = max(len(a), len(b))
        if n == 0:
            continue
        dist = 0.0
        for i in range(n):
            sa = a[i] if i < len(a) else (-1, -1, -1)
            sb = b[i] if i < len(b) else (-1, -1, -1)
            # emphasize different commit steps even when final token matches
            dist += 0.0 if sa == sb else (0.5 if sa[2] == sb[2] else 1.0)
        total += dist / n
    return total / len(pairs)


def _sibling_overlap_metrics(root: MCTSNode) -> Tuple[float, float]:
    overlaps: List[float] = []
    token_agreements: List[float] = []

    def walk(node: MCTSNode) -> None:
        if len(node.children) >= 2:
            edge_positions = [_edge_change_positions(node, child) for child in node.children]
            edge_tokens = [_edge_change_tokens(node, child) for child in node.children]
            for (pa, pb), (ta, tb) in zip(combinations(edge_positions, 2), combinations(edge_tokens, 2)):
                sa, sb = set(pa), set(pb)
                union = sa | sb
                inter = sa & sb
                overlaps.append((len(inter) / len(union)) if union else 0.0)

                tok_map_a = dict(zip(pa, ta))
                tok_map_b = dict(zip(pb, tb))
                shared = sorted(inter)
                if shared:
                    agree = sum(1 for pos in shared if tok_map_a.get(pos) == tok_map_b.get(pos))
                    token_agreements.append(agree / len(shared))
                else:
                    token_agreements.append(0.0)
        for child in node.children:
            walk(child)

    walk(root)
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    mean_agreement = sum(token_agreements) / len(token_agreements) if token_agreements else 0.0
    return mean_overlap, mean_agreement


def tree_diversity_metrics(root: MCTSNode, leaves: Iterable[MCTSNode]) -> Dict[str, float]:
    leaves = list(leaves)
    if not leaves:
        return {
            "leaf_text_unique_frac": 0.0,
            "leaf_path_unique_frac": 0.0,
            "leaf_schedule_unique_frac": 0.0,
            "avg_pairwise_leaf_hamming": 0.0,
            "avg_pairwise_schedule_distance": 0.0,
            "mean_sibling_position_overlap": 0.0,
            "mean_sibling_token_agreement": 0.0,
            "unique_steps_in_edge_count": 0.0,
        }

    leaf_tokens = [_response_tokens(leaf) for leaf in leaves]
    path_sigs = [leaf_path_signature(leaf) for leaf in leaves]
    sched_sigs = [leaf_schedule_signature(leaf) for leaf in leaves]

    steps_in_edge: List[int] = []

    def collect(node: MCTSNode) -> None:
        for child in node.children:
            if child.steps_in_edge is not None:
                steps_in_edge.append(int(child.steps_in_edge))
            collect(child)

    collect(root)

    mean_overlap, mean_agreement = _sibling_overlap_metrics(root)

    n = max(len(leaves), 1)
    return {
        "leaf_text_unique_frac": len(set(leaf_tokens)) / n,
        "leaf_path_unique_frac": len(set(path_sigs)) / n,
        "leaf_schedule_unique_frac": len(set(sched_sigs)) / n,
        "avg_pairwise_leaf_hamming": _pairwise_hamming(leaf_tokens),
        "avg_pairwise_schedule_distance": _pairwise_schedule_distance(sched_sigs),
        "mean_sibling_position_overlap": mean_overlap,
        "mean_sibling_token_agreement": mean_agreement,
        "unique_steps_in_edge_count": float(len(set(steps_in_edge))) if steps_in_edge else 0.0,
    }
