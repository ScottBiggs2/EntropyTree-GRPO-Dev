import torch

from dream.src.observability import (
    leaf_path_signature,
    leaf_schedule_signature,
    tree_diversity_metrics,
)
from dream.src.tree_node import MCTSNode


def _node(state, prompt_len=1, step_index=0, parent=None, mask_id=99):
    return MCTSNode(
        state=torch.tensor(state, dtype=torch.long),
        attention_mask=torch.ones(len(state), dtype=torch.long),
        prompt_len=prompt_len,
        step_index=step_index,
        parent=parent,
        mask_id=mask_id,
    )


def test_same_final_tokens_can_have_distinct_path_and_schedule():
    root = _node([1, 99, 99], step_index=0)

    a = _node([1, 2, 99], step_index=1, parent=root)
    a.steps_in_edge = 1
    leaf1 = _node([1, 2, 3], step_index=2, parent=a)
    leaf1.steps_in_edge = 1

    b = _node([1, 99, 3], step_index=1, parent=root)
    b.steps_in_edge = 1
    leaf2 = _node([1, 2, 3], step_index=2, parent=b)
    leaf2.steps_in_edge = 1

    root.children = [a, b]
    a.children = [leaf1]
    b.children = [leaf2]

    assert leaf_path_signature(leaf1) != leaf_path_signature(leaf2)
    assert leaf_schedule_signature(leaf1) != leaf_schedule_signature(leaf2)

    metrics = tree_diversity_metrics(root, [leaf1, leaf2])
    assert metrics["leaf_text_unique_frac"] == 0.5
    assert metrics["leaf_path_unique_frac"] == 1.0
    assert metrics["leaf_schedule_unique_frac"] == 1.0
    assert metrics["avg_pairwise_leaf_hamming"] == 0.0
    assert metrics["avg_pairwise_schedule_distance"] > 0.0


def test_sibling_overlap_metrics_reflect_distinct_position_updates():
    root = _node([1, 99, 99], step_index=0)
    c1 = _node([1, 2, 99], step_index=1, parent=root)
    c1.steps_in_edge = 1
    c2 = _node([1, 99, 3], step_index=1, parent=root)
    c2.steps_in_edge = 1
    root.children = [c1, c2]

    metrics = tree_diversity_metrics(root, [c1, c2])
    assert metrics["mean_sibling_position_overlap"] == 0.0
    assert metrics["mean_sibling_token_agreement"] == 0.0
    assert metrics["unique_steps_in_edge_count"] == 1.0
