"""Phase 5: Advantage computation tests (synthetic tree, no model)."""

import torch
import pytest
from src.tree_node import MCTSNode
from src.advantages import AdvantageComputer


def _make_synthetic_tree():
    """Root -> 2 children (L1, L2) with rewards; no state needed for advantage calc."""
    mask_id = 2
    root = MCTSNode(
        state=torch.zeros(5),
        attention_mask=torch.ones(5),
        prompt_len=2,
        step_index=0,
        mask_id=mask_id,
    )
    c1 = MCTSNode(
        state=torch.zeros(5),
        attention_mask=torch.ones(5),
        prompt_len=2,
        step_index=1,
        parent=root,
        mask_id=mask_id,
        sampling_prob=0.5,
    )
    c2 = MCTSNode(
        state=torch.zeros(5),
        attention_mask=torch.ones(5),
        prompt_len=2,
        step_index=1,
        parent=root,
        mask_id=mask_id,
        sampling_prob=0.5,
    )
    root.children = [c1, c2]
    return root, [c1, c2]


def test_simple_leaf_advantages_zero_mean():
    root, leaves = _make_synthetic_tree()
    rewards = [1.0, -1.0]  # mean 0
    AdvantageComputer.compute_advantages(root, leaves, rewards, mode="simple")
    assert leaves[0].advantage == 1.0
    assert leaves[1].advantage == -1.0
    assert sum(l.advantage for l in leaves) == 0.0


def test_simple_internal_mean_of_children():
    root, leaves = _make_synthetic_tree()
    rewards = [2.0, 0.0]
    AdvantageComputer.compute_advantages(root, leaves, rewards, mode="simple")
    # mean reward = 1, so leaf advantages = 1 and -1; root = mean(children) = 0
    assert root.advantage == pytest.approx(0.0, abs=1e-5)


def test_branchgrpo_fused_respects_weights():
    root, leaves = _make_synthetic_tree()
    leaves[0].reward = 1.0
    leaves[1].reward = 0.0
    AdvantageComputer._fuse_rewards_path_weighted(root)
    # root.fused = 0.5*1 + 0.5*0 = 0.5
    assert root.fused_reward == 0.5
    assert leaves[0].fused_reward == 1.0
    assert leaves[1].fused_reward == 0.0


def test_branchgrpo_depth_norm_zero_mean_unit_var():
    root, leaves = _make_synthetic_tree()
    rewards = [1.0, 3.0]  # mean 2, so at depth 1 advantages will be normalized
    AdvantageComputer.compute_advantages(root, leaves, rewards, mode="branchgrpo")
    # At depth 1 (leaves): fused = [1, 3], mean=2, std=1.414 -> adv = (1-2)/1.414, (3-2)/1.414
    import numpy as np
    leaf_advs = [leaves[0].advantage, leaves[1].advantage]
    assert np.mean(leaf_advs) == pytest.approx(0.0, abs=1e-5)
    assert np.std(leaf_advs) == pytest.approx(1.0, abs=1e-5)
