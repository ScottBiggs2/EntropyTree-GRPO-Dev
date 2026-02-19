"""Phase 3: MCTSNode and TreeTransition tests."""

import torch
import pytest
from src.tree_node import MCTSNode, TreeTransition


def _make_root(prompt_len: int = 5, response_len: int = 10, mask_id: int = 2):
    """Root: prompt tokens 0..prompt_len-1, then response_len mask tokens."""
    seq_len = prompt_len + response_len
    state = torch.zeros(seq_len, dtype=torch.long)
    state[:prompt_len] = 1  # fake prompt
    state[prompt_len:] = mask_id
    attn = torch.ones(seq_len)
    node = MCTSNode(
        state=state,
        attention_mask=attn,
        prompt_len=prompt_len,
        step_index=0,
        mask_id=mask_id,
    )
    return node


def test_root_masking_ratio_one():
    root = _make_root(prompt_len=5, response_len=10, mask_id=2)
    assert root.masking_ratio() == 1.0
    assert root.num_masked_tokens() == 10


def test_child_lower_masking_ratio():
    root = _make_root(prompt_len=5, response_len=10, mask_id=2)
    # Child: same prompt, 5 masks left (5 unmasked)
    state = root.state.clone()
    state[5:10] = 3  # unmask first half of response
    child = MCTSNode(
        state=state,
        attention_mask=root.attention_mask.clone(),
        prompt_len=5,
        step_index=32,
        parent=root,
        mask_id=2,
        depth=1,
    )
    root.children.append(child)
    assert child.masking_ratio() == 0.5
    assert child.num_masked_tokens() == 5
    assert child.masking_ratio() < root.masking_ratio()


def test_is_leaf():
    root = _make_root(response_len=5, mask_id=2)
    assert root.is_leaf() is True
    child = MCTSNode(
        state=root.state.clone(),
        attention_mask=root.attention_mask.clone(),
        prompt_len=root.prompt_len,
        step_index=1,
        parent=root,
        mask_id=2,
    )
    root.children.append(child)
    assert root.is_leaf() is False
    assert child.is_leaf() is True


def test_depth_tracking():
    root = _make_root(mask_id=2)
    root.depth = 0
    c1 = MCTSNode(
        state=root.state.clone(),
        attention_mask=root.attention_mask.clone(),
        prompt_len=root.prompt_len,
        step_index=1,
        parent=root,
        mask_id=2,
        depth=1,
    )
    root.children.append(c1)
    c2 = MCTSNode(
        state=c1.state.clone(),
        attention_mask=c1.attention_mask.clone(),
        prompt_len=root.prompt_len,
        step_index=2,
        parent=c1,
        mask_id=2,
        depth=2,
    )
    c1.children.append(c2)
    assert root.depth == 0 and c1.depth == 1 and c2.depth == 2


def test_tree_transition():
    parent_state = torch.randint(0, 100, (10,))
    child_state = parent_state.clone()
    child_state[5] = 99
    trans = TreeTransition(
        parent_state=parent_state,
        child_state=child_state,
        parent_attention_mask=torch.ones(10),
        child_attention_mask=torch.ones(10),
        step_index=0,
        advantage=0.5,
        entropy=3.0,
        time_weight=0.1,
        entropy_weight=0.2,
    )
    assert trans.step_index == 0
    assert trans.advantage == 0.5
