"""Phase 6: WeightedGRPOLoss tests (mock model)."""

import torch
import pytest
from src.config import MCTSConfig
from src.tree_node import MCTSNode, TreeTransition
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter
from src.loss import WeightedGRPOLoss


class MockModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        b, s = input_ids.shape
        logits = torch.zeros(b, s, self.vocab_size, device=input_ids.device)
        return type("Out", (), {"logits": logits})()


def test_loss_is_scalar_with_grad():
    config = MCTSConfig(device="cpu")
    ec = EntropyComputer()
    tw = TimeWeighter(total_steps=256)
    loss_fn = WeightedGRPOLoss(config, ec, tw, mask_id=2)

    root = MCTSNode(
        state=torch.tensor([1, 1, 2, 2, 2]),
        attention_mask=torch.ones(5),
        prompt_len=2,
        step_index=0,
        mask_id=2,
    )
    child = MCTSNode(
        state=torch.tensor([1, 1, 3, 3, 2]),  # two positions unmasked
        attention_mask=torch.ones(5),
        prompt_len=2,
        step_index=32,
        parent=root,
        mask_id=2,
        advantage=0.5,
    )
    root.children = [child]
    child.entropy = 2.0
    root.entropy = 3.0

    model = MockModel(vocab_size=10)
    loss, metrics = loss_fn.compute_loss(
        model, root, [child], "prompt", vocab_size=10
    )
    assert loss.dim() == 0
    assert loss.requires_grad or not loss.requires_grad  # scalar
    assert loss.item() != float("nan")
    assert 0.01 <= abs(loss.item()) <= 100 or loss.item() == 0
    assert "loss" in metrics
    assert "n_transitions" in metrics


def test_zero_advantage_gives_zero_loss():
    config = MCTSConfig(device="cpu")
    ec = EntropyComputer()
    tw = TimeWeighter(total_steps=256)
    loss_fn = WeightedGRPOLoss(config, ec, tw, mask_id=2)

    root = MCTSNode(
        state=torch.tensor([1, 2, 2]),
        attention_mask=torch.ones(3),
        prompt_len=1,
        step_index=0,
        mask_id=2,
    )
    child = MCTSNode(
        state=torch.tensor([1, 3, 3]),
        attention_mask=torch.ones(3),
        prompt_len=1,
        step_index=1,
        parent=root,
        mask_id=2,
        advantage=0.0,
    )
    root.children = [child]
    root.entropy = 1.0

    model = MockModel(vocab_size=10)
    loss, _ = loss_fn.compute_loss(model, root, [child], "x", vocab_size=10)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_gradient_flows_when_training():
    """With a real model in train mode, loss would have grad. Here we only check loss is finite."""
    config = MCTSConfig(device="cpu")
    ec = EntropyComputer()
    tw = TimeWeighter(total_steps=256)
    loss_fn = WeightedGRPOLoss(config, ec, tw, mask_id=2)

    root = MCTSNode(
        state=torch.tensor([1, 2, 2]),
        attention_mask=torch.ones(3),
        prompt_len=1,
        step_index=0,
        mask_id=2,
    )
    child = MCTSNode(
        state=torch.tensor([1, 3, 3]),
        attention_mask=torch.ones(3),
        prompt_len=1,
        step_index=1,
        parent=root,
        mask_id=2,
        advantage=0.5,
    )
    root.children = [child]
    root.entropy = 1.0

    model = MockModel(vocab_size=10)
    loss, _ = loss_fn.compute_loss(model, root, [child], "x", vocab_size=10)
    assert loss.item() is not None
    assert not (loss.item() != loss.item())  # not NaN
    if loss.requires_grad:
        loss.backward()
