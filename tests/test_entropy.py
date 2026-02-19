"""Phase 2: EntropyComputer tests (no model for pure math tests)."""

import math
import torch
import torch.nn.functional as F
import pytest
from src.entropy import EntropyComputer


def test_expected_entropy_full_mask():
    # masking_ratio=1.0, vocab 50000 -> log(50000) ~ 10.82
    h = EntropyComputer.expected_entropy(1.0, 50000)
    assert abs(h - math.log(50000)) < 0.01


def test_expected_entropy_zero():
    assert EntropyComputer.expected_entropy(0.0, 50000) == 0.0


def test_aggregate_entropy_mean():
    e = torch.tensor([[1.0, 2.0, 3.0]])
    assert abs(EntropyComputer.aggregate_entropy(e, method="mean") - 2.0) < 1e-5


def test_aggregate_entropy_max():
    e = torch.tensor([[1.0, 2.0, 3.0]])
    assert EntropyComputer.aggregate_entropy(e, method="max") == 3.0


def test_aggregate_entropy_mean_over_masked_only():
    token_entropy = torch.tensor([[0.0, 5.0, 10.0]])  # pos 0 unmasked, 1,2 masked
    mask_positions = torch.tensor([[False, True, True]])
    mean = EntropyComputer.aggregate_entropy(
        token_entropy, mask_positions=mask_positions, method="mean"
    )
    assert abs(mean - 7.5) < 1e-5  # (5+10)/2


def test_compute_entropy_weight():
    # measured > expected -> weight > 1
    w = EntropyComputer.compute_entropy_weight(7.5, 0.5, 50000)
    expected = 0.5 * math.log(50000)
    assert w > 1.0
    assert abs(w - 7.5 / expected) < 0.01


def test_compute_entropy_weight_zero_expected():
    w = EntropyComputer.compute_entropy_weight(1.0, 0.0, 50000)
    assert w == 0.0


def test_uniform_distribution_entropy():
    """Uniform over vocab -> entropy = log(vocab_size)."""
    vocab_size = 100
    logits = torch.zeros(1, 1, vocab_size)
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    assert abs(entropy.item() - math.log(vocab_size)) < 0.01


def test_onehot_entropy_zero():
    """One-hot -> entropy 0."""
    logits = torch.full((1, 1, 10), -1e9)
    logits[0, 0, 0] = 1e9
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    assert entropy.item() < 0.01


def test_compute_token_entropy_mock_model():
    """Test compute_token_entropy with a mock model that returns logits."""

    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size=10):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids, attention_mask=None):
            # Return uniform logits -> high entropy
            b, s, _ = input_ids.shape[0], input_ids.shape[1], self.vocab_size
            logits = torch.zeros(b, s, self.vocab_size)
            return type("Out", (), {"logits": logits})()

    model = MockModel(vocab_size=10)
    input_ids = torch.randint(0, 10, (1, 5))
    attn = torch.ones(1, 5)
    entropy = EntropyComputer.compute_token_entropy(model, input_ids, attn)
    assert entropy.shape == (1, 5)
    # Uniform logits -> entropy = log(10) ~ 2.30
    assert abs(entropy.mean().item() - math.log(10)) < 0.1
