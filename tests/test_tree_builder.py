"""Phase 4: EntropyGuidedTreeBuilder tests (mock model/tokenizer, no download)."""

import torch
import pytest
from src.config import MCTSConfig
from src.tree_node import MCTSNode
from src.entropy import EntropyComputer
from src.tree_builder import EntropyGuidedTreeBuilder


class MockModel(torch.nn.Module):
    """Returns uniform logits so entropy is constant; state evolves via argmax."""

    def __init__(self, vocab_size=1000, seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask=None):
        b, s = input_ids.shape
        logits = torch.zeros(b, s, self.vocab_size, device=input_ids.device)
        return type("Out", (), {"logits": logits})()


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    mask_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        # Return a short prompt
        return [10, 11, 12, 13]  # 4 tokens


def _count_nodes(root: MCTSNode) -> int:
    n = 1
    for c in root.children:
        n += _count_nodes(c)
    return n


def test_tree_builder_small_config():
    """Build tree with max_tree_nodes=7, branch_width=2, steps_per_expansion=8."""
    device = "cpu"
    config = MCTSConfig(
        max_tree_nodes=7,
        branch_width=2,
        steps_per_expansion=8,
        max_new_tokens=16,
        device=device,
    )
    model = MockModel(vocab_size=100, seq_len=4 + 16).to(device)
    tokenizer = MockTokenizer()
    entropy_computer = EntropyComputer()
    builder = EntropyGuidedTreeBuilder(model, tokenizer, config, entropy_computer)

    root, leaves = builder.build_tree("hello")

    assert root is not None
    assert root.mask_id == 2
    assert root.masking_ratio() == 1.0
    assert len(leaves) >= 1
    total_nodes = _count_nodes(root)
    assert total_nodes <= 7 + 10  # may have completion nodes added
    for leaf in leaves:
        assert leaf.is_completed or leaf.num_masked_tokens() == 0
    assert root.entropy is not None


def test_tree_has_multiple_leaves():
    config = MCTSConfig(
        max_tree_nodes=10,
        branch_width=2,
        steps_per_expansion=4,
        max_new_tokens=8,
        device="cpu",
    )
    model = MockModel(vocab_size=50, seq_len=4 + 8).to("cpu")
    tokenizer = MockTokenizer()
    builder = EntropyGuidedTreeBuilder(
        model, tokenizer, config, EntropyComputer()
    )
    root, leaves = builder.build_tree("hi")
    assert len(leaves) >= 1
    # With branch_width=2 we should get at least 2 leaves after one expansion
    assert _count_nodes(root) >= 2


def test_sampling_prob_set_on_children():
    config = MCTSConfig(
        max_tree_nodes=5,
        branch_width=2,
        steps_per_expansion=2,
        max_new_tokens=4,
        device="cpu",
    )
    model = MockModel(vocab_size=20, seq_len=4 + 4).to("cpu")
    tokenizer = MockTokenizer()
    builder = EntropyGuidedTreeBuilder(
        model, tokenizer, config, EntropyComputer()
    )
    root, leaves = builder.build_tree("x")

    def check_sampling_prob(node: MCTSNode):
        for c in node.children:
            assert c.sampling_prob > 0 and c.sampling_prob <= 1.0
            check_sampling_prob(c)

    check_sampling_prob(root)


def test_depth_increments():
    config = MCTSConfig(
        max_tree_nodes=6,
        branch_width=2,
        steps_per_expansion=2,
        max_new_tokens=4,
        device="cpu",
    )
    model = MockModel(vocab_size=20, seq_len=4 + 4).to("cpu")
    tokenizer = MockTokenizer()
    builder = EntropyGuidedTreeBuilder(
        model, tokenizer, config, EntropyComputer()
    )
    root, _ = builder.build_tree("x")
    assert root.depth == 0
    for c in root.children:
        assert c.depth == 1
