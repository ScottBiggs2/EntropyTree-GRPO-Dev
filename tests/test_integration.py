"""Phase 7: Integration test — one training step with mock model (no download)."""

import torch
import pytest
from src.config import MCTSConfig
from src.tree_node import MCTSNode
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter
from src.rewards import SyntaxReward
from src.advantages import AdvantageComputer
from src.loss import WeightedGRPOLoss
from src.tree_builder import EntropyGuidedTreeBuilder
from src.trainer import EntropyMCTSTrainer


class MockModel(torch.nn.Module):
    """Has parameters so loss.backward() runs."""

    def __init__(self, vocab_size=100, seq_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = torch.nn.Parameter(torch.randn(1, seq_len, vocab_size) * 0.01)

    def forward(self, input_ids, attention_mask=None):
        b, s = input_ids.shape
        # Slice to length s so grad flows to embed
        logits = self.embed[:, :s, :].expand(b, s, -1)
        return type("Out", (), {"logits": logits})()


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    mask_token_id = 2
    vocab_size = 50

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return [10, 11, 12]

    def decode(self, ids, skip_special_tokens=True):
        return "def foo(): return 1"


@pytest.mark.slow
def test_one_training_step_completes():
    """Full pipeline: build tree, rewards, advantages, loss, backward, step."""
    config = MCTSConfig(
        max_tree_nodes=5,
        branch_width=2,
        steps_per_expansion=4,
        max_new_tokens=8,
        device="cpu",
    )
    model = MockModel(vocab_size=50, seq_len=3 + 8).to("cpu")
    tokenizer = MockTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ec = EntropyComputer()
    tw = TimeWeighter(256)
    loss_fn = WeightedGRPOLoss(config, ec, tw, tokenizer.mask_token_id)
    trainer = EntropyMCTSTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=SyntaxReward(),
        advantage_computer=AdvantageComputer(),
        loss_computer=loss_fn,
        optimizer=optimizer,
    )
    metrics = trainer.train_step("def fib(n):")
    assert "loss" in metrics
    assert "avg_reward" in metrics
    assert "tree_nodes" in metrics
    assert "tree_leaves" in metrics
    assert metrics["loss"] == metrics["loss"]  # not NaN
    assert metrics["tree_leaves"] >= 1
