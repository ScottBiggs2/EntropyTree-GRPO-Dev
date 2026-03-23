import torch
import torch.nn as nn

from dream.src.config import MCTSConfig
from dream.src.trainer import BaselineGRPOTrainer, EntropyMCTSTrainer


class _TinyMockModel(nn.Module):
    """Very small mock model: linear head over dummy features."""

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(4, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bsz, seqlen = input_ids.shape
        dummy = torch.zeros(bsz, seqlen, 4, device=input_ids.device)
        logits = self.proj(dummy)

        class _Out:
            def __init__(self, logits):
                self.logits = logits

        return _Out(logits)


class _TinyMockTokenizer:
    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size
        self.mask_token_id = 2
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        # Deterministic but trivial mapping: length-limited.
        ids = [3 + (ord(c) % (self.vocab_size - 4)) for c in text[:8]]
        if add_special_tokens:
            ids = [1] + ids + [1]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(t) for t in token_ids)


def _dummy_reward(completion: str, prompt: str) -> float:
    # Simple heuristic: longer completions score higher.
    return float(len(completion))


def test_entropy_mcts_trainer_single_step_runs():
    """Smoke test: one train_step runs end to end with mock model."""
    cfg = MCTSConfig(
        model_type="mdlm",
        max_tree_nodes=5,
        branch_width=2,
        steps_per_expansion=4,
        max_new_tokens=16,
        total_denoising_steps=32,
    )
    # Put model on config device so tree builder tensors match (works on CPU, MPS, or CUDA).
    model = _TinyMockModel(vocab_size=32).to(cfg.device)
    tok = _TinyMockTokenizer(vocab_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = EntropyMCTSTrainer(
        model=model,
        tokenizer=tok,
        config=cfg,
        reward_fn=_dummy_reward,
        optimizer=optimizer,
    )
    metrics = trainer.train_step("test prompt")
    assert "loss" in metrics
    assert metrics["tree_nodes"] >= 1.0
    assert metrics["tree_leaves"] >= 1.0
    assert metrics.get("n_loss_forwards", 0) <= metrics.get("n_transitions", 0)


def test_baseline_grpo_trainer_single_step_runs():
    """Flat GRPO path: K trajectories, no MCTS (mock model)."""
    cfg = MCTSConfig(
        model_type="mdlm",
        max_tree_nodes=5,
        branch_width=2,
        steps_per_expansion=4,
        max_new_tokens=16,
        total_denoising_steps=32,
        num_baseline_samples=2,
    )
    model = _TinyMockModel(vocab_size=32).to(cfg.device)
    tok = _TinyMockTokenizer(vocab_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = BaselineGRPOTrainer(
        model=model,
        tokenizer=tok,
        config=cfg,
        reward_fn=_dummy_reward,
        optimizer=optimizer,
    )
    metrics = trainer.train_step("test prompt")
    assert "loss" in metrics
    assert metrics["tree_nodes"] == 0.0
    assert metrics["tree_leaves"] == float(cfg.num_baseline_samples)
    assert metrics.get("n_transitions", 0) > 0


def test_entropy_mcts_trainer_eval_step_runs():
    """eval_step builds tree and scores without optimizer / loss graph."""
    cfg = MCTSConfig(
        model_type="mdlm",
        max_tree_nodes=5,
        branch_width=2,
        steps_per_expansion=4,
        max_new_tokens=16,
        total_denoising_steps=32,
    )
    model = _TinyMockModel(vocab_size=32).to(cfg.device)
    tok = _TinyMockTokenizer(vocab_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = EntropyMCTSTrainer(
        model=model,
        tokenizer=tok,
        config=cfg,
        reward_fn=_dummy_reward,
        optimizer=optimizer,
    )
    metrics = trainer.eval_step("test prompt")
    assert metrics["tree_nodes"] >= 1.0
    assert metrics["tree_leaves"] >= 1.0
    assert metrics["n_transitions"] == 0.0

