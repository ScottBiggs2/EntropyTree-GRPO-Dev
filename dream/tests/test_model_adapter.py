import torch
import torch.nn as nn

from dream.src.model_adapter import ModelAdapter


class _MockTokenizer:
    def __init__(self, mask_token_id: int, vocab_size: int):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1


class _MockMDLMModel(nn.Module):
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self._proj = nn.Linear(4, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bsz, seqlen = input_ids.shape
        dummy = torch.zeros(bsz, seqlen, 4, device=input_ids.device)
        logits = self._proj(dummy)

        class _Out:
            def __init__(self, logits):
                self.logits = logits

        return _Out(logits)


def test_mdlm_adapter_forward_shape():
    model = _MockMDLMModel(vocab_size=101)
    tok = _MockTokenizer(mask_token_id=42, vocab_size=101)
    adapter = ModelAdapter(model, tok, model_type="mdlm")
    ids = torch.randint(0, 100, (2, 16))
    attn = torch.ones_like(ids)
    logits = adapter.forward_logits(ids, attn)
    assert logits.shape == (2, 16, 101)


def test_mdlm_adapter_sample_and_confidence():
    model = _MockMDLMModel(vocab_size=50)
    tok = _MockTokenizer(mask_token_id=7, vocab_size=50)
    adapter = ModelAdapter(model, tok, model_type="mdlm")
    logits = torch.randn(16, 50)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[4:10] = True
    x0, conf = adapter.sample_and_confidence(
        logits, mask_positions=mask, temperature=0.8, top_p=0.95
    )
    assert x0.shape == (16,)
    assert conf.shape == (16,)
    assert (conf[mask] >= 0).all()
    assert (conf[~mask] < -1e8).all()


def test_transfer_count_mdlm_uniform():
    model = _MockMDLMModel(vocab_size=50)
    tok = _MockTokenizer(mask_token_id=7, vocab_size=50)
    adapter = ModelAdapter(model, tok, model_type="mdlm")
    k = adapter.transfer_count(n_masked=100, step=0, total_steps=10)
    assert k == 10


class _MockDreamModel(nn.Module):
    """Minimal module so ModelAdapter(..., model_type='dream') has config + device."""

    def __init__(self, vocab_size: int = 1024):
        super().__init__()
        self.config = type("Cfg", (), {"vocab_size": vocab_size})()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        raise RuntimeError("not used in sampling tests")


def test_dream_sample_bf16_top_p_no_crash():
    """Regression: bf16 logits + top_p used to yield all-zero probs / Categorical error."""
    model = _MockDreamModel(vocab_size=4096)
    tok = _MockTokenizer(mask_token_id=99, vocab_size=4096)
    adapter = ModelAdapter(model, tok, model_type="dream")
    logits = torch.randn(128, 4096, dtype=torch.bfloat16)
    mask = torch.ones(128, dtype=torch.bool)
    x0, conf = adapter.sample_and_confidence(
        logits, mask_positions=mask, temperature=1.0, top_p=0.95
    )
    assert x0.shape == (128,)
    assert conf.shape == (128,)
    assert torch.isfinite(conf[mask]).all()
    assert (conf[mask] >= 0).all()

