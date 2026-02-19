"""Phase 1: MCTSConfig and utils (no model required)."""

import torch
import pytest

from src.config import MCTSConfig
from src.utils import get_device, create_masked_response, add_gumbel_noise, get_num_transfer_tokens


def test_mcts_config_defaults():
    config = MCTSConfig()
    assert config.max_tree_nodes == 15
    assert config.branch_width == 3
    assert config.steps_per_expansion == 32
    assert config.temperature == 0.8
    assert config.remasking == "low_confidence"
    assert config.alpha_time == 1.0
    assert config.alpha_entropy == 0.5
    assert config.total_denoising_steps == 256
    assert config.max_new_tokens == 256
    assert config.block_size == 64
    assert config.learning_rate == 1e-5
    assert config.model_name_or_path == "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
    assert config.device is not None


def test_get_device():
    dev = get_device()
    assert dev in ("mps", "cpu")


def test_create_masked_response_no_tokenizer_skip():
    """Test create_masked_response with a dummy tokenizer-like object (no HF download)."""
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        mask_token_id = 2

    prompt_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)  # 3 tokens
    tokenizer = FakeTokenizer()
    input_ids, attention_mask, prompt_len = create_masked_response(
        tokenizer, prompt_ids, max_new_tokens=5, device="cpu"
    )
    assert prompt_len == 3
    assert input_ids.shape == (1, 8)  # 3 + 5
    assert (input_ids[0, :3] == torch.tensor([10, 11, 12])).all()
    assert (input_ids[0, 3:8] == 2).all()  # mask_token_id
    assert attention_mask[0, :8].sum() == 8


def test_add_gumbel_noise_temperature_zero():
    logits = torch.randn(2, 3, 4)
    out = add_gumbel_noise(logits, temperature=0.0)
    assert torch.allclose(out, logits)


def test_add_gumbel_noise_temperature_positive():
    logits = torch.randn(2, 3, 4)
    out = add_gumbel_noise(logits, temperature=0.8)
    assert out.shape == logits.shape
    assert not torch.allclose(out.float(), logits.float())


def test_get_num_transfer_tokens():
    # 5 masked positions, 3 steps -> 2, 2, 1 or similar
    mask_index = torch.ones(1, 10, dtype=torch.bool)
    mask_index[0, 5:] = 0  # 5 masks
    out = get_num_transfer_tokens(mask_index, 3, mask_index.device)
    assert out.shape == (1, 3)
    assert out.sum().item() == 5
