"""MCTS tree node and transition dataclasses (Phase 3)."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class MCTSNode:
    """Node in the MCTS tree (partial denoising state)."""

    state: torch.Tensor  # [seq_len] token IDs
    attention_mask: torch.Tensor  # [seq_len] 1 = valid
    prompt_len: int
    step_index: int  # number of denoising steps taken to reach this node
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    mask_id: Optional[int] = None  # tokenizer.mask_token_id; set by tree builder

    entropy: Optional[float] = None
    token_entropy: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    fused_reward: Optional[float] = None
    advantage: Optional[float] = None
    sampling_prob: float = 1.0
    depth: int = 0
    is_completed: bool = False

    def _response_len(self) -> int:
        """Length of response region (from prompt_len to last valid)."""
        valid = int(self.attention_mask.sum().item())
        return max(0, valid - self.prompt_len)

    def num_masked_tokens(self) -> int:
        """Count mask tokens in the response region only. Requires mask_id set."""
        if self.mask_id is None:
            return 0
        resp_len = self._response_len()
        if resp_len == 0:
            return 0
        response_slice = self.state[self.prompt_len : self.prompt_len + resp_len]
        return (response_slice == self.mask_id).sum().item()

    def masking_ratio(self) -> float:
        """Fraction of response tokens that are still masked."""
        resp_len = self._response_len()
        if resp_len == 0:
            return 0.0
        n = self.num_masked_tokens()
        return n / resp_len

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class TreeTransition:
    """One parent -> child transition for loss computation."""

    parent_state: torch.Tensor
    child_state: torch.Tensor
    parent_attention_mask: torch.Tensor
    child_attention_mask: torch.Tensor
    step_index: int
    advantage: float
    entropy: float
    time_weight: float
    entropy_weight: float
