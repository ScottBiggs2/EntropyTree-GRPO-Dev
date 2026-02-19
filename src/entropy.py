"""Entropy computation for MDLM (exact Shannon entropy from logits). Phase 2."""

from typing import Optional, Union

import torch
import torch.nn.functional as F
import math


class EntropyComputer:
    """Compute per-token Shannon entropy and aggregate / weight helpers."""

    @staticmethod
    @torch.no_grad()
    def compute_token_entropy(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass, then Shannon entropy per token.
        input_ids: [batch, seq_len], attention_mask: [batch, seq_len] (1 = valid).
        Returns: [batch, seq_len] per-token entropy.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
        return entropy

    @staticmethod
    def aggregate_entropy(
        token_entropy: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        method: str = "mean",
    ) -> float:
        """
        Aggregate over positions. If mask_positions given, aggregate only over those (masked tokens).
        method: "mean", "max", "sum".
        """
        if mask_positions is not None:
            token_entropy = token_entropy.masked_fill(~mask_positions, 0.0)
            count = mask_positions.sum().clamp(min=1).item()
            if method == "mean":
                return token_entropy.sum().item() / count
            if method == "sum":
                return token_entropy.sum().item()
            if method == "max":
                if mask_positions.any():
                    return token_entropy.masked_fill(~mask_positions, -1.0).max().item()
                return 0.0
        if method == "mean":
            return token_entropy.mean().item()
        if method == "max":
            return token_entropy.max().item()
        if method == "sum":
            return token_entropy.sum().item()
        raise ValueError(f"Unknown aggregation method: {method}")

    @staticmethod
    def expected_entropy(masking_ratio: float, vocab_size: int) -> float:
        """Theoretical expected entropy H_bar = masking_ratio * log(V)."""
        if masking_ratio <= 0:
            return 0.0
        return masking_ratio * math.log(vocab_size)

    @staticmethod
    def compute_entropy_weight(
        measured: float,
        masking_ratio: float,
        vocab_size: int,
        eps: float = 1e-6,
    ) -> float:
        """w_ent = measured / expected. Returns 0 when expected ~ 0."""
        expected = EntropyComputer.expected_entropy(masking_ratio, vocab_size)
        if expected < eps:
            return 0.0
        return measured / expected
