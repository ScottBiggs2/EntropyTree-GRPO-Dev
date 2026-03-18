"""Entropy computation for Dream/MDLM (corrected normalization)."""

from typing import Optional
import math

import torch
import torch.nn.functional as F


class EntropyComputer:
    """Compute per-token Shannon entropy and aggregate / weight helpers."""

    @staticmethod
    def compute_token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy per token from pre-computed logits.

        Args:
            logits: [batch, seq_len, vocab_size]

        Returns:
            entropy: [batch, seq_len]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    @torch.no_grad()
    def compute_token_entropy(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass, then Shannon entropy per token (MDLM-style helper).

        For Dream, callers should prefer compute_token_entropy_from_logits
        with logits obtained via the ModelAdapter.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        return EntropyComputer.compute_token_entropy_from_logits(logits)

    @staticmethod
    def aggregate_entropy(
        token_entropy: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        method: str = "mean",
    ) -> float:
        """Aggregate entropy across positions.

        If mask_positions is provided, aggregates only over those positions
        (e.g., masked response tokens).
        """
        if mask_positions is not None:
            # mask_positions expected shape: [batch, seq_len]
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
    def compute_entropy_weight(
        measured_masked_mean: float,
        vocab_size: int,
        masking_ratio: float = 0.0,
        mode: str = "analytic",
        stage_baseline: float = 0.0,
        eps: float = 1e-6,
    ) -> float:
        """Compute entropy weight with a consistent normalization convention.

        Args:
            measured_masked_mean: mean entropy over masked positions only.
            vocab_size: tokenizer size V.
            masking_ratio: fraction of response tokens still masked (for diagnostics).
            mode:
                - "analytic": w_ent = H_masked_mean / log(V).
                - "stage_aware": w_ent = H_masked_mean / E[H_masked_mean | masking_ratio].
            stage_baseline: empirical baseline E[H_masked_mean | masking_ratio] (if available).
        """
        if mode == "analytic":
            log_v = math.log(max(vocab_size, 1))
            if log_v < eps:
                return 0.0
            return measured_masked_mean / log_v

        if mode == "stage_aware":
            if stage_baseline < eps:
                return 0.0
            return measured_masked_mean / stage_baseline

        raise ValueError(f"Unknown entropy norm mode: {mode}")

    @staticmethod
    def expected_entropy(masking_ratio: float, vocab_size: int) -> float:
        """Sequence-averaged upper-bound entropy (legacy helper).

        This corresponds to masking_ratio * log(V) and is kept for diagnostics;
        it is no longer used as the primary normalization denominator.
        """
        if masking_ratio <= 0:
            return 0.0
        return masking_ratio * math.log(max(vocab_size, 1))

