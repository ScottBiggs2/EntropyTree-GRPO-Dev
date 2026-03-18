"""Time-based loss weighting with interval-aware support (TempFlow-style)."""

from typing import Literal

import torch


NormMode = Literal["sum_to_one", "mean_to_one"]


class TimeWeighter:
    """Precomputed time weights w(t) with configurable normalization.

    Base schedule: w_base(t) = (1 - t/T)^2 for t in [0, T-1].

    norm_mode:
        - "sum_to_one": sum_t w(t) = 1       (original behavior, each O(1/T))
        - "mean_to_one": mean_t w(t) = 1     (each O(1), comparable to entropy weights)
    """

    def __init__(self, total_steps: int, norm_mode: NormMode = "mean_to_one"):
        self.total_steps = total_steps
        self.norm_mode: NormMode = norm_mode
        self._precompute()

    def _precompute(self) -> None:
        T = max(self.total_steps, 1)
        timesteps = torch.arange(T, dtype=torch.float32)
        raw = (1.0 - timesteps / T) ** 2
        if self.norm_mode == "sum_to_one":
            self.weights = raw / (raw.sum() + 1e-10)
        elif self.norm_mode == "mean_to_one":
            self.weights = raw / (raw.mean() + 1e-10)
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    def get_weight(self, step_index: int) -> float:
        """Point weight for a single step index (backward-compatible API)."""
        if step_index < 0 or step_index >= self.total_steps:
            return 0.0
        return float(self.weights[step_index].item())

    def get_interval_weight(self, start: int, end: int) -> float:
        """Sum of weights over interval [start, end).

        This is the correct weighting for a macro-edge spanning multiple
        denoising steps, especially when edge lengths vary.
        """
        if end <= start:
            return 0.0
        start_clamped = max(0, start)
        end_clamped = min(end, self.total_steps)
        if start_clamped >= end_clamped:
            return 0.0
        return float(self.weights[start_clamped:end_clamped].sum().item())

