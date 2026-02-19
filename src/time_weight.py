"""Time-based loss weighting (TempFlow-GRPO style). Phase 2."""

import math
from typing import Optional

import torch


class TimeWeighter:
    """Precomputed normalized weights w(t) = (1 - t/T)^2 / Z for step_index 0..T-1."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self._precompute()

    def _precompute(self) -> None:
        T = self.total_steps
        timesteps = torch.arange(T, dtype=torch.float32)
        weights = (1.0 - timesteps / T) ** 2
        weights = weights / (weights.sum() + 1e-10)
        self.weights = weights

    def get_weight(self, step_index: int) -> float:
        """Weight for this step (0 = start, T-1 = last step before done)."""
        if step_index >= self.total_steps or step_index < 0:
            return 0.0
        return self.weights[step_index].item()
