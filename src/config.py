"""Central configuration for entropy-guided MCTS-GRPO (Phase 1)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSConfig:
    """Hyperparameters for entropy-guided MCTS and GRPO training."""

    # Tree construction (DeepSearch style) — D-009 defaults
    max_tree_nodes: int = 15
    branch_width: int = 3
    steps_per_expansion: int = 32

    # Sampling — D-006: low_confidence
    temperature: float = 0.8
    remasking: str = "low_confidence"

    # Loss weighting — D-010
    alpha_time: float = 1.0
    alpha_entropy: float = 0.5

    # Model / generation
    total_denoising_steps: int = 256
    max_new_tokens: int = 256
    block_size: int = 64

    # Training
    batch_size: int = 1
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # Model path and device (D-001: MPS/CPU, no CUDA)
    model_name_or_path: str = "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
    device: Optional[str] = None  # None = auto-detect via get_device()

    # Experiment (Phase 8)
    num_epochs: int = 2
    num_baseline_samples: int = 4  # K completions per prompt for baseline GRPO
    run_name: Optional[str] = None  # WandB / checkpoint run name
    checkpoint_dir: str = "checkpoints"  # root dir; method subdirs added (baseline_grpo, entropy_mcts_grpo)
    wandb_project: str = "entropy-tree-grpo"
    save_every_steps: Optional[int] = None  # None = save only at end of run

    def __post_init__(self) -> None:
        if self.device is None:
            from src.utils import get_device
            self.device = get_device()
