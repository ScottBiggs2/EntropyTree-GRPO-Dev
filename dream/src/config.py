"""Configuration for Dream/MDLM entropy-tree GRPO (Dream substack)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MCTSConfig:
    """Hyperparameters for entropy-guided MCTS and GRPO training (Dream stack).

    This mirrors the parent project's MCTSConfig but:
    - adds Dream-specific fields (model_type, top_p, alg, alg_temp)
    - adds adaptive stepping controls
    - adds explicit entropy/time normalization mode switches
    """

    # --- Model ---
    model_type: str = "dream"  # "dream" or "mdlm"
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    device: Optional[str] = None  # None = auto-detect via get_device()

    # --- Tree construction (DeepSearch style) ---
    max_tree_nodes: int = 15
    branch_width: int = 3
    # Used only when adaptive_stepping=False
    steps_per_expansion: int = 32

    # --- Adaptive stepping (entropy-threshold branching) ---
    adaptive_stepping: bool = False
    min_steps_per_expansion: int = 8
    max_steps_per_expansion: int = 48
    branch_threshold: float = 1.1  # threshold on H_masked_mean / log(V)

    # --- Sampling ---
    # MDLM-style temperature; Dream recommends low temp for code.
    temperature: float = 0.2
    top_p: float = 0.95
    remasking: str = "low_confidence"
    # Dream-specific generation params (for baseline GRPO / diffusion_generate)
    alg: str = "entropy"
    alg_temp: float = 0.0

    # --- Loss weighting (D-010) ---
    alpha_time: float = 1.0
    alpha_entropy: float = 0.5
    # Stability clamps (D-014)
    entropy_weight_min: float = 0.5
    entropy_weight_max: float = 2.0
    advantage_clip: float = 2.0

    # --- Entropy normalization convention (D-017) ---
    # "analytic":  H_masked_mean / log(V)
    # "stage_aware": H_masked_mean / E[H_masked_mean | masking_ratio]
    entropy_norm_mode: str = "analytic"

    # --- Time weighting normalization (D-016) ---
    # "sum_to_one": sum_t w_time(t) = 1         (original behavior, each O(1/T))
    # "mean_to_one": mean_t w_time(t) = 1       (each O(1), comparable to entropy weights)
    time_weight_norm: str = "mean_to_one"

    # --- Generation ---
    total_denoising_steps: int = 256
    max_new_tokens: int = 256

    # --- Training ---
    batch_size: int = 1
    learning_rate: float = 5e-6  # conservative for 7B
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.05
    gradient_checkpointing: bool = True
    # PEFT LoRA: only adapter weights are trainable — required for ~32GB + full
    # sequence backward on 7B (full fine-tune needs ~28GB+ just weights+grads).
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # If True, call backward() once per tree transition instead of on mean(loss).
    # Same gradients, much lower peak VRAM (important for 7B on ~24–32GB GPUs).
    loss_backward_per_transition: bool = True
    # After tree build (no_grad), release cached blocks to reduce fragmentation
    # before the first training forward (see DEVELOPMENT_PLAN Appendix D).
    cuda_empty_cache_after_tree: bool = True
    # One forward(parent) for all sibling edges; same gradients, fewer 7B passes.
    loss_group_by_parent: bool = True

    # --- Experiment metadata ---
    num_epochs: int = 2
    num_baseline_samples: int = 4
    run_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "entropy-tree-grpo-dream"
    save_every_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.device is None:
            # Local import to avoid circular dependency when packaging.
            from dream.src.utils import get_device  # type: ignore

            self.device = get_device()

