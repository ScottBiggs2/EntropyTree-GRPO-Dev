"""Minimal Dream trainer wiring adapter, tree builder, and loss.

This module is designed to remain lightweight on the local laptop:
tests should use tiny mock models; real Dream weights are intended
to be loaded only on a cloud GPU.
"""

from typing import Callable, Dict, List, Optional

import torch

from dream.src.config import MCTSConfig
from dream.src.tree_builder import EntropyGuidedTreeBuilder
from dream.src.tree_node import MCTSNode
from dream.src.advantages import AdvantageComputer
from dream.src.loss import WeightedGRPOLoss
from dream.src.entropy import EntropyComputer
from dream.src.time_weight import TimeWeighter
from dream.src.model_adapter import ModelAdapter


RewardFn = Callable[[str, str], float]


class EntropyMCTSTrainer:
    """Single-step trainer: build tree, compute loss, apply one optimizer step.

    This mirrors the parent project's EntropyMCTSTrainer but is wired
    against the Dream-specific adapter / loss stack.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig,
        reward_fn: RewardFn,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        advantage_computer: Optional[AdvantageComputer] = None,
        loss_computer: Optional[WeightedGRPOLoss] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.adapter = ModelAdapter(model, tokenizer, model_type=config.model_type)
        self.entropy_computer = EntropyComputer()
        self.tree_builder = EntropyGuidedTreeBuilder(
            self.adapter, tokenizer, config, self.entropy_computer
        )
        self.time_weighter = TimeWeighter(
            config.total_denoising_steps, norm_mode=config.time_weight_norm
        )
        self.advantage_computer = advantage_computer or AdvantageComputer()
        self.loss_computer = loss_computer or WeightedGRPOLoss(
            config,
            self.entropy_computer,
            self.time_weighter,
            tokenizer.mask_token_id,
            self.adapter,
        )
        # Use adapter's vocab_size for correct entropy normalization.
        self.vocab_size = self.adapter.vocab_size
        self._diag_count = 0

    def train_step(self, prompt: str) -> Dict[str, float]:
        """One training step: build tree, compute loss, update model."""
        self.model.eval()
        root, leaves = self.tree_builder.build_tree(prompt)
        if not leaves:
            return {
                "loss": 0.0,
                "avg_reward": 0.0,
                "max_reward": 0.0,
                "tree_nodes": 1,
                "tree_leaves": 0,
            }

        completions = [
            self.tokenizer.decode(
                leaf.state[root.prompt_len : root.prompt_len + self.config.max_new_tokens].tolist(),
                skip_special_tokens=True,
            )
            for leaf in leaves
        ]
        rewards = [self.reward_fn(c, prompt) for c in completions]

        if self._diag_count < 3:
            self._diag_count += 1
            best_idx = rewards.index(max(rewards)) if rewards else 0
            sample = completions[best_idx] if completions else ""
            preview = sample[:160].replace("\n", "\\n")
            print(
                f"[dream-mcts diag] prompt={prompt[:40]!r} "
                f"rewards={[round(r, 3) for r in rewards]} "
                f"best_completion={preview!r}"
            )

        self.advantage_computer.compute_advantages(
            root,
            leaves,
            rewards,
            mode="branchgrpo",
            advantage_clip=getattr(self.config, "advantage_clip", 2.0),
        )

        self.model.train()
        per_trans = getattr(self.config, "loss_backward_per_transition", True)
        self.optimizer.zero_grad()
        loss, loss_metrics = self.loss_computer.compute_loss(
            self.model,
            root,
            leaves,
            prompt,
            self.vocab_size,
            backward_per_transition=per_trans,
        )
        if not per_trans:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        def count_nodes(n: MCTSNode) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        lr = self.optimizer.param_groups[0]["lr"]
        metrics: Dict[str, float] = {
            "loss": float(loss.item()),
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "tree_nodes": float(count_nodes(root)),
            "tree_leaves": float(len(leaves)),
            "avg_entropy": avg_entropy,
            "lr": lr,
        }
        # Merge in loss-internal diagnostics (n_transitions, weights, etc.).
        metrics.update({k: float(v) for k, v in loss_metrics.items()})
        return metrics

