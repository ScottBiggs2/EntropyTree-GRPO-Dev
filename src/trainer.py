"""Entropy-MCTS GRPO trainer (Phase 7). Baseline GRPO trainer (Phase 8.1)."""

from typing import Dict, List

import torch

from src.config import MCTSConfig
from src.tree_builder import EntropyGuidedTreeBuilder
from src.tree_node import MCTSNode
from src.rewards import RewardFunction
from src.advantages import AdvantageComputer
from src.loss import WeightedGRPOLoss
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter


class BaselineGRPOTrainer:
    """Trajectory-level GRPO: K completions per prompt, reward - mean advantage, no tree (Phase 8.1)."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig,
        reward_fn: RewardFunction,
        loss_computer: WeightedGRPOLoss,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.entropy_computer = EntropyComputer()
        self.tree_builder = EntropyGuidedTreeBuilder(
            model, tokenizer, config, self.entropy_computer
        )

    def train_step(self, prompt: str) -> Dict[str, float]:
        """Generate K completions, compute rewards and trajectory log probs, GRPO loss, step."""
        K = self.config.num_baseline_samples
        self.model.eval()
        trajectories: List[tuple] = []
        for _ in range(K):
            completion, trans = self.tree_builder.generate_one_trajectory(prompt)
            trajectories.append((completion, trans))
        rewards = [self.reward_fn(c, prompt) for c, _ in trajectories]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        advantages = [r - mean_reward for r in rewards]

        self.model.train()
        log_probs = []
        for _, trans in trajectories:
            lp = self.loss_computer.trajectory_log_prob(self.model, trans)
            log_probs.append(lp)
        if not log_probs:
            return {"loss": 0.0, "avg_reward": 0.0, "max_reward": 0.0}
        adv_tensor = torch.tensor(
            advantages, dtype=log_probs[0].dtype, device=log_probs[0].device
        )
        log_prob_stack = torch.stack(log_probs)
        loss = -(adv_tensor * log_prob_stack).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "avg_reward": mean_reward,
            "max_reward": max(rewards) if rewards else 0.0,
            "tree_nodes": 0,
            "tree_leaves": K,
            "avg_entropy": 0.0,
        }

    def train_epoch(self, prompts: List[str]) -> Dict[str, float]:
        """Aggregate metrics over one epoch."""
        keys = ["loss", "avg_reward", "max_reward", "tree_nodes", "tree_leaves", "avg_entropy"]
        agg = {k: [] for k in keys}
        for prompt in prompts:
            m = self.train_step(prompt)
            for k in keys:
                if k in m:
                    agg[k].append(m[k])
        return {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}


class EntropyMCTSTrainer:
    """Single training step: build tree, rewards, advantages, weighted GRPO loss, step."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig,
        reward_fn: RewardFunction,
        advantage_computer: AdvantageComputer,
        loss_computer: WeightedGRPOLoss,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.advantage_computer = advantage_computer
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.entropy_computer = EntropyComputer()
        self.tree_builder = EntropyGuidedTreeBuilder(
            model, tokenizer, config, self.entropy_computer
        )
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    def train_step(self, prompt: str) -> Dict[str, float]:
        """One step: build tree, rewards, advantages, loss, backward, step."""
        self.model.eval()
        root, leaves = self.tree_builder.build_tree(prompt)
        if not leaves:
            return {"loss": 0.0, "avg_reward": 0.0, "tree_nodes": 0, "tree_leaves": 0}

        completions = [
            self.tokenizer.decode(
                leaf.state[root.prompt_len : root.prompt_len + self.config.max_new_tokens].tolist(),
                skip_special_tokens=True,
            )
            for leaf in leaves
        ]
        rewards = [self.reward_fn(c, prompt) for c in completions]
        self.advantage_computer.compute_advantages(
            root, leaves, rewards, mode="branchgrpo"
        )

        self.model.train()
        loss, loss_metrics = self.loss_computer.compute_loss(
            self.model, root, leaves, prompt, self.vocab_size
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        def count_nodes(n: MCTSNode) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        return {
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "tree_nodes": count_nodes(root),
            "tree_leaves": len(leaves),
            "avg_entropy": avg_entropy,
            **loss_metrics,
        }

    def train_epoch(self, prompts: List[str]) -> Dict[str, float]:
        """Aggregate metrics over one epoch of prompts."""
        keys = ["loss", "avg_reward", "max_reward", "tree_nodes", "tree_leaves", "avg_entropy"]
        agg = {k: [] for k in keys}
        for prompt in prompts:
            m = self.train_step(prompt)
            for k in keys:
                if k in m:
                    agg[k].append(m[k])
        return {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}
