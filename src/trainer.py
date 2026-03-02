"""Entropy-MCTS GRPO trainer (Phase 7). Baseline GRPO trainer (Phase 8.1)."""

from typing import Dict, List, Optional

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
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.entropy_computer = EntropyComputer()
        self.tree_builder = EntropyGuidedTreeBuilder(
            model, tokenizer, config, self.entropy_computer
        )
        self._diag_count = 0

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

        # Z-score normalize advantages for stable GRPO gradients
        advantages = [r - mean_reward for r in rewards]
        adv_std = (sum(a * a for a in advantages) / max(len(advantages), 1)) ** 0.5
        if adv_std > 1e-8:
            advantages = [a / adv_std for a in advantages]

        if self._diag_count < 3:
            self._diag_count += 1
            best_idx = rewards.index(max(rewards)) if rewards else 0
            sample = trajectories[best_idx][0] if trajectories else ""
            preview = sample[:200].replace("\n", "\\n")
            print(f"[diag baseline] prompt={prompt[:40]!r} rewards={[round(r,3) for r in rewards]} best_completion={preview!r}")

        self.model.train()
        log_probs = []
        n_tokens_list = []
        for _, trans in trajectories:
            lp, n_tok = self.loss_computer.trajectory_log_prob_with_count(self.model, trans)
            log_probs.append(lp)
            n_tokens_list.append(n_tok)
        if not log_probs:
            return {"loss": 0.0, "avg_reward": 0.0, "max_reward": 0.0}

        # Per-token normalization: divide each trajectory's summed log-prob by
        # the number of tokens it covers, so loss scale doesn't depend on sequence length.
        norm_log_probs = []
        for lp, nt in zip(log_probs, n_tokens_list):
            norm_log_probs.append(lp / max(nt, 1))

        adv_tensor = torch.tensor(
            advantages, dtype=norm_log_probs[0].dtype, device=norm_log_probs[0].device
        )
        log_prob_stack = torch.stack(norm_log_probs)
        loss = -(adv_tensor * log_prob_stack).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss": loss.item(),
            "avg_reward": mean_reward,
            "max_reward": max(rewards) if rewards else 0.0,
            "tree_nodes": 0,
            "tree_leaves": K,
            "avg_entropy": 0.0,
            "lr": lr,
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
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.advantage_computer = advantage_computer
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.entropy_computer = EntropyComputer()
        self.tree_builder = EntropyGuidedTreeBuilder(
            model, tokenizer, config, self.entropy_computer
        )
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        self._diag_count = 0

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

        if self._diag_count < 3:
            self._diag_count += 1
            best_idx = rewards.index(max(rewards)) if rewards else 0
            sample = completions[best_idx] if completions else ""
            preview = sample[:200].replace("\n", "\\n")
            print(f"[diag mcts] prompt={prompt[:40]!r} rewards={[round(r,3) for r in rewards]} best_completion={preview!r}")

        self.advantage_computer.compute_advantages(
            root, leaves, rewards, mode="branchgrpo",
            advantage_clip=getattr(self.config, "advantage_clip", 2.0),
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
        if self.scheduler is not None:
            self.scheduler.step()

        def count_nodes(n: MCTSNode) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        n_transitions = loss_metrics.get("n_transitions")
        if n_transitions is not None:
            expected_transitions = count_nodes(root) - 1
            assert (
                n_transitions == expected_transitions
            ), f"n_transitions={n_transitions} but tree_nodes-1={expected_transitions}"

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "tree_nodes": count_nodes(root),
            "tree_leaves": len(leaves),
            "avg_entropy": avg_entropy,
            "lr": lr,
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
