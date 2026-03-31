"""Minimal Dream trainer wiring adapter, tree builder, and loss.

This module is designed to remain lightweight on the local laptop:
tests should use tiny mock models; real Dream weights are intended
to be loaded only on a cloud GPU.
"""

from typing import Callable, Dict, Iterable, List, Optional

import torch

from dream.src.config import MCTSConfig
from dream.src.tree_builder import EntropyGuidedTreeBuilder
from dream.src.tree_node import MCTSNode
from dream.src.advantages import AdvantageComputer
from dream.src.loss import WeightedGRPOLoss
from dream.src.entropy import EntropyComputer
from dream.src.time_weight import TimeWeighter
from dream.src.model_adapter import ModelAdapter
from dream.src.observability import tree_diversity_metrics


RewardFn = Callable[[str, str], float]


def _reward_stats(rewards: List[float]) -> Dict[str, float]:
    if not rewards:
        return {
            "reward_std": 0.0,
            "reward_range": 0.0,
            "unique_rewards_frac": 0.0,
        }
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)
    std = var**0.5
    n_unique = len(set(round(r, 6) for r in rewards))
    return {
        "reward_std": float(std),
        "reward_range": float(max(rewards) - min(rewards)),
        "unique_rewards_frac": float(n_unique) / float(len(rewards)),
    }


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        total_sq += float(g.float().pow(2).sum().item())
    return float(total_sq**0.5)


class BaselineGRPOTrainer:
    """Trajectory-level GRPO without MCTS: K independent completions per prompt.

    Same reward and per-token normalized policy gradient as the parent project's
    ``BaselineGRPOTrainer``, but wired through Dream's ``ModelAdapter`` and
    ``generate_one_trajectory`` (iterative unmasking). Use with the **same**
    ``MCTSConfig.use_lora`` / ``lora_r`` / ``lora_alpha`` as tree runs to
    isolate the effect of entropy-tree search vs flat sampling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig,
        reward_fn: RewardFn,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
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
        self.loss_computer = loss_computer or WeightedGRPOLoss(
            config,
            self.entropy_computer,
            self.time_weighter,
            tokenizer.mask_token_id,
            self.adapter,
        )
        self._diag_count = 0

    def train_step(self, prompt: str) -> Dict[str, float]:
        """Generate K completions, GRPO loss, one optimizer step."""
        K = self.config.num_baseline_samples
        self.model.eval()
        trajectories = []
        for _ in range(K):
            completion, trans = self.tree_builder.generate_one_trajectory(
                prompt,
                temperature_override=self.config.train_sampling_temperature,
            )
            trajectories.append((completion, trans))
        rewards = [self.reward_fn(c, prompt) for c, _ in trajectories]
        exec_frac_mean = 0.0
        shaping_bonus_mean = 0.0
        if hasattr(self.reward_fn, "score_components"):
            comps = [self.reward_fn.score_components(c, prompt) for c, _ in trajectories]  # type: ignore[attr-defined]
            if comps:
                exec_frac_mean = float(sum(d.get("exec_frac", 0.0) for d in comps) / len(comps))
                shaping_bonus_mean = float(
                    sum(d.get("shaping_bonus", 0.0) for d in comps) / len(comps)
                )
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        rstats = _reward_stats(rewards)

        advantages = [r - mean_reward for r in rewards]
        adv_std = (sum(a * a for a in advantages) / max(len(advantages), 1)) ** 0.5
        if adv_std > 1e-8:
            advantages = [a / adv_std for a in advantages]

        if self._diag_count < 3:
            self._diag_count += 1
            best_idx = rewards.index(max(rewards)) if rewards else 0
            sample = trajectories[best_idx][0] if trajectories else ""
            preview = sample[:200].replace("\n", "\\n")
            n_unique = len(set(round(r, 6) for r in rewards))
            print(
                f"[dream-baseline-grpo diag] prompt={prompt[:40]!r} "
                f"rewards={[round(r, 3) for r in rewards]} "
                f"unique_rewards={n_unique}/{K} adv_std={adv_std:.6f} "
                f"best_completion={preview!r}"
            )
            if adv_std < 1e-8:
                print(
                    "[dream-baseline-grpo WARN] all rewards identical — "
                    "advantages are zero, no gradient this step"
                )

        self.model.train()
        log_probs = []
        n_tokens_list = []
        n_trans = 0
        for _, trans in trajectories:
            lp, n_tok = self.loss_computer.trajectory_log_prob_with_count(
                self.model, trans
            )
            log_probs.append(lp)
            n_tokens_list.append(n_tok)
            n_trans += len(trans)

        if not log_probs:
            return {
                "loss": 0.0,
                "avg_reward": 0.0,
                "mean_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "tree_nodes": 0.0,
                "tree_leaves": float(K),
                "avg_entropy": 0.0,
                "n_transitions": 0.0,
                "n_loss_forwards": 0.0,
            }

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
        grad_norm = _grad_norm(p for p in self.model.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]["lr"]
        min_r = min(rewards) if rewards else 0.0
        return {
            "loss": float(loss.item()),
            "avg_reward": mean_reward,
            "mean_reward": mean_reward,
            "min_reward": min_r,
            "max_reward": max(rewards) if rewards else 0.0,
            **rstats,
            "exec_frac_mean": float(exec_frac_mean),
            "shaping_bonus_mean": float(shaping_bonus_mean),
            "adv_std_raw": float(adv_std),
            "mean_norm_logp": float(log_prob_stack.detach().mean().item()),
            "std_norm_logp": float(log_prob_stack.detach().std(unbiased=False).item()),
            "grad_norm": float(grad_norm),
            "tree_nodes": 0.0,
            "tree_leaves": float(K),
            "avg_entropy": 0.0,
            "lr": lr,
            "n_transitions": float(n_trans),
            "n_loss_forwards": float(sum(len(t) for _, t in trajectories)),
        }


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

    def eval_step(self, prompt: str) -> Dict[str, float]:
        """Build tree and score completions; no backward / optimizer (initial baseline)."""
        self.model.eval()
        with torch.no_grad():
            root, leaves = self.tree_builder.build_tree(prompt)
        if not leaves:
            return {
                "loss": 0.0,
                "avg_reward": 0.0,
                "mean_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "tree_nodes": 1.0,
                "tree_leaves": 0.0,
                "avg_entropy": 0.0,
                "n_transitions": 0.0,
                "n_loss_forwards": 0.0,
            }

        completions = [
            self.tokenizer.decode(
                leaf.state[
                    root.prompt_len : root.prompt_len + self.config.max_new_tokens
                ].tolist(),
                skip_special_tokens=True,
            )
            for leaf in leaves
        ]
        rewards = [self.reward_fn(c, prompt) for c in completions]

        def count_nodes(n: MCTSNode) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        min_reward = min(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        return {
            "loss": 0.0,
            "avg_reward": avg_reward,
            "mean_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "tree_nodes": float(count_nodes(root)),
            "tree_leaves": float(len(leaves)),
            "avg_entropy": avg_entropy,
            "n_transitions": 0.0,
            "n_loss_forwards": 0.0,
            **tree_diversity_metrics(root, leaves),
        }

    def train_step(self, prompt: str) -> Dict[str, float]:
        """One training step: build tree, compute loss, update model."""
        self.model.eval()
        root, leaves = self.tree_builder.build_tree(prompt)
        if not leaves:
            return {
                "loss": 0.0,
                "avg_reward": 0.0,
                "mean_reward": 0.0,
                "min_reward": 0.0,
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
        rstats = _reward_stats(rewards)
        exec_frac_mean = 0.0
        shaping_bonus_mean = 0.0
        if hasattr(self.reward_fn, "score_components"):
            comps = [self.reward_fn.score_components(c, prompt) for c in completions]  # type: ignore[attr-defined]
            if comps:
                exec_frac_mean = float(sum(d.get("exec_frac", 0.0) for d in comps) / len(comps))
                shaping_bonus_mean = float(
                    sum(d.get("shaping_bonus", 0.0) for d in comps) / len(comps)
                )

        if self._diag_count < 3:
            self._diag_count += 1
            best_idx = rewards.index(max(rewards)) if rewards else 0
            sample = completions[best_idx] if completions else ""
            preview = sample[:160].replace("\n", "\\n")
            n_unique = len(set(round(r, 6) for r in rewards))
            print(
                f"[dream-mcts diag] prompt={prompt[:40]!r} "
                f"rewards={[round(r, 3) for r in rewards]} "
                f"unique_rewards={n_unique}/{len(rewards)} "
                f"best_completion={preview!r}"
            )
            if n_unique < 2:
                print(
                    "[dream-mcts WARN] all leaf rewards identical — "
                    "BranchGRPO advantages will be zero at each depth"
                )

        self.advantage_computer.compute_advantages(
            root,
            leaves,
            rewards,
            mode="branchgrpo",
            advantage_clip=getattr(self.config, "advantage_clip", 2.0),
        )

        if getattr(self.config, "cuda_empty_cache_after_tree", True):
            if torch.cuda.is_available() and str(self.config.device).startswith(
                "cuda"
            ):
                torch.cuda.empty_cache()

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
        grad_norm = _grad_norm(p for p in self.model.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        def count_nodes(n: MCTSNode) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        min_reward = min(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        entropies = [n.entropy for n in [root] + leaves if n.entropy is not None]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        lr = self.optimizer.param_groups[0]["lr"]
        metrics: Dict[str, float] = {
            "loss": float(loss.item()),
            "avg_reward": avg_reward,
            "mean_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            **rstats,
            "exec_frac_mean": float(exec_frac_mean),
            "shaping_bonus_mean": float(shaping_bonus_mean),
            "grad_norm": float(grad_norm),
            "tree_nodes": float(count_nodes(root)),
            "tree_leaves": float(len(leaves)),
            "avg_entropy": avg_entropy,
            "lr": lr,
        }
        # Merge in loss-internal diagnostics (n_transitions, weights, etc.).
        metrics.update({k: float(v) for k, v in loss_metrics.items()})
        metrics.update(tree_diversity_metrics(root, leaves))
        return metrics

