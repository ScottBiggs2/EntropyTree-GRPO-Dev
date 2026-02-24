"""Weighted GRPO loss (time + entropy weighting). Phase 6."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.config import MCTSConfig
from src.tree_node import MCTSNode, TreeTransition
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter


class WeightedGRPOLoss:
    """Compute time- and entropy-weighted GRPO loss from tree transitions."""

    def __init__(
        self,
        config: MCTSConfig,
        entropy_computer: EntropyComputer,
        time_weighter: TimeWeighter,
        mask_id: int,
    ):
        self.config = config
        self.entropy_computer = entropy_computer
        self.time_weighter = time_weighter
        self.mask_id = mask_id

    def compute_loss(
        self,
        model: torch.nn.Module,
        root: MCTSNode,
        leaves: List[MCTSNode],
        prompt: str,
        vocab_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (loss, metrics_dict)."""
        transitions = self._collect_transitions(root, vocab_size)
        if not transitions:
            return torch.tensor(0.0, device=next(model.parameters()).device), {}

        total_loss = 0.0
        # Debug aggregates to understand stability of the weighted GRPO signal.
        n = len(transitions)
        sum_abs_adv = 0.0
        sum_w_time = 0.0
        sum_w_ent = 0.0
        sum_weight = 0.0
        sum_weighted_term = 0.0  # - w * A * log_prob per transition

        for trans in transitions:
            log_prob = self._log_prob_transition(
                model, trans.parent_state, trans.child_state, trans.parent_attention_mask
            )
            w_time = trans.time_weight
            w_ent = trans.entropy_weight
            w = self.config.alpha_time * w_time + self.config.alpha_entropy * w_ent
            contrib = -w * trans.advantage * log_prob
            total_loss = total_loss + contrib

            # Floating-point summaries (on CPU) for logging.
            lp_val = float(log_prob.item())
            adv_val = float(trans.advantage)
            sum_abs_adv += abs(adv_val)
            sum_w_time += float(w_time)
            sum_w_ent += float(w_ent)
            sum_weight += float(w)
            sum_weighted_term += float(-w * adv_val * lp_val)

        loss = total_loss / n
        metrics = {
            "loss": loss.item(),
            "n_transitions": n,
            "mean_abs_adv": sum_abs_adv / n,
            "mean_w_time": sum_w_time / n,
            "mean_w_ent": sum_w_ent / n,
            "mean_weight": sum_weight / n,
            "mean_weighted_adv_logp": sum_weighted_term / n,
        }
        return loss, metrics

    def _collect_transitions(self, root: MCTSNode, vocab_size: int) -> List[TreeTransition]:
        out: List[TreeTransition] = []

        def go(node: MCTSNode) -> None:
            for c in node.children:
                tw = self.time_weighter.get_weight(node.step_index)
                expected_h = self.entropy_computer.expected_entropy(
                    node.masking_ratio(), vocab_size=vocab_size
                )
                if expected_h < 1e-6:
                    ew = 0.0
                else:
                    ew = (node.entropy or 0.0) / expected_h
                # Clamp entropy weight for stability (D-014)
                ew = max(
                    getattr(self.config, "entropy_weight_min", 0.5),
                    min(getattr(self.config, "entropy_weight_max", 2.0), ew),
                )
                trans = TreeTransition(
                    parent_state=node.state,
                    child_state=c.state,
                    parent_attention_mask=node.attention_mask,
                    child_attention_mask=c.attention_mask,
                    step_index=node.step_index,
                    advantage=c.advantage or 0.0,
                    entropy=node.entropy or 0.0,
                    time_weight=tw,
                    entropy_weight=ew,
                )
                out.append(trans)
                go(c)

        go(root)
        return out

    def _log_prob_transition(
        self,
        model: torch.nn.Module,
        parent_state: torch.Tensor,
        child_state: torch.Tensor,
        parent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log p(child | parent): only positions that changed (were mask -> unmasked) contribute."""
        changed = (parent_state != child_state) & (parent_state == self.mask_id)
        if not changed.any():
            return torch.tensor(0.0, device=parent_state.device)
        logits = model(
            parent_state.unsqueeze(0),
            attention_mask=parent_attention_mask.unsqueeze(0),
        ).logits[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(
            -1, child_state.unsqueeze(-1)
        ).squeeze(-1)
        return (token_lp * changed.float()).sum()

    def trajectory_log_prob(
        self,
        model: torch.nn.Module,
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Sum of log prob over (parent_state, child_state, parent_attn) for baseline GRPO."""
        if not transitions:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        total = sum(
            self._log_prob_transition(model, p, c, a) for p, c, a in transitions
        )
        return total
