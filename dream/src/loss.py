"""Weighted GRPO loss with corrected time/entropy weighting for Dream stack."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from dream.src.config import MCTSConfig
from dream.src.tree_node import MCTSNode, TreeTransition
from dream.src.entropy import EntropyComputer
from dream.src.time_weight import TimeWeighter
from dream.src.model_adapter import ModelAdapter


class WeightedGRPOLoss:
    """Compute time- and entropy-weighted GRPO loss from tree transitions."""

    def __init__(
        self,
        config: MCTSConfig,
        entropy_computer: EntropyComputer,
        time_weighter: TimeWeighter,
        mask_id: int,
        adapter: ModelAdapter,
    ):
        self.config = config
        self.entropy_computer = entropy_computer
        self.time_weighter = time_weighter
        self.mask_id = mask_id
        self.adapter = adapter

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
            dev = next(model.parameters()).device
            return torch.tensor(0.0, device=dev), {"n_transitions": 0}

        total_loss = torch.tensor(
            0.0, device=next(model.parameters()).device, dtype=torch.float32
        )
        n = len(transitions)

        # Debug aggregates.
        sum_abs_adv = 0.0
        sum_w_time = 0.0
        sum_w_ent = 0.0
        sum_weight = 0.0
        sum_weighted_term = 0.0

        for trans in transitions:
            raw_log_prob = self._log_prob_transition(
                self.adapter,
                trans.parent_state,
                trans.child_state,
                trans.parent_attention_mask,
            )
            changed = (
                (trans.parent_state != trans.child_state)
                & (trans.parent_state == self.mask_id)
            )
            n_tok = max(int(changed.sum().item()), 1)
            log_prob = raw_log_prob / n_tok

            w_time = trans.time_weight
            w_ent = trans.entropy_weight
            w = self.config.alpha_time * w_time + self.config.alpha_entropy * w_ent
            contrib = -w * float(trans.advantage) * log_prob
            total_loss = total_loss + contrib

            lp_val = float(log_prob.item())
            adv_val = float(trans.advantage)
            sum_abs_adv += abs(adv_val)
            sum_w_time += float(w_time)
            sum_w_ent += float(w_ent)
            sum_weight += float(w)
            sum_weighted_term += float(-w * adv_val * lp_val)

        loss = total_loss / n
        metrics = {
            "loss": float(loss.item()),
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
                parent_step = node.step_index
                child_step = c.step_index
                # Interval-aware time weighting.
                tw = self.time_weighter.get_interval_weight(parent_step, child_step)

                # Corrected entropy weight (H_masked_mean / log(V) by default).
                ew = self.entropy_computer.compute_entropy_weight(
                    measured_masked_mean=node.entropy or 0.0,
                    vocab_size=vocab_size,
                    masking_ratio=node.masking_ratio(),
                    mode=self.config.entropy_norm_mode,
                )
                ew = max(
                    self.config.entropy_weight_min,
                    min(self.config.entropy_weight_max, ew),
                )

                trans = TreeTransition(
                    parent_state=node.state,
                    child_state=c.state,
                    parent_attention_mask=node.attention_mask,
                    child_attention_mask=c.attention_mask,
                    step_index=parent_step,
                    child_step_index=child_step,
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
        adapter: ModelAdapter,
        parent_state: torch.Tensor,
        child_state: torch.Tensor,
        parent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log p(child | parent): only mask->token changes contribute."""
        changed = (
            (parent_state != child_state) & (parent_state == self.mask_id)
        )
        if not changed.any():
            return torch.tensor(0.0, device=parent_state.device)
        logits = adapter.forward_logits(
            parent_state.unsqueeze(0),
            parent_attention_mask.unsqueeze(0),
        )[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, child_state.unsqueeze(-1)).squeeze(-1)
        return (token_lp * changed.float()).sum()

