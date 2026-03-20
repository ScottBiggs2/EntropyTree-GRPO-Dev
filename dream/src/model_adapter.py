"""ModelAdapter abstraction for MDLM vs Dream APIs."""

from typing import Tuple

import torch
import torch.nn.functional as F
import torch.distributions as dists

from dream.src.utils import add_gumbel_noise


class ModelAdapter:
    """Encapsulate model-specific forward pass and sampling.

    This keeps the tree builder, entropy computer, and loss largely
    model-agnostic.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, model_type: str = "dream"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mask_id = tokenizer.mask_token_id
        # Use the actual logits vocabulary size when available.
        # Some Dream tokenizers can have extra/added tokens vs model logits dim.
        self.vocab_size = (
            getattr(getattr(model, "config", None), "vocab_size", None)
            or getattr(tokenizer, "vocab_size", None)
            or len(tokenizer)
        )
        self.device = next(model.parameters()).device

    # ---- Forward / logits ----

    def forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return logits [B, L, V] with model-specific transforms applied."""
        if self.model_type == "dream":
            tok_idx = self._compute_tok_idx(input_ids)
            outputs = self.model(
                input_ids, attention_mask="full", tok_idx=tok_idx
            )
            logits = outputs.logits  # [B, L, V]
            # Dream right-shift: logits[i] predicts token at position i+1 in raw output.
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            return logits

        # MDLM-style model: standard masked LM API.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    # ---- Sampling ----

    def sample_and_confidence(
        self,
        logits: torch.Tensor,
        mask_positions: torch.Tensor,
        temperature: float,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens and return (x0_pred, confidence) for all positions.

        x0_pred: [L] predicted tokens (will be ignored at non-mask positions by caller).
        confidence: [L] probability of chosen token; -inf at non-mask positions.
        """
        if self.model_type == "dream":
            return self._dream_sample(logits, mask_positions, temperature, top_p)
        return self._mdlm_sample(logits, mask_positions, temperature)

    def transfer_count(
        self, n_masked: int, step: int, total_steps: int, eps: float = 1e-3
    ) -> int:
        """Number of tokens to unmask at this step."""
        if self.model_type == "dream":
            # Dream's linear timestep schedule from 1 → eps.
            timesteps = torch.linspace(1, eps, total_steps + 1)
            t = timesteps[step].item()
            s = timesteps[step + 1].item()
            return max(1, int(n_masked * (1 - s / t)))

        # MDLM uniform schedule over remaining steps.
        steps_left = max(total_steps - step, 1)
        return max(1, n_masked // steps_left)

    # ---- Internal helpers ----

    def _dream_sample(
        self,
        logits: torch.Tensor,
        mask_positions: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample only on masked positions.

        Dream sometimes produces degenerate probability tensors on
        non-masked positions after top-p filtering in bf16 (all zeros),
        which breaks `torch.distributions.Categorical`.
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected logits shape [L, V], got {logits.shape}")

        L, V = logits.shape
        if mask_positions.shape[0] != L:
            raise ValueError(
                f"mask_positions shape mismatch: {mask_positions.shape} vs logits {logits.shape}"
            )

        # Defaults for non-masked positions (caller will ignore x0 there).
        x0 = torch.zeros(L, dtype=torch.long, device=logits.device)
        conf = torch.full((L,), -1e9, dtype=logits.dtype, device=logits.device)

        if not mask_positions.any():
            return x0, conf

        idx = mask_positions.nonzero(as_tuple=False).squeeze(-1)  # [M]
        logits_m = logits.index_select(0, idx)  # [M, V]

        if temperature > 0:
            logits_t = logits_m / temperature
            logits_t = self._top_p_filter(logits_t, top_p)
            probs = F.softmax(logits_t, dim=-1)

            # Safety: if probs are degenerate (all zeros / non-finite), fallback to argmax.
            row_sums = probs.sum(dim=-1)
            invalid = (~torch.isfinite(row_sums)) | (row_sums <= 0)
            if invalid.any():
                x0_m = torch.argmax(logits_t, dim=-1)
                # Use a normalized one-hot for confidence on invalid rows.
                conf_m = torch.zeros_like(x0_m, dtype=logits.dtype)
                if (~invalid).any():
                    good_idx = (~invalid).nonzero(as_tuple=False).squeeze(-1)
                    probs_good = probs.index_select(0, good_idx)
                    x0_good = dists.Categorical(probs=probs_good).sample()
                    conf_good = torch.gather(
                        probs_good, -1, x0_good.unsqueeze(-1)
                    ).squeeze(-1)
                    x0_m = x0_m.clone()
                    conf_m = conf_m.clone()
                    x0_m[good_idx] = x0_good
                    conf_m[good_idx] = conf_good
                x0[idx] = x0_m
                conf[idx] = conf_m
                return x0, conf

            x0_m = dists.Categorical(probs=probs).sample()
            conf_m = torch.gather(probs, -1, x0_m.unsqueeze(-1)).squeeze(-1)
        else:
            x0_m = torch.argmax(logits_m, dim=-1)
            probs = F.softmax(logits_m, dim=-1)
            conf_m = torch.gather(probs, -1, x0_m.unsqueeze(-1)).squeeze(-1)

        x0[idx] = x0_m
        conf[idx] = conf_m
        return x0, conf

    def _mdlm_sample(
        self,
        logits: torch.Tensor,
        mask_positions: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_n = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_n, dim=-1)
        probs = F.softmax(logits, dim=-1)
        conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(
            mask_positions,
            conf,
            torch.full_like(conf, -1e9),
        )
        return x0, conf

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p filtering to logits along the vocabulary dimension."""
        if top_p <= 0 or top_p >= 1:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # Mask tokens above the nucleus threshold.
        mask = cumulative_probs > top_p
        # Ensure at least one token is kept even if top_p is tiny.
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        # Scatter back to original order.
        return torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)

    def _compute_tok_idx(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute position ids for Dream when there is no padding."""
        bsz, seqlen = input_ids.shape
        return torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(
            bsz, -1
        )

