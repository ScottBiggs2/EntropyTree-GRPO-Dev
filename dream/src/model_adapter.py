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
            # timesteps has length total_steps+1 (indices 0..total_steps); need step+1 in range.
            if total_steps <= 0:
                return max(1, n_masked)
            timesteps = torch.linspace(1, eps, total_steps + 1)
            step_clamped = min(max(step, 0), total_steps - 1)
            t = timesteps[step_clamped].item()
            s = timesteps[step_clamped + 1].item()
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
        """Sample only on masked positions.

        Follows Dream ``generation_utils.sample_tokens``: top-p via shifted
        nucleus mask + ``finfo.min`` masking, softmax and ``Categorical`` in
        float32, with max-prob fallback if sampling fails.
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
            logits_t = self._dream_top_p_logits(logits_t, top_p)
            # FP32 softmax + sampling: bf16 underflows to all-zero probs otherwise
            # (breaks torch.distributions.Categorical simplex checks on GPU).
            logits_32 = logits_t.float()
            probs = F.softmax(logits_32, dim=-1)
            try:
                x0_m = dists.Categorical(probs=probs).sample()
                conf_m = torch.gather(probs, -1, x0_m.unsqueeze(-1)).squeeze(-1)
            except Exception:
                # Same fallback as Dream `generation_utils.sample_tokens`.
                conf_m, x0_m = probs.max(dim=-1)
            conf_m = conf_m.to(logits.dtype)
        else:
            x0_m = torch.argmax(logits_m, dim=-1)
            probs = F.softmax(logits_m.float(), dim=-1)
            conf_m = torch.gather(probs, -1, x0_m.unsqueeze(-1)).squeeze(-1).to(
                logits.dtype
            )

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

    @staticmethod
    def _dream_top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus filter matching Dream HF `generation_utils.top_p_logits`.

        Uses (1) FP32 softmax for cumulative mass, (2) the shifted remove mask
        so the boundary token is kept, and (3) `finfo(dtype).min` instead of
        `-inf` for masked logits (Dream upstream).
        """
        if top_p is None or top_p <= 0 or top_p >= 1:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits.float(), dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = False
        mask = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
        mask = mask.scatter(-1, sorted_indices, sorted_indices_to_remove)
        finfo_min = torch.finfo(logits.dtype).min
        return logits.masked_fill(mask, finfo_min)

    def _compute_tok_idx(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute position ids for Dream when there is no padding."""
        bsz, seqlen = input_ids.shape
        return torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(
            bsz, -1
        )

