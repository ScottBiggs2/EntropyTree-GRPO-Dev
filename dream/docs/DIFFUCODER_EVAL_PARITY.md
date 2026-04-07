# DiffuCoder paper vs this repo (eval)

Primary reference: [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639) (arXiv:2506.20639), **Appendix B.2 Evaluation Details** and **Tables 5–6** (prompt templates).

## Inference / benchmark defaults (paper)

| Setting | DiffuCoder (Appendix B.2) | Notes |
|--------|-----------------------------|--------|
| Max sequence length & diffusion depth | **512** and **512** | Stated when comparing to LLaDA / Dream defaults. |
| Low-temperature decoding | **0.2** | Used in the comparison sentence with Dream’s “top negative entropy remasking”. |
| Higher temperature (analysis / RL rollouts) | **1.2** | Main text + §4.3: diversity for pass@10-style search; **coupled-GRPO rollouts** use **T = 1.2** (Appendix B.1). |
| Code benchmarks | **Qwen2.5-Coder `qwencoder-eval`** | Same chat templates as Tables 5–6. |
| GSM8K | **EleutherAI `lm-evaluation-harness`** | Not `diffusion_generate` in the paper. |
| Chat templates | Tables 5–6 | Mirrored in `dream/src/eval_prompts.py` for HumanEval / MBPP / BigCodeBench-style fences. |

## Training-phase hyperparameters (paper; not identical to benchmark inference)

These matter for **reproducing training** (coupled-GRPO), not for reproducing Table 4 EvalPlus numbers:

| Setting | Where | Value |
|--------|--------|--------|
| GRPO rollouts | Appendix B.1 | **Diffusion timesteps 256**, **G = 10**, **temperature 1.2**, coupled-time range **[0.2, 0.8]** |
| PPO / GRPO coeffs | Appendix B.1 | Reference sync **64**, **μ = 2**, **β = 0.01**, **ε = 0.5**, **lr = 1e-6**, max completion length **256** |

Our **training** stack (`dream/scripts/run_dream_comparison.py`, tree trainers) uses `MCTSConfig` budgets (e.g. `max_new_tokens`, `total_denoising_steps`) that follow project defaults — they are **not** automatically aligned to the paper’s GRPO rollout table unless you set them explicitly.

## This repository’s **eval** defaults (explicit project choice)

| Setting | Our drivers (`eval_humaneval.py`, `eval_mbpp.py`, `eval_base_dream_evalplus.sbatch`) |
|--------|----------------------------------------------------------------------------------------|
| `max_new_tokens` | **512** (`--max-new-tokens`, env `MAX_NEW_TOKENS`) |
| `steps` | **128** (`--steps`, env `STEPS`) — **differs from paper inference 512** |
| `temperature` | **0.2** pass@1, **0.4** pass@10 in Slurm (`TEMP_PASS1` / `TEMP_PASS10` can override; CLI `--temperature` on scripts) |
| `top_p` | **0.95** (fixed in `eval_generate.generate_completions` unless extended) |

**CLI:** `dream/scripts/eval_humaneval.py` and `eval_mbpp.py` expose `--temperature`, `--steps`, `--max-new-tokens` — all configurable without editing code.

## Why in-harness scores can lag the paper (especially for DiffuCoder checkpoints)

1. **512 vs 128 denoising steps**: Appendix C.3 / Figure 11 discuss quality loss when using **fewer** decoding timesteps (e.g. “2× speedup”). Our default **128** steps is a deliberate cost/quality trade-off.
2. **pass@10 temperature**: We use **0.4** in `eval_base_dream_evalplus.sbatch`; the paper emphasizes **1.2** for diversity experiments — not necessarily what `qwencoder-eval` used for every table cell; match the exact setting from their released eval code when possible.
3. **GSM8K**: Paper uses **lm-eval-harness**; our `dream/scripts/eval_gsm8k.py` uses `diffusion_generate` + `apply_chat_template` — useful for **internal** tracking, not a drop-in reproduction of their GSM8K number.

See also `dream/docs/EVAL_PROTOCOL.md` and `research_decisions.md` **D-031** (training prompts vs DiffuCoder-aligned eval).
