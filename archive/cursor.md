# Notes for cursor (use as scratchpad)

dLLM Collective GitHub: https://github.com/ZHZisZZ/dllm 

dLLM Mini Model Source (which we will use for now): https://huggingface.co/dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1 

## Key Documents (read order for agents)

1. **`scaffold_plan.md`** — THE implementation plan. Phases 0-9, file structure, verification checklists. Start here.
2. **`research_decisions.md`** — All research choices (OPEN / DECIDED / DEFERRED). Check before coding.
3. **`entropy_mcts_grpo_design.md`** — Design rationale and algorithm details (conceptual pseudocode, not literal API).
4. **`entropy_mcts_implementation.md`** — Math foundations and implementation skeleton (conceptual).
5. **`branchgrpo_update.md`** — BranchGRPO stability additions (depth normalization, path-weighted fusion).
6. **`literature_reference.md`** — Paper references and reading priorities.

## Critical: Model API does NOT take a timestep parameter
The design docs show `model(z_t, timestep=t)` — this is **conceptual shorthand only**.
The real API is: `model(input_ids=z_t, attention_mask=mask).logits`
See decision D-002 in research_decisions.md.