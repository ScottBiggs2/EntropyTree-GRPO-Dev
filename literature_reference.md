# Literature Reference Guide

Quick reference for papers and repositories relevant to entropy-guided MCTS-GRPO training for diffusion language models.

---

## Core Diffusion Language Models

### MDLM: Masked Diffusion Language Models
**Paper**: [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) (NeurIPS 2024)  
**Authors**: Sahoo et al. (Cornell)  
**GitHub**: https://github.com/kuleshov-group/mdlm  
**HuggingFace**: https://huggingface.co/kuleshov-group/mdlm-owt

**Key Contributions**:
- Substitution-based parameterization simplifies absorbing-state diffusion to mixture of masked language modeling losses
- Achieves SOTA perplexity among diffusion models, approaches autoregressive performance
- Encoder-only architecture with efficient samplers (3-4x faster than D3PM/SEDD)
- Exact Shannon entropy available from discrete token distributions

**Relevance**: Foundation for our entropy computation—MDLM gives us exact $H = -\sum_v p(v) \log p(v)$ per token.

---

### BD3LM: Block Discrete Denoising Diffusion
**Paper**: [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) (ICLR 2025 Oral)  
**Authors**: Arriola et al. (Cornell)  
**GitHub**: https://github.com/kuleshov-group/bd3lms  
**HuggingFace**: https://huggingface.co/collections/kuleshov-group/bd3-lms-6868ccc95a81c4a8fa4e0a47

**Key Contributions**:
- Interpolates between AR and diffusion by decomposing sequences into blocks, performing discrete diffusion within each block
- Achieves SOTA likelihoods among diffusion models, enables arbitrary-length generation
- Data-driven noise schedules minimize gradient variance
- Consistently outperforms vanilla MDLM on math and code tasks

**Relevance**: Shows block-based decomposition reduces variance—potential extension for our tree construction (coarse-grained branching).

---

### dLLM Framework
**Repository**: https://github.com/ZHZisZZ/dllm  
**Authors**: Zhou et al. (Berkeley)  
**Models**: https://huggingface.co/collections/dllm-collection/tiny-a2d-6787dc07e2cb38f5e37d2c0f

**Key Contributions**:
- Minimal, clean implementation for training MDLM and BD3LM models
- Recipes for converting any autoregressive model (Qwen, LLaMA, GPT-2) to diffusion
- Provides Qwen3-0.6B and Qwen2.5-Coder-0.5B diffusion variants (our target models)
- Supports both masked diffusion and block diffusion training

**Relevance**: Primary codebase we build on—well-structured, actively maintained, small models for fast iteration.

---

### Dream: Large-Scale Diffusion LLM
**Paper**: [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)  
**Authors**: Ye et al. (Alibaba/HKU)  
**GitHub**: https://github.com/DreamLM/Dream  
**HuggingFace**: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B

**Key Contributions**:
- 7B parameter diffusion LLM with competitive performance to AR models of similar size
- AR-based initialization + context-adaptive token-level noise rescheduling
- Superior planning abilities, arbitrary-order generation, infilling capabilities
- Demonstrates diffusion LLMs can scale to production-quality models

**Relevance**: Validation that diffusion LLMs work at scale—our methods should eventually scale from 0.5B to 7B.

---

## Reinforcement Learning for Diffusion Models

### TreeGRPO: Tree-Based GRPO for Diffusion
**Paper**: [TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models](https://openreview.net/forum?id=3rZdp4TmUb) (ICLR 2026)  
**Authors**: Ding & Ye

**Key Contributions**:
- Recasts denoising as search tree, branches from shared initial noise for computational efficiency
- Fine-grained credit assignment via reward backpropagation computing step-specific advantages
- Achieves 2.4× faster training while establishing superior Pareto frontier in efficiency-reward trade-off
- Amortized computation: multi-child branching enables multiple policy updates per forward pass

**Relevance**: Core inspiration—we adapt their tree structure but use entropy for branching instead of uniform timesteps.

---

### TempFlow-GRPO: Time-Aware GRPO
**Paper**: [TempFlow-GRPO: When Timing Matters for GRPO in Flow Models](https://arxiv.org/abs/2508.04324)  
**Authors**: He et al.

**Key Contributions**:
- Trajectory branching mechanism for precise credit assignment without intermediate reward models
- Noise-aware weighting modulates optimization according to exploration potential at each timestep
- Prioritizes learning during high-impact early stages while ensuring stable refinement later
- Achieves SOTA performance in human preference alignment on text-to-image benchmarks

**Relevance**: Provides our time-weighting formula $w_{\text{time}}(t) = (1 - t/T)^2$—early steps matter more.

---

### Flow-GRPO: GRPO for Flow Matching
**Paper**: [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)  
**Authors**: Liu et al.

**Key Contributions**:
- ODE-to-SDE conversion enables statistical sampling for RL exploration in flow models
- Denoising Reduction strategy improves efficiency by reducing training steps while preserving inference quality
- Dramatic improvements: GenEval 63%→95%, visual text rendering 59%→92%
- Minimal reward hacking observed

**Relevance**: Shows GRPO works for continuous diffusion—validates approach, though we focus on discrete (MDLM).

---

### DiffuCoder: Coupled-GRPO for Code
**Paper**: [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)  
**Authors**: Gong et al. (Apple)  
**GitHub**: https://github.com/apple/ml-diffucoder  
**HuggingFace**: https://huggingface.co/collections/apple/diffucoder-6868139f56672ae046fe04e8

**Key Contributions**:
- Coupled-GRPO: constructs complementary mask noise for variance reduction in RL training
- dLLMs can decide generation causality without semi-AR decoding
- Temperature diversifies both token choices AND generation order (rich search space for RL)
- Achieves +4.4% improvement on EvalPlus code benchmarks

**Relevance**: Shows GRPO works for discrete diffusion on code—validates our target domain. Coupled-GRPO is complementary to our entropy approach.

---

## MCTS and Exploration Strategies

### DeepSearch: MCTS in RLVR Training
**Paper**: [DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search](https://arxiv.org/abs/2509.25454)  
**Authors**: Various (OpenReview submission)

**Key Contributions**:
- **Global frontier selection**: prioritizes promising nodes across entire search tree (not just local children)
- **Entropy-based guidance**: identifies confident paths for supervision
- Achieves 62.95% on AIME with 1.5B model (new SOTA for size), uses 5.7× fewer GPU hours than extended training
- Critical finding: breadth over depth—larger exploration constants needed to overcome model confidence biases

**Relevance**: **Primary inspiration for our exploration strategy**—we adopt global frontier selection with entropy as node selection criterion.

---

### McDiffuSE: MCTS for Diffusion Slot Ordering
**Paper**: [Can I Have Your Order? Monte-Carlo Tree Search for Slot Filling Ordering in Diffusion Language Models](https://arxiv.org/abs/2602.12586)  
**Authors**: Various (recent arXiv)

**Key Contributions**:
- Formulates slot selection as decision-making problem, optimizes infilling orders through MCTS
- Uses model's intrinsic confidence scores as prior probabilities for MCTS expansion
- Hybrid reward: immediate denoising quality + rollout-based long-term trajectory coherence
- Critical finding: needs HIGH exploration breadth (large exploration constant), not deep simulation

**Relevance**: Alternative MCTS approach focused on slot ordering—we considered but chose DeepSearch's global selection instead (see design doc for comparison).

---

### TreeRL: On-Policy Tree Search for LLM RL
**Paper**: [TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](https://arxiv.org/abs/2506.11902) (ACL 2025)  
**Authors**: Hou et al.

**Key Contributions**:
- Directly incorporates on-policy tree search into LLM RL training rather than relying on independent chain sampling
- Uses dense, on-policy intermediate supervision, avoiding a separate process reward model and reducing reward-model mismatch / reward hacking risk
- Shows better search efficiency under a fixed generation token budget by branching from **high-uncertainty intermediate states** rather than random branch points
- Improves performance on challenging math and code reasoning benchmarks relative to chain-based RL baselines

**Relevance**: Strong conceptual support for our move away from fixed branch intervals toward **dynamic branching at uncertainty peaks**. It also reinforces a more process-level view of supervision, which matters if we later move from macro-step edges to finer-grained transition logging.

---

### Fast-MCTD: Efficient MCTS for Diffusion Planning
**Paper**: [Fast Monte Carlo Tree Diffusion: 100× Speedup via Parallel Sparse Planning](https://arxiv.org/abs/2506.09498)  
**Authors**: Yoon et al.

**Key Contributions**:
- Addresses computational inefficiency of sequential MCTS and iterative denoising
- Macro-step expansion: groups multiple denoising steps into single transition (reduces tree depth)
- Achieves 100× speedup through parallelization and sparse planning
- Selective, partial denoising allows generation of only high-uncertainty tokens

**Relevance**: Provides our coarse-grained branching strategy—expand by 32-64 steps, not 1 step, to keep trees manageable.

---

### Entropy-Guided MCTS (General)
**Paper**: [Entropy-Guided Exploration in AlphaZero: Enhancing MCTS with Information Gain for Strategic Decision Making](https://dl.acm.org/doi/10.1145/3696271.3696296)  
**Authors**: Li et al.

**Key Contributions**:
- Integrates information gain principles from information theory into MCTS selection
- Entropy quantifies uncertainty or information content of game states
- Adds new dimension to action selection beyond immediate value estimates and exploration bonuses
- Enhanced efficiency in game tree navigation and adaptation to unfamiliar situations

**Relevance**: Theoretical foundation for using entropy as exploration bonus in tree search—applies to our diffusion setting.

---

## Experimental Benchmarks

### Code Generation
- **HumanEval**: https://github.com/openai/human-eval (164 Python programming problems)
- **MBPP**: https://github.com/google-research/google-research/tree/master/mbpp (974 Python problems)
- **EvalPlus**: https://github.com/evalplus/evalplus (Extended HumanEval with more test cases)

### Math Reasoning
- **GSM8K**: https://github.com/openai/grade-school-math (Grade school math word problems)
- **MATH**: https://github.com/hendrycks/math (Competition math problems)
- **AIME**: American Invitational Mathematics Examination problems

### General Reasoning
- **ARC-Challenge**: https://allenai.org/data/arc (Science questions requiring reasoning)

---

## Related Techniques (For Future Extension)

### Entropy in RL for LLMs
**Paper**: [Reasoning with Exploration: An Entropy Perspective](https://arxiv.org/abs/2506.14758)  
**Key Point**: High entropy correlates with pivotal tokens, reflective actions, and rare behaviors—validates using entropy for branching decisions.

### ETTRL: Entropy Test-Time RL
**Paper**: [ETTRL: Balancing Exploration and Exploitation in LLM Test-Time Reinforcement Learning](https://arxiv.org/abs/2508.11356)  
**Key Point**: Entropy-fork branching at high-entropy tokens achieves greater diversity with lower token budget—supports our branching strategy.

### Diffusion Tree Sampling
**Paper**: [Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models](https://arxiv.org/abs/2506.20701)  
**Key Point**: Casts denoising as finite-horizon tree with MCTS rollouts—similar framing to ours but for inference, not training.

---

## Key GitHub Resources

### Primary Codebases
- **dLLM**: https://github.com/ZHZisZZ/dllm (our base framework)
- **MDLM**: https://github.com/kuleshov-group/mdlm (original MDLM implementation)
- **BD3LM**: https://github.com/kuleshov-group/bd3lms (block diffusion)
- **Dream**: https://github.com/DreamLM/Dream (7B diffusion LLM)
- **DiffuCoder**: https://github.com/apple/ml-diffucoder (Apple's coupled-GRPO)

### Evaluation Tools
- **HumanEval**: https://github.com/openai/human-eval
- **EvalPlus**: https://github.com/evalplus/evalplus
- **GSM8K**: https://github.com/openai/grade-school-math

### Reference Implementations
- **MCTS for LLMs**: https://github.com/zzli2022/Awesome-System2-Reasoning-LLM (comprehensive list)

---

## Quick Decision Reference

**Q: Which paper for entropy computation?**  
A: MDLM (Sahoo et al.) → Exact Shannon entropy from softmax distributions

**Q: Which paper for time weighting?**  
A: TempFlow-GRPO (He et al.) → $w(t) = (1 - t/T)^2$

**Q: Which paper for tree structure?**  
A: TreeGRPO (Ding & Ye) → Tree-based advantage backpropagation

**Q: Which paper for exploration strategy?**  
A: DeepSearch → Global frontier selection with entropy

**Q: Which paper for dynamic branch timing?**  
A: TreeRL → Branch from high-uncertainty intermediate states under a fixed token budget

**Q: Which paper for coarse-grained branching?**  
A: Fast-MCTD (Yoon et al.) → Macro-step expansion (32-64 steps)

**Q: Which paper for discrete diffusion RL?**  
A: DiffuCoder (Apple) → Shows GRPO works for masked diffusion on code

**Q: Which codebase to build on?**  
A: dLLM (ZHZisZZ) → Clean, minimal, actively maintained, has our target models

---

## Paper Reading Priority

For quick implementation:
1. **MDLM** (Sahoo) - understand entropy computation
2. **DeepSearch** - understand global frontier selection
3. **TreeRL** (Hou et al.) - understand high-uncertainty branch timing and process supervision
4. **TreeGRPO** (Ding & Ye) - understand tree-based advantages
5. **TempFlow-GRPO** (He) - understand time weighting

For deeper understanding:
6. **DiffuCoder** (Apple) - see discrete diffusion RL in practice
7. **Fast-MCTD** (Yoon) - understand coarse-grained expansion
8. **BD3LM** (Arriola) - understand variance reduction techniques

For scaling up later:
9. **Dream** (Ye) - see how diffusion LLMs scale to 7B
10. **Flow-GRPO** (Liu) - understand continuous diffusion RL

