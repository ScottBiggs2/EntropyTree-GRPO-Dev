\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage{graphicx}

\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  urlcolor=blue,
  citecolor=blue
}

\setlist[itemize]{leftmargin=*,noitemsep,topsep=2pt}
\setlist[enumerate]{leftmargin=*,noitemsep,topsep=2pt}

\title{\vspace{-1.2em}Progress Report: Entropy-Guided Tree GRPO for Diffusion Language Models (Dream 7B Track)\vspace{-0.6em}}
\author{Scott Biggs}
\date{April 1, 2026}

\begin{document}
\maketitle
\vspace{-1.0em}

\begin{abstract}
This project studies whether \textbf{entropy-guided tree search} can improve \textbf{GRPO} training for \textbf{diffusion language models} on code-generation tasks. The current milestone is a working, validated Dream 7B code-GRPO stack in \texttt{dream/} with (i) a flat trajectory GRPO baseline and (ii) an entropy-guided MCTS-style tree GRPO trainer, both using execution-based rewards. I report implementation progress, preliminary HPC/GPU smoke-test results, observed failure modes (reward degeneracy and GPU OOM), and the next experiments needed to produce benchmark-quality results on HumanEval+/MBPP+ via EvalPlus.
\end{abstract}

\vspace{-0.2em}
\section*{1.1 Methodology / Approach}

\subsection*{Overall approach}
I implement and compare two rollout structures for diffusion-model GRPO on code tasks:
\begin{itemize}
  \item \textbf{Flat GRPO baseline (trajectory GRPO):} sample $K$ independent completions per prompt; compute advantages by reward centering; update via GRPO using trajectory log-probabilities.
  \item \textbf{Entropy-guided tree GRPO (MCTS-style):} build a search tree over intermediate denoising states; expand the global frontier using node entropy; compute BranchGRPO-style advantages; update via a weighted GRPO loss over tree transitions.
\end{itemize}
The central research question is: \emph{holding model family, prompt/task distribution, and reward constant, does entropy-guided tree construction yield better learning signal and downstream code performance than flat trajectory GRPO under comparable compute budgets?}

\subsection*{Model / system design}
\paragraph{Base model.}
The primary target model is \textbf{Dream-org/Dream-v0-Instruct-7B}, an instruct-tuned diffusion LLM. The Dream work is isolated in \texttt{dream/} to avoid disturbing the earlier toy MDLM stack in \texttt{src/}.

\paragraph{Key components implemented (Dream stack).}
The Dream stack implements the following core algorithmic and engineering components:
\begin{itemize}
  \item \textbf{ModelAdapter abstraction} to support Dream vs MDLM-style interfaces.
  \item \textbf{Corrected entropy normalization} for masked-position uncertainty signals (node entropy and loss weighting).
  \item \textbf{Interval-aware time weighting} for variable-length tree edges (crucial once adaptive branching is enabled).
  \item \textbf{Adaptive stepping / branching} (uncertainty-triggered branching within bounds) plus fixed-step branching baselines.
  \item \textbf{Two trainers:} \texttt{BaselineGRPOTrainer} (flat) and \texttt{EntropyMCTSTrainer} (tree) with a \textbf{grouped weighted GRPO loss}.
  \item \textbf{LoRA training} as the default low-memory adaptation mode.
  \item \textbf{Trajectory observability / diversity metrics} for diffusion trees (e.g., schedule uniqueness vs text uniqueness).
  \item \textbf{Execution-first code rewards} with task registry + formatting/extraction utilities.
\end{itemize}

\subsection*{Data}
\paragraph{Current bring-up data.}
The training loop currently supports JSONL code tasks with execution-based tests; the repo includes small sample datasets under \texttt{dream/data/} for smoke testing.

\paragraph{Scale-up target data.}
The active plan is to train on the \textbf{AceCode-89K hard split} (DiffuCoder-aligned filtering) and evaluate on \textbf{HumanEval+} and \textbf{MBPP+}. Conversion scripts exist for AceCode, HumanEval, and MBPP to a shared \texttt{CodeTask} JSONL schema.

\subsection*{Reward and execution}
\paragraph{Primary reward.}
The training objective uses an \textbf{execution-shaped reward} computed by running model-generated code against test cases. The system supports multiple test formats (assertion strings and args/expected pairs), which is critical because AceCode uses assertion-style tests.

\paragraph{Sandboxing and backends.}
To safely execute untrusted code, the project provides a container sandbox image (Docker locally, Apptainer on HPC) and a pluggable execution backend abstraction. Training scripts expose flags to select the backend and sandbox image.

\subsection*{Evaluation metrics and protocol}
\paragraph{Primary metrics.}
The primary downstream evaluation is \textbf{pass@1} (and optionally pass@10) on \textbf{HumanEval+} and \textbf{MBPP+}, scored via \textbf{EvalPlus}.

\paragraph{Prompt regimes (important confound).}
There are two prompting regimes:
\begin{itemize}
  \item \textbf{Training-aligned prompts:} build a user message from the task’s canonical prompt and apply the model chat template.
  \item \textbf{DiffuCoder-aligned eval prompts:} explicit system/user/assistant prefill templates matching DiffuCoder-style evaluation prompts.
\end{itemize}
Because these regimes may not match by default, experiments must explicitly record which prompt template is used for evaluation and (optionally) align training prompts accordingly.

\subsection*{Baselines and experiment arms}
The comparison runner supports multiple phases/arms. Table~\ref{tab:arms} summarizes the main ones used for current validation and planned comparisons.

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}p{2.4cm}p{3.2cm}p{4.8cm}@{}}
\toprule
\textbf{Arm / Phase} & \textbf{Rollout structure} & \textbf{What it isolates} \\
\midrule
\texttt{initial\_eval} & Tree build + eval only (no optimizer) & Tree construction + reward plumbing + observability metrics \\
\texttt{grpo\_lora\_baseline} & Flat trajectory GRPO + LoRA & Baseline learning signal and reward variance (no tree) \\
\texttt{baseline\_train} & Entropy-guided tree GRPO + LoRA & Tree-based credit assignment and weighted loss (fixed stepping) \\
\texttt{adaptive\_*} & Tree GRPO + adaptive stepping & Effect of uncertainty-triggered branching vs fixed-step edges \\
\bottomrule
\end{tabular}
\caption{Core Dream comparison arms implemented in \texttt{dream/scripts/run\_dream\_comparison.py}.}
\label{tab:arms}
\end{table}

\subsection*{Analysis plan}
Beyond aggregate benchmark scores, I will analyze:
\begin{itemize}
  \item \textbf{Reward variance / degeneracy} across sampled completions (critical for GRPO gradients).
  \item \textbf{Tree observability metrics} (trajectory diversity vs text diversity) to detect branch collapse.
  \item \textbf{Stability diagnostics} from the weighted loss (clamps, effective weights, transition counts).
  \item \textbf{Compute/memory scaling} (tree budget vs token budget vs VRAM; OOM thresholds).
  \item \textbf{Ablations} over $(\texttt{max\_tree\_nodes}, \texttt{branch\_width}, \texttt{steps\_per\_expansion})$ and weight-balancing hyperparameters.
\end{itemize}

\section*{1.2 Current Results / Progress}

\subsection*{Implementation progress (completed)}
The Dream track (\texttt{dream/}) is now a self-contained stack with the following implemented and validated capabilities:
\begin{itemize}
  \item \textbf{Core trainers:} flat GRPO baseline and entropy-tree GRPO trainer; LoRA support; grouped weighted GRPO loss.
  \item \textbf{Corrected mechanics:} entropy normalization, interval-aware time weighting, and adaptive stepping.
  \item \textbf{Code task pipeline:} task registry/schema, prompt formatting, robust code extraction, reward factory.
  \item \textbf{Execution infrastructure:} container sandbox image supporting assertion and args/expected tests; Docker+Apptainer support via an execution backend abstraction; script-level CLI wiring.
  \item \textbf{Data tooling:} converters for AceCode-89K, HumanEval, and MBPP to JSONL task format.
  \item \textbf{Evaluation tooling:} DiffuCoder-aligned prompt templates and generation harness; batch scripts for EvalPlus scoring.
  \item \textbf{Testing:} Dream unit tests pass locally (57 tests) including registry/formatting/reward and backend behavior.
\end{itemize}

\subsection*{Preliminary HPC/GPU results}
\paragraph{(A) Tree build and evaluation-only sanity (\texttt{initial\_eval}).}
On HPC/GPU with Dream 7B and the sample dataset, tree building runs end-to-end and logs nontrivial metrics.
In one run (3 prompts), \texttt{initial\_eval} produced trees with $\approx 19$--$23$ nodes and 8 leaves, with mean execution-shaped reward $\approx 0.153$.
The run also logged diffusion-tree observability metrics (e.g., \texttt{leaf\_schedule\_unique\_frac} $\approx 0.33$ on average), indicating measurable trajectory-level diversity even when text-level diversity is modest.

\paragraph{(B) Flat GRPO baseline shows reward degeneracy on tiny data.}
For \texttt{grpo\_lora\_baseline} on the same small sample tasks, the training logs show that all $K$ sampled completions for a prompt can receive \emph{identical reward}, triggering the warning:
\emph{``all rewards identical --- advantages are zero, no gradient this step''}.
This demonstrates a current limitation: at very small scale (few tasks, coarse reward discretization), the baseline can have near-zero learning signal because GRPO requires reward variance across samples.

\paragraph{(C) Tree GRPO can be memory-limited at aggressive budgets.}
When launching the tree-training phase (\texttt{baseline\_train}) under an ablation setting, the run encountered a CUDA out-of-memory error on an A100-class GPU, failing during tree entropy computation. This indicates that the current settings (tree budget and token budget) are near the VRAM limit and motivates:
\begin{itemize}
  \item smaller default token budgets for tree training,
  \item tighter tree node caps during early scaling,
  \item and/or memory optimizations in tree entropy computation and cache management.
\end{itemize}

\subsection*{Key observations}
\begin{itemize}
  \item \textbf{The Dream code-GRPO stack is operational end-to-end} (task loading $\rightarrow$ rollout $\rightarrow$ execution reward $\rightarrow$ logging), which de-risks further scaling.
  \item \textbf{Reward variance is a first-order bottleneck for GRPO} on small task sets: identical rewards across samples imply zero advantage and no update.
  \item \textbf{Tree training introduces real VRAM pressure} beyond evaluation-only runs; budgets must be tuned and profiled before large ablations.
  \item \textbf{Trajectory diversity metrics are necessary}: diffusion trees can explore distinct denoising schedules even when final code text matches, so relying on output-text uniqueness alone would under-measure exploration.
\end{itemize}

\subsection*{Challenges / limitations}
\begin{itemize}
  \item \textbf{Data limitations (current):} the sample dataset is too small and rewards too coarse to reliably provide GRPO gradients.
  \item \textbf{Compute constraints:} tree+backward training can hit OOM under large token or tree budgets even on large GPUs.
  \item \textbf{Evaluation discipline:} training prompts and DiffuCoder-aligned eval prompts may differ; this must be documented and potentially aligned to avoid misinterpreting benchmark results.
  \item \textbf{Throughput:} full HumanEval+/MBPP+ evaluation is expensive (many tasks $\times$ diffusion steps $\times$ samples); careful batching and caching are required.
\end{itemize}

\subsection*{Next steps (concrete)}
\begin{enumerate}
  \item \textbf{AceCode subset training validation (immediate):} run end-to-end GRPO on a $\sim$100-task AceCode subset with execution reward enabled, using conservative tree/token budgets (and both subprocess and container execution backends).
  \item \textbf{Stabilize learning signal:} increase reward granularity/variance (e.g., larger task set; richer tests) and tune $K$ to avoid reward degeneracy in flat GRPO; then revisit tree vs flat comparisons.
  \item \textbf{Memory/throughput profiling:} map safe regions in $(\texttt{max\_tree\_nodes}, \texttt{max\_new\_tokens}, \texttt{branch\_width})$; adopt defaults that avoid OOM.
  \item \textbf{Benchmark baseline eval:} run EvalPlus on the \emph{base} Dream model with DiffuCoder-aligned prompts to establish a reference pass@1/pass@10.
  \item \textbf{Checkpoint evaluation:} evaluate trained LoRA adapters on HumanEval+/MBPP+ via the same protocol; compare tree-trained vs flat-trained adapters under matched compute budgets.
\end{enumerate}

\paragraph{Expansion: math / CoT tasks (verifier-first).}
After code GRPO is stable at AceCode scale, the next domain expansion is to \textbf{math and reasoning} with verifier-based rewards. Concretely:
\begin{itemize}
  \item Start with \textbf{answer-only optimization} (no enforced visible chain-of-thought) and a deterministic verifier (exact numeric/string match or unit tests for symbolic math) to minimize reward hacking.
  \item Add a \textbf{math task schema} parallel to \texttt{CodeTask} (prompt, target answer, verifier spec), and reuse the same training loop with a swapped reward function.
  \item Only then explore \textbf{CoT-formatted outputs} as an ablation (format constraints and extraction become part of the experimental condition).
\end{itemize}

\paragraph{Multi-GPU compatibility.}
To scale beyond single-GPU runs, the training harness should be made robust under \textbf{DDP/Accelerate}:
\begin{itemize}
  \item Ensure tree rollout + reward evaluation are \textbf{rank-consistent}: each rank should process distinct prompts/tasks, but log and checkpoint in a coordinated way (rank 0 aggregation).
  \item Address \textbf{execution backends} in distributed settings (container/subprocess calls): enforce per-rank isolation of temp directories and rate-limit concurrent sandbox calls.
  \item Validate that LoRA parameter partitioning and gradient synchronization are correct, and that memory-saving options (gradient checkpointing, mixed precision) behave consistently across ranks.
\end{itemize}

\paragraph{Primary ablation hyperparameters.}
The most important knobs for systematic ablations (after the 100-task AceCode validation) are:
\begin{itemize}
  \item \textbf{Tree budget/shape:} \texttt{max\_tree\_nodes}, \texttt{branch\_width}, \texttt{steps\_per\_expansion}; and for adaptive stepping: \texttt{branch\_threshold}, \texttt{min\_steps\_per\_expansion}, \texttt{max\_steps\_per\_expansion}.
  \item \textbf{Generation budget:} \texttt{max\_new\_tokens} and \texttt{total\_denoising\_steps} (and the eval-side \texttt{steps} parameter when running EvalPlus scripts).
  \item \textbf{Sampling/diversity:} \texttt{temperature}, \texttt{top\_p}, and the training-time sampling temperature used for rollout diversity.
  \item \textbf{GRPO signal/control:} number of baseline samples $K$ (\texttt{num\_baseline\_samples}), advantage clipping, and entropy/time weight scaling and clamps (e.g., \texttt{alpha\_entropy}, \texttt{alpha\_time}, \texttt{entropy\_weight\_min/max}).
  \item \textbf{Execution reward settings:} backend choice (subprocess vs container), per-task timeout, and reward tie-breaking/shaping choices that affect reward variance.
\end{itemize}

\section*{1.3 References}
\begin{thebibliography}{9}
\bibitem{dream7b}
Y.~Ye et~al.
\newblock \emph{Dream 7B: Diffusion Large Language Models}.
\newblock arXiv:2508.15487, 2025.
\newblock \href{https://arxiv.org/abs/2508.15487}{https://arxiv.org/abs/2508.15487}

\bibitem{treegrpo}
Y.~Ding and Y.~Ye.
\newblock \emph{TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models}.
\newblock ICLR 2026 submission (OpenReview).
\newblock \href{https://openreview.net/forum?id=3rZdp4TmUb}{https://openreview.net/forum?id=3rZdp4TmUb}

\bibitem{tempflowgrpo}
Z.~He et~al.
\newblock \emph{TempFlow-GRPO: When Timing Matters for GRPO in Flow Models}.
\newblock arXiv:2508.04324, 2025.
\newblock \href{https://arxiv.org/abs/2508.04324}{https://arxiv.org/abs/2508.04324}

\bibitem{diffucoder}
Y.~Gong et~al.
\newblock \emph{DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation}.
\newblock arXiv:2506.20639, 2025.
\newblock \href{https://arxiv.org/abs/2506.20639}{https://arxiv.org/abs/2506.20639}

\bibitem{evalplus}
EvalPlus contributors.
\newblock \emph{EvalPlus: Rigorous Evaluation of LLM-Generated Code}.
\newblock \href{https://github.com/evalplus/evalplus}{https://github.com/evalplus/evalplus}

\bibitem{acecoder}
TIGER-AI-Lab.
\newblock \emph{AceCoder / AceCode-89K}.
\newblock \href{https://github.com/TIGER-AI-Lab/AceCoder}{https://github.com/TIGER-AI-Lab/AceCoder}
\end{thebibliography}

\end{document}