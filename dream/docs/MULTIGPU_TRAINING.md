# Multi-GPU training (DDP) — verification playbook

This repo’s Tree-GRPO loop is **not a DataLoader**; it iterates a Python list of prompts and performs one optimizer step per prompt. Under DDP, **every rank must execute the same number of backward calls**, or the run can hang. The current implementation pads each rank’s prompt shard by repeating within-shard prompts so all ranks take the same number of steps.

## Run location (common failure)

**Do not run `torchrun` on a login node.** Login nodes usually have **no GPUs**; you will see errors like `ProcessGroupNCCL is only supported with GPUs, no GPUs found!` or the script will raise that CUDA is unavailable.

- **Interactive GPU shell** (Explorer-style example; adjust partition/GRES to your site):

```bash
srun --partition=gpu --gres=gpu:2 --cpus-per-task=8 --mem=64G --time=01:00:00 --pty bash
# then: conda activate ... ; cd repo ; torchrun ...
```

- **Batch**: use `sbatch train_acecode_mcts_ramp_evalpipe_mgpu.sbatch` (or your job script) so processes start **on a compute node with GPUs**.

## 0) Pre-flight

- Ensure you can run the single-GPU evalpipe successfully first:
  - `baseline_train_final.pt` is created
  - `adapter/` exists (when `--lora`)
  - eval jobs are submitted via `afterok`

## 1) DDP smoke test (no execution reward)

Use a cheap reward so you only validate distributed wiring.

```bash
export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  dream/scripts/run_dream_comparison.py \
  --phase baseline_train \
  --device cuda \
  --run_name ddp_smoke_2gpu \
  --dataset "$ACECODE_JSONL" \
  --dataset-split train \
  --max-tasks 4 \
  --num_epochs 1 \
  --reward syntax \
  --execution-backend subprocess \
  --lora \
  --max_tree_nodes 3 \
  --branch_width 2 \
  --steps_per_expansion 2 \
  --max_new_tokens 32 \
  --total_denoising_steps 32 \
  --save-checkpoints \
  --checkpoint-dir "/scratch/$USER/entropy_tree_grpo_dream"
```

**Verify**
- Logs include `ddp=1 rank=... local_rank=... world_size=... device=cuda:<n>` on each rank.
- Only **rank 0** creates:
  - `/scratch/$USER/entropy_tree_grpo_dream/dream_comparison/ddp_smoke_2gpu/baseline_train_final.pt`
  - `/scratch/$USER/entropy_tree_grpo_dream/dream_comparison/ddp_smoke_2gpu/adapter/`

## 2) DDP smoke test with Apptainer execution reward

```bash
export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl
export SANDBOX_SIF=/scratch/$USER/containers/dream-sandbox.sif
export DREAM_SANDBOX_TMP_ROOT=/scratch/$USER/dream_sandbox_tmp

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  dream/scripts/run_dream_comparison.py \
  --phase baseline_train \
  --device cuda \
  --run_name ddp_exec_2gpu \
  --dataset "$ACECODE_JSONL" \
  --dataset-split train \
  --max-tasks 2 \
  --num_epochs 1 \
  --reward execution_shaped \
  --reward-timeout 2.5 \
  --execution-backend apptainer \
  --sandbox-image "$SANDBOX_SIF" \
  --lora \
  --max_tree_nodes 3 \
  --branch_width 2 \
  --steps_per_expansion 2 \
  --max_new_tokens 32 \
  --total_denoising_steps 32 \
  --save-checkpoints \
  --checkpoint-dir "/scratch/$USER/entropy_tree_grpo_dream"
```

If Apptainer/fs becomes unstable due to concurrency, retry with per-node serialization:

```bash
export DREAM_SANDBOX_MAX_CONCURRENT_PER_NODE=1
```

**Verify**
- Temporary sandbox dirs are rank-scoped (prefix includes `r<rank>`).
- Run completes without temp directory collisions.

## Troubleshooting: job exits with `rc=1`, torchrun shows one rank failed

Slurm stderr often only shows `ChildFailedError` and `rank N exitcode: 1` without the Python traceback. Get the real error:

```bash
# Full stderr (traceback is usually hundreds of lines above the elastic summary)
less -S logs/pipeline_<JOBID>_p1_mcts_mgpu.err
grep -E "Traceback|Error|CUDA out of memory|NCCL" logs/pipeline_<JOBID>_p1_mcts_mgpu.err
```

Common causes:

- **DDP + LoRA + variable tree shapes**: some adapter weights can be unused in a given step; DDP must use `find_unused_parameters=True` (set in code). If you still see reduction errors, try smaller `MAX_TREE_NODES` / `MAX_NEW_TOKENS`.
- **OOM on one rank**: different prompts → different peak memory; one rank can OOM while others look fine. Reduce budgets or set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Apptainer**: rare per-rank failures; try `export DREAM_SANDBOX_MAX_CONCURRENT_PER_NODE=1`.

For more detail on the next run:

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
```

## 3) Full pipeline job (4 GPUs) with automatic afterok eval

```bash
export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl
sbatch train_acecode_mcts_ramp_evalpipe_mgpu.sbatch
```

**Verify**
- Training log shows 4 ranks starting (`world_size=4`).
- Final artifacts exist:
  - `$CHECKPOINT_DIR/dream_comparison/$RUN_NAME/baseline_train_final.pt`
  - `$CHECKPOINT_DIR/dream_comparison/$RUN_NAME/adapter/`
  - `$CHECKPOINT_DIR/dream_comparison/$RUN_NAME/checkpoints_manifest.jsonl`
- The job submits eval jobs exactly once (from the parent sbatch script), and they begin after the train job completes successfully (`afterok` dependency).

