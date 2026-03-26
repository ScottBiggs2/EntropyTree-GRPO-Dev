# HPC / second clone: keep Dream scripts in sync

If `python dream/scripts/single_step_dream.py -h` **does not** list `--dataset` and `--reward`, your checkout is **older** than the code-GRPO CLI. No combination of flags will add those options; you must update the files (or pull the branch that contains them).

## Quick check (run on the GPU node)

```bash
python dream/scripts/single_step_dream.py -h 2>&1 | grep -E 'dataset|reward'
python dream/scripts/run_dream_comparison.py -h 2>&1 | grep -E 'dataset|reward|max-tasks'
```

If these print nothing, you are on stale scripts.

## Fix: update the repo

From the clone root (e.g. `EntropyTree-GRPO-Dream`):

```bash
git status
git remote -v
git fetch origin
git pull origin main   # or whatever branch tracks this work (e.g. dev)
```

If this repo is only on your laptop, **copy** the updated tree from the machine that has the latest commits, or push from laptop and pull on HPC.

## Minimum file set (if you must rsync instead of git)

At least:

- `dream/scripts/single_step_dream.py`
- `dream/scripts/run_dream_comparison.py`
- `dream/src/rewards.py`
- `dream/src/task_registry.py`
- `dream/src/formatting.py`
- `dream/data/` (sample JSONL tasks)

## Always invoke scripts with `python`

Do not run `dream/scripts/foo.py` directly unless the file is executable; use:

```bash
python dream/scripts/single_step_dream.py -h
```
