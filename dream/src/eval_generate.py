"""Shared Dream diffusion generation loop for EvalPlus JSONL export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from dream.src.config import MCTSConfig
from dream.src.eval_prompts import extract_diffucoder_completion
from dream.src.task_registry import CodeTask, load_code_tasks
from dream.src.utils import get_device, load_model_and_tokenizer


def _configure_hf_cache() -> None:
    """Prefer scratch-based HF caches when HF_HOME is unset (HPC-friendly)."""
    if os.environ.get("HF_HOME"):
        return
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    scratch_root = os.environ.get("SCRATCH") or "/scratch"
    hf_home = os.path.join(scratch_root, user, "hf_home")
    try:
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    except OSError:
        pass


def load_dream_model_for_eval(
    model_name_or_path: str,
    *,
    device: Optional[str] = None,
    adapter_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any]:
    """Load Dream ``AutoModel`` + tokenizer; optionally wrap a PEFT adapter."""
    dev = device or get_device()
    cfg = MCTSConfig(model_type="dream", model_name_or_path=model_name_or_path, device=dev)
    model, tokenizer = load_model_and_tokenizer(cfg)

    if adapter_path:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def _try_tqdm(it: Sequence[CodeTask], *, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, desc=desc, unit="task")
    except ImportError:
        return it


def generate_completions(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[CodeTask],
    *,
    prompt_builder: Callable[[str], str],
    slot_from_task: Callable[[CodeTask], str],
    n_samples: int = 1,
    max_new_tokens: int = 512,
    steps: int = 32,
    temperature: float = 0.2,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    device: Optional[str] = None,
    completion_postprocess: Optional[Callable[[CodeTask, str], str]] = None,
    quiet: bool = False,
) -> List[Tuple[str, str]]:
    """Generate ``n_samples`` completions per task; return ``(task_id, completion)`` rows.

    ``completion_postprocess`` can adjust text before JSONL export (e.g. HumanEval
    body-only for EvalPlus ``prompt + completion``).

    Slurm / batch jobs often buffer ``tqdm``; when ``quiet`` is False we print one
    flushed line per task so ``tail -f`` on the job log shows steady progress.
    """
    dev = torch.device(device) if device else next(model.parameters()).device

    out: List[Tuple[str, str]] = []
    task_list = list(tasks)
    # tqdm is useless in Slurm (non-TTY, buffered); prefer line-based [eval-gen] logs.
    try:
        _tty = sys.stderr.isatty()
    except Exception:
        _tty = False
    iterator = _try_tqdm(task_list, desc="Generating") if (_tty and not quiet) else task_list

    for ti, task in enumerate(iterator):
        t_task0 = time.perf_counter()
        slot = slot_from_task(task)
        full_prompt = prompt_builder(slot)
        enc = tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=dev)
        else:
            attention_mask = attention_mask.to(dev)

        for _ in range(n_samples):
            with torch.no_grad():
                generated = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    steps=steps,
                    temperature=temperature,
                    top_p=top_p,
                    alg=alg,
                    alg_temp=alg_temp,
                    return_dict_in_generate=True,
                )
            seq = generated.sequences[0]
            prompt_len = input_ids.shape[1]
            gen_ids = seq[prompt_len:]
            raw_suffix = tokenizer.decode(gen_ids, skip_special_tokens=False)
            completion = extract_diffucoder_completion(raw_suffix)
            if completion_postprocess is not None:
                completion = completion_postprocess(task, completion)
            out.append((task.task_id, completion))

        if not quiet:
            wall = time.perf_counter() - t_task0
            print(
                f"[eval-gen] {ti + 1}/{len(task_list)} task_id={task.task_id} "
                f"wall_s={wall:.1f} completion_chars={len(completion)}",
                flush=True,
                file=sys.stdout,
            )

    return out


def write_evalplus_jsonl(path: Union[str, Path], rows: Sequence[Tuple[str, str]]) -> None:
    """Write EvalPlus-style JSONL: one ``{\"task_id\", \"completion\"}`` per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task_id, completion in rows:
            obj = {"task_id": task_id, "completion": completion}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_tasks_from_jsonl(path: Union[str, Path], *, max_tasks: int = 0) -> List[CodeTask]:
    """Load :class:`CodeTask` list; ``max_tasks`` 0 means all."""
    tasks = load_code_tasks(path)
    if max_tasks and max_tasks > 0:
        return tasks[:max_tasks]
    return tasks


def run_evalplus(samples: Union[str, Path], *, dataset: str) -> int:
    """Invoke EvalPlus scoring (CLI). Returns process exit code."""
    samples = Path(samples)
    cmd_sets = [
        ["evalplus.evaluate", "--dataset", dataset, "--samples", str(samples)],
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", str(samples)],
    ]
    last_err: Optional[BaseException] = None
    for cmd in cmd_sets:
        try:
            proc = subprocess.run(cmd, check=False)
            return proc.returncode
        except FileNotFoundError as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not run EvalPlus (tried `evalplus.evaluate` and `python -m evalplus.evaluate`). "
        "Install with `pip install evalplus` and ensure the console script is on PATH."
    ) from last_err
