"""GSM8K answer extraction and normalization for `eval_gsm8k.py`."""

from __future__ import annotations

import re
from typing import Optional


def reference_final_answer(reference: str) -> str:
    """Ground-truth final segment after ``####`` (GSM8K test format)."""
    s = (reference or "").strip()
    if "####" in s:
        return s.split("####")[-1].strip()
    return s


def extract_predicted_answer(text: str) -> str:
    """Prefer ``####`` line in model output; else last signed number-like token."""
    t = (text or "").strip()
    m = re.search(r"####\s*([^\n#]+)", t)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    if nums:
        return nums[-1].strip()
    return ""


def answers_match(predicted: str, expected: str) -> bool:
    """Loose equality: strip, ignore commas, compare numeric if both parse as float."""
    p = (predicted or "").strip().replace(",", "")
    e = (expected or "").strip().replace(",", "")
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 1e-6
    except ValueError:
        return p.lower() == e.lower()
