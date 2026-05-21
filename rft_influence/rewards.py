"""Reward functions used by the surrogate trainer and the estimator.

The paper uses a binary correctness reward:
    r = 1 if the model's final answer matches the ground truth, else -1.

For GSM8K we additionally accept any reasonable numeric format
(`#### 42`, `\\boxed{42}`, plain trailing number) for robustness.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_HASH_RE = re.compile(r"####\s*([\-+]?\d[\d,]*\.?\d*)")
_TRAILING_NUM_RE = re.compile(r"([\-+]?\d[\d,]*\.?\d*)\s*\$?\s*$")


def _normalise_number(s: str) -> Optional[str]:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if v.is_integer():
        return str(int(v))
    return f"{v:.6g}"


def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Best-effort numeric answer extractor for GSM8K-style outputs."""
    if text is None:
        return None
    m = _HASH_RE.search(text)
    if m:
        return _normalise_number(m.group(1))
    m = _BOXED_RE.search(text)
    if m:
        return _normalise_number(m.group(1))
    m = _TRAILING_NUM_RE.search(text.strip())
    if m:
        return _normalise_number(m.group(1))
    return None


def gsm8k_reward(
    completions: Sequence[str],
    references: Sequence[str],
    positive: float = 1.0,
    negative: float = -1.0,
) -> List[float]:
    """Binary correctness reward used by the paper's RFT setup."""
    rewards: List[float] = []
    for pred, ref in zip(completions, references):
        gold = extract_answer_gsm8k(ref)
        ans = extract_answer_gsm8k(pred)
        if gold is None:
            rewards.append(negative)
        elif ans is not None and ans == gold:
            rewards.append(positive)
        else:
            rewards.append(negative)
    return rewards
