"""Dataset utilities. Defaults to GSM8K for the example pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset


SYSTEM_PROMPT = (
    "You are a careful math reasoner. Solve the problem step by step, "
    "then output the final answer in the form `#### <number>`."
)


@dataclass
class RFTSample:
    """A single (prompt, ground-truth) pair used by the estimator."""

    prompt: str
    answer: str  # raw GSM8K answer string (chain + #### number)
    sample_id: Optional[str] = None


def _format_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:"


def load_gsm8k(
    split: str = "train",
    limit: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = True,
) -> List[RFTSample]:
    """Load GSM8K and return a list of `RFTSample`."""
    ds = load_dataset("gsm8k", "main", split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    out: List[RFTSample] = []
    for i, row in enumerate(ds):
        out.append(
            RFTSample(
                prompt=_format_prompt(row["question"]),
                answer=row["answer"],
                sample_id=f"gsm8k-{split}-{i}",
            )
        )
    return out


def to_hf_dataset(samples: Iterable[RFTSample]) -> Dataset:
    """Convert a list of RFTSample to a HF Dataset suitable for TRL trainers.

    Produces columns: `prompt`, `answer`, `sample_id`.
    """
    rows = [
        {"prompt": s.prompt, "answer": s.answer, "sample_id": s.sample_id}
        for s in samples
    ]
    return Dataset.from_list(rows)
