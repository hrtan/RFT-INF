"""Surrogate RFT trainer that produces the per-epoch checkpoints used by
the RFT-Inf estimator (Algorithm 1, line 1 of the paper).

The trainer is a thin wrapper around TRL's :class:`GRPOTrainer`. It runs
E LoRA epochs over the training set and saves one adapter directory per
epoch. Those directories are what :class:`RFTInfEstimator` later loads.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SurrogateTrainConfig:
    base_model: str
    output_dir: str
    learning_rate: float = 1e-6
    num_epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8           # GRPO group size G
    max_prompt_length: int = 512
    max_completion_length: int = 512
    temperature: float = 1.0
    weight_decay: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True
    seed: int = 42
    logging_steps: int = 10
    save_strategy: str = "epoch"        # one ckpt per epoch
    report_to: str = "none"


def _build_reward_callable(
    user_reward_fn: Callable[[Sequence[str], Sequence[str]], List[float]],
):
    """Adapt our (completions, references) reward to the TRL GRPOTrainer
    signature: ``func(prompts, completions, **kwargs) -> list[float]``.

    The ground-truth answer travels in the dataset column `answer` and is
    forwarded to the reward via the ``answer`` kwarg by TRL.
    """

    def _reward(prompts=None, completions=None, **kwargs):
        if completions is None:
            return []
        # GRPO collators may pass completions as list[str] or list[list[dict]]
        if isinstance(completions[0], list):
            texts = [c[0]["content"] if isinstance(c[0], dict) else c[0] for c in completions]
        else:
            texts = list(completions)
        refs = kwargs.get("answer", [""] * len(texts))
        return user_reward_fn(texts, refs)

    return _reward


def run_surrogate_training(
    cfg: SurrogateTrainConfig,
    train_dataset: Dataset,
    reward_fn: Callable[[Sequence[str], Sequence[str]], List[float]],
) -> List[str]:
    """Run E epochs of GRPO + LoRA and return the list of checkpoint dirs."""
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        seed=cfg.seed,
        report_to=cfg.report_to,
        remove_unused_columns=False,
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=cfg.base_model,
        reward_funcs=_build_reward_callable(reward_fn),
        args=grpo_cfg,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    trainer.train()

    ckpts: List[str] = []
    for name in sorted(os.listdir(cfg.output_dir)):
        full = os.path.join(cfg.output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            ckpts.append(full)
    if not ckpts:
        # Fallback: use the final model directory
        final_dir = os.path.join(cfg.output_dir, "final")
        trainer.save_model(final_dir)
        ckpts = [final_dir]
    return ckpts
