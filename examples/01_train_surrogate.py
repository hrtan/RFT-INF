"""Step 1 — Surrogate RFT training (Algorithm 1, line 1).

Runs a small number of GRPO + LoRA epochs over the chosen training set
(GSM8K by default) and saves one LoRA adapter per epoch under
``--output_dir``. Those checkpoint directories are the input to step 2.

Example
-------
    python examples/01_train_surrogate.py \\
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --output_dir runs/surrogate_qwen15b \\
        --train_size 4096 \\
        --num_epochs 2
"""
from __future__ import annotations

import argparse
import json
import os

from rft_influence.data import load_gsm8k, to_hf_dataset
from rft_influence.rewards import gsm8k_reward
from rft_influence.surrogate import SurrogateTrainConfig, run_surrogate_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_size", type=int, default=4096)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    samples = load_gsm8k(split="train", limit=args.train_size, seed=args.seed)
    ds = to_hf_dataset(samples)

    cfg = SurrogateTrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        lora_rank=args.lora_rank,
        seed=args.seed,
    )

    ckpts = run_surrogate_training(cfg, ds, reward_fn=gsm8k_reward)
    meta_path = os.path.join(args.output_dir, "rftinf_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "checkpoint_paths": ckpts,
                "learning_rate": args.learning_rate,
                "dataset_size": len(samples),
                "num_generations": args.num_generations,
                "sample_ids": [s.sample_id for s in samples],
            },
            f,
            indent=2,
        )
    print(f"Saved {len(ckpts)} checkpoint(s):")
    for c in ckpts:
        print(f"  - {c}")
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
