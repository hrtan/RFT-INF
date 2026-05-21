"""Step 2 — Cache the cumulative gradient G_Z^(t) for every checkpoint.

This is an O(N * C) one-off pass; afterwards any individual sample can
be scored with a single forward+backward per checkpoint (Step 3).

Example
-------
    python examples/02_compute_global_gradient.py \\
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --surrogate_dir runs/surrogate_qwen15b \\
        --cache_dir runs/global_grads_qwen15b
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from rft_influence import InfluenceConfig, RFTInfEstimator, gsm8k_reward
from rft_influence.data import load_gsm8k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--surrogate_dir", required=True,
                   help="Directory produced by 01_train_surrogate.py")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_response_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--final_layer_only", action="store_true")
    p.add_argument(
        "--full_param_grads", action="store_true",
        help="Use grads of *all* params instead of LoRA-only.",
    )
    return p.parse_args()


def _discover_checkpoints(meta: dict, surrogate_dir: str) -> List[str]:
    if "checkpoint_paths" in meta and meta["checkpoint_paths"]:
        return meta["checkpoint_paths"]
    return sorted(
        os.path.join(surrogate_dir, n)
        for n in os.listdir(surrogate_dir)
        if n.startswith("checkpoint-")
    )


def main():
    args = parse_args()
    meta_path = os.path.join(args.surrogate_dir, "rftinf_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing surrogate metadata at {meta_path}. "
            f"Did you run examples/01_train_surrogate.py?"
        )
    with open(meta_path) as f:
        meta = json.load(f)

    ckpts = _discover_checkpoints(meta, args.surrogate_dir)
    print(f"Discovered {len(ckpts)} checkpoint(s):")
    for c in ckpts:
        print(f"  - {c}")

    cfg = InfluenceConfig(
        base_model=args.base_model,
        checkpoint_paths=ckpts,
        learning_rates=meta.get("learning_rate", 1e-6),
        dataset_size=meta["dataset_size"],
        group_size=args.group_size,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        use_lora_only_grads=not args.full_param_grads,
        final_layer_only=args.final_layer_only,
        is_lora_checkpoint=True,
        dtype=args.dtype,
        device=args.device,
    )

    samples = load_gsm8k(
        split="train", limit=meta["dataset_size"], seed=meta.get("seed", 42),
    )

    estimator = RFTInfEstimator(cfg, reward_fn=gsm8k_reward)
    estimator.compute_global_gradients(samples, save_dir=args.cache_dir)
    print(f"Cached global gradients into {args.cache_dir}")


if __name__ == "__main__":
    main()
