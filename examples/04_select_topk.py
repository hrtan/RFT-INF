"""Step 4 — Top-delta data selection using RFT-Inf scores.

Computes D_hat(z) for every sample in the surrogate set and writes a
JSONL file with the scores. The top-delta% can then be used as the
curated training set for the formal RFT run (Algorithm 1, line 5).

Example
-------
    python examples/04_select_topk.py \\
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --surrogate_dir runs/surrogate_qwen15b \\
        --cache_dir runs/global_grads_qwen15b \\
        --output_file runs/scores.jsonl \\
        --selection_ratio 0.2
"""
from __future__ import annotations

import argparse
import json
import os

from rft_influence import InfluenceConfig, RFTInfEstimator, gsm8k_reward
from rft_influence.data import load_gsm8k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--surrogate_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--selection_ratio", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--group_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.surrogate_dir, "rftinf_meta.json")) as f:
        meta = json.load(f)

    cfg = InfluenceConfig(
        base_model=args.base_model,
        checkpoint_paths=meta["checkpoint_paths"],
        learning_rates=meta.get("learning_rate", 1e-6),
        dataset_size=meta["dataset_size"],
        group_size=args.group_size,
        device=args.device,
        dtype=args.dtype,
        is_lora_checkpoint=True,
    )
    estimator = RFTInfEstimator(cfg, reward_fn=gsm8k_reward)
    estimator.load_global_gradients(args.cache_dir)

    samples = load_gsm8k(
        split="train", limit=meta["dataset_size"], seed=meta.get("seed", 42),
    )
    scores = estimator.score_dataset(samples)

    with open(args.output_file, "w") as f:
        for s, sc in zip(samples, scores):
            f.write(json.dumps(
                {"sample_id": s.sample_id, "score": sc, "prompt": s.prompt[:200]}
            ) + "\n")

    ranked = sorted(zip(samples, scores), key=lambda x: x[1], reverse=True)
    keep = int(len(ranked) * args.selection_ratio)
    selected = [s.sample_id for s, _ in ranked[:keep]]
    sel_path = os.path.splitext(args.output_file)[0] + f".top{keep}.json"
    with open(sel_path, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Wrote {len(scores)} scores to {args.output_file}")
    print(f"Top-{args.selection_ratio:.0%} ({keep} ids) saved to {sel_path}")
    print("\nTop-5 most influential samples:")
    for s, sc in ranked[:5]:
        print(f"  {sc:+.4e}  {s.sample_id}  | {s.prompt[:80]}...")


if __name__ == "__main__":
    main()
