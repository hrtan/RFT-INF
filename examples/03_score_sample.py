"""Step 3 — Compute the RFT-Inf score for a *single* training sample.

Given the cached global gradients (Step 2) this script is fast: one
forward + backward per checkpoint.

Two ways to specify the sample:
  * ``--sample_id gsm8k-train-7`` — pick by id from the surrogate dataset.
  * ``--prompt_file q.txt --answer_file a.txt`` — supply a custom sample.

Example
-------
    python examples/03_score_sample.py \\
        --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --surrogate_dir runs/surrogate_qwen15b \\
        --cache_dir runs/global_grads_qwen15b \\
        --sample_id gsm8k-train-7
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rft_influence import InfluenceConfig, RFTInfEstimator, gsm8k_reward
from rft_influence.data import RFTSample, load_gsm8k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--surrogate_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--sample_id", default=None)
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--answer_file", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--group_size", type=int, default=8)
    return p.parse_args()


def _resolve_sample(args, meta) -> RFTSample:
    if args.prompt_file and args.answer_file:
        return RFTSample(
            prompt=Path(args.prompt_file).read_text(),
            answer=Path(args.answer_file).read_text(),
            sample_id="custom-sample",
        )
    if args.sample_id is None:
        raise ValueError(
            "Provide either --sample_id or both --prompt_file/--answer_file."
        )
    samples = load_gsm8k(
        split="train", limit=meta["dataset_size"], seed=meta.get("seed", 42),
    )
    for s in samples:
        if s.sample_id == args.sample_id:
            return s
    raise ValueError(f"Sample id {args.sample_id} not found in surrogate set.")


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

    sample = _resolve_sample(args, meta)
    out = estimator.score_sample(sample, return_components=True)

    print("============= RFT-Inf score =============")
    print(f"Sample id : {out['sample_id']}")
    print(f"Prompt    : {sample.prompt[:160]}{'...' if len(sample.prompt) > 160 else ''}")
    print(f"Reference : {sample.answer.strip().splitlines()[-1]}")
    print(f"D_hat(z)  : {out['score']:+.6e}")
    print()
    print("Per-checkpoint contributions:")
    for t, (eta, dot) in enumerate(out["per_checkpoint"]):
        contrib = 2.0 * eta * dot / cfg.dataset_size
        print(f"  ckpt {t}: eta={eta:.2e}  <g_z, G_Z>={dot:+.4e}  "
              f"contrib={contrib:+.4e}")


if __name__ == "__main__":
    main()
