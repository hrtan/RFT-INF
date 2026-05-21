"""Quickstart — score one sample using *your own* RFT checkpoints.

Use this example when you already have a trained surrogate (any RFT run
that saved per-epoch LoRA adapters or full HF state dirs) and want to
compute the RFT-Inf score for a particular training example.

Replace the four user-supplied variables below and run.
"""
from __future__ import annotations

from rft_influence import InfluenceConfig, RFTInfEstimator, gsm8k_reward
from rft_influence.data import RFTSample, load_gsm8k


# ----------------------------- USER INPUTS -----------------------------
BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"

# One LoRA adapter directory per epoch, in chronological order.
CHECKPOINTS = [
    "runs/surrogate_qwen15b/checkpoint-256",   # epoch 1
    "runs/surrogate_qwen15b/checkpoint-512",   # epoch 2
]

# eta_t used to produce each checkpoint
LEARNING_RATES = [1e-6, 1e-6]

# N — size of the surrogate training set (denominator in Eq. 5)
DATASET_SIZE = 4096

# The sample whose influence you want to compute
QUERY_SAMPLE = RFTSample(
    prompt="Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast "
           "and bakes muffins with 4. She sells the rest at $2 each. How much does "
           "she make per day?\n\nSolution:",
    answer="9 eggs sold * $2 = $18 per day. #### 18",
    sample_id="janet-ducks",
)
# ----------------------------------------------------------------------


def main():
    # The training set used for surrogate RFT — needed only to *cache*
    # the global gradient G_Z^(t). After caching, querying any new sample
    # is a constant-time call.
    surrogate_set = load_gsm8k(split="train", limit=DATASET_SIZE)

    cfg = InfluenceConfig(
        base_model=BASE_MODEL,
        checkpoint_paths=CHECKPOINTS,
        learning_rates=LEARNING_RATES,
        dataset_size=DATASET_SIZE,
        group_size=8,
        is_lora_checkpoint=True,
        use_lora_only_grads=True,
    )

    estimator = RFTInfEstimator(cfg, reward_fn=gsm8k_reward)

    # Step A — cache G_Z^(t) once. Skip if already cached on disk.
    cache_dir = "runs/_global_grads"
    try:
        estimator.load_global_gradients(cache_dir)
        print(f"Loaded cached global gradients from {cache_dir}")
    except FileNotFoundError:
        print("Computing global gradients (one-off cost)...")
        estimator.compute_global_gradients(surrogate_set, save_dir=cache_dir)

    # Step B — score the query sample.
    out = estimator.score_sample(QUERY_SAMPLE, return_components=True)

    print(f"\nRFT-Inf score for `{out['sample_id']}`: {out['score']:+.6e}")
    for t, (eta, dot) in enumerate(out["per_checkpoint"]):
        print(f"  ckpt {t}: eta={eta:.1e}  <g_z, G_Z>={dot:+.4e}")


if __name__ == "__main__":
    main()
