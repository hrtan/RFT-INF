# RFT-Inf — Data Influence Estimator for Reinforcement Fine-Tuning

A lightweight, well-packaged implementation of the data-influence estimator proposed in

> Tan et al., **Understanding Data Influence in Reinforcement Finetuning**, NeurIPS 2025.

The estimator quantifies how much each training example contributes to the final reward of a Reinforcement Fine-Tuning (RFT) run. It is a first-order gradient-based approximation that requires only one short surrogate-training pass plus a few cheap forward/backward passes per query.

## Algorithm in one line

For each sample $z$, the RFT-Inf score (Eq. 5 of the paper) is

$$
\hat{D}(z) \;=\; \sum_{t}\frac{2\,\eta_t}{N}\;\bigl\langle G_z^{(t)}, \; G_Z^{(t)}\bigr\rangle,
$$

where

- $G_z^{(t)}$ is the per-sample policy gradient at checkpoint $t$ (GRPO group form: $\sum_i \hat{A}(s,a_i)\,\nabla\!\log\pi_{\theta_t}(a_i|s)$);
- $G_Z^{(t)} = \sum_{z' \in Z} G_{z'}^{(t)}$ is the cumulative gradient over the surrogate training set;
- $\eta_t$ is the surrogate learning rate, $N$ the surrogate set size.

A higher score means the sample's gradient direction is more aligned with the global optimisation trajectory, i.e. it is more representative and beneficial.

## Project layout

```
rft_influence/
├── config.py        InfluenceConfig dataclass
├── data.py          GSM8K loader / RFTSample
├── rewards.py       Binary correctness reward (paper convention)
├── checkpoints.py   LoRA + full-state-dict loading helpers
├── gradients.py     Per-sample policy-gradient computer
├── estimator.py     RFTInfEstimator: the public class
└── surrogate.py     TRL GRPO + LoRA surrogate-training driver

examples/
├── 01_train_surrogate.py            Algorithm 1 line 1
├── 02_compute_global_gradient.py    Cache G_Z^(t) once
├── 03_score_sample.py               Score a single sample
├── 04_select_topk.py                Top-δ data selection
└── quickstart_existing_checkpoints.py    Plug-and-play API for your own ckpts
```

## Installation

```bash
pip install -r requirements.txt
```

The estimator is pure PyTorch and uses [TRL](https://github.com/huggingface/trl) only inside `rft_influence/surrogate.py` for the optional surrogate trainer; you can skip TRL entirely if you bring your own checkpoints.

## End-to-end example with GSM8K

The full pipeline takes three commands. We use `Qwen/Qwen2.5-Math-1.5B-Instruct` as the base model and a 4 K subset of GSM8K's train split.

### 1. Surrogate RFT training (Algorithm 1, line 1)

```bash
python examples/01_train_surrogate.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --output_dir runs/surrogate_qwen15b \
    --train_size 4096 \
    --num_epochs 2 \
    --lora_rank 16
```

This runs 2 epochs of GRPO+LoRA. With `save_strategy=epoch`, TRL writes one `checkpoint-*` directory per epoch, plus a `rftinf_meta.json` describing the run.

### 2. Cache the global gradient G_Z^(t)

```bash
python examples/02_compute_global_gradient.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --surrogate_dir runs/surrogate_qwen15b \
    --cache_dir runs/global_grads_qwen15b
```

This is the one expensive pass (`O(N · C)` rollouts, where `C` is the number of checkpoints). Afterwards each per-sample query is cheap.

### 3. Score a sample (the user's main ask)

```bash
python examples/03_score_sample.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --surrogate_dir runs/surrogate_qwen15b \
    --cache_dir runs/global_grads_qwen15b \
    --sample_id gsm8k-train-7
```

Output (truncated):

```
============= RFT-Inf score =============
Sample id : gsm8k-train-7
Prompt    : You are a careful math reasoner. Solve the problem step by step ...
Reference : #### 18
D_hat(z)  : +3.214821e-04

Per-checkpoint contributions:
  ckpt 0: eta=1.00e-06  <g_z, G_Z>=+5.21e+02  contrib=+2.55e-04
  ckpt 1: eta=1.00e-06  <g_z, G_Z>=+1.36e+02  contrib=+6.66e-05
```

You can also pass `--prompt_file` and `--answer_file` to score an arbitrary sample.

### 4. (Optional) Top-δ selection for the formal RFT run

```bash
python examples/04_select_topk.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --surrogate_dir runs/surrogate_qwen15b \
    --cache_dir runs/global_grads_qwen15b \
    --output_file runs/scores.jsonl \
    --selection_ratio 0.2
```

## Plugging your own RFT checkpoints

If you already trained a surrogate RFT run with any framework (TRL, [VERL](https://github.com/volcengine/verl), OpenRLHF, ...), you just need:

* the base model id/path,
* one LoRA adapter (or full-weights) directory per checkpoint,
* the learning rate(s) used.

```python
from rft_influence import InfluenceConfig, RFTInfEstimator, gsm8k_reward
from rft_influence.data import RFTSample, load_gsm8k

cfg = InfluenceConfig(
    base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    checkpoint_paths=[
        "runs/surrogate/checkpoint-256",   # epoch 1
        "runs/surrogate/checkpoint-512",   # epoch 2
    ],
    learning_rates=[1e-6, 1e-6],
    dataset_size=4096,
    group_size=8,
    is_lora_checkpoint=True,           # flip to False for full-weight ckpts
    use_lora_only_grads=True,
)
estimator = RFTInfEstimator(cfg, reward_fn=gsm8k_reward)

# One-off: cache G_Z for each checkpoint.
estimator.compute_global_gradients(
    load_gsm8k(split="train", limit=4096), save_dir="runs/_global_grads",
)

# Query: score any sample.
score = estimator.score_sample(RFTSample(prompt="...", answer="..."))
print(f"D_hat(z) = {score:+.4e}")
```

## Implementation notes

* **Linear complexity.** Per Section 4.3 of the paper, the estimator is `O(N · E + N · C)`, with E surrogate epochs and C cached checkpoints (the paper uses `C = E = 2`).
* **Gradient footprint.** With `use_lora_only_grads=True` the gradient lives entirely in the LoRA params (~few MB for rank 16), so caching `G_Z^(t)` on disk is trivial.
* **Backtracking through optimisation.** Eq. (10) of the paper expresses the per-step difference $\Delta D_t$ as $\frac{2\eta_t}{N}\langle \nabla J(\theta_{t-1}), g_z^{t-1}\rangle$. We materialise both terms exactly: `G_Z` is the gradient of the policy loss summed over the dataset, `g_z` is the same loss restricted to the queried sample. Both use the GRPO group form.
* **Sign convention.** Loss = $-\,A\,\log\pi$, so `autograd.grad` produces $-G$. The minus signs cancel in the inner product so we apply gradients directly without flipping.

## Reward function

The default `gsm8k_reward` follows the paper's binary scheme:

* `+1` if the model's extracted answer matches the gold answer;
* `-1` otherwise.

It accepts `#### 42`, `\boxed{42}` and trailing-number formats.

## Citation

```bibtex
@inproceedings{tan2025rftinf,
  title     = {Understanding Data Influence in Reinforcement Finetuning},
  author    = {Tan, Haoru and Wu, Xiuzhe and Wu, Sitong and Zhang, Shaofeng and Chen, Yanfeng and Sun, Xingwu and Shen, Jeanne and Qi, Xiaojuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
