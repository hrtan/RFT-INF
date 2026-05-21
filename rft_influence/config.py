"""Configuration for the RFT-Inf estimator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union


@dataclass
class InfluenceConfig:
    """Hyper-parameters for computing the RFT-Inf score.

    Attributes
    ----------
    base_model:
        HuggingFace name or local path of the base policy model.
    checkpoint_paths:
        A list of paths, each pointing to a checkpoint produced by the
        surrogate RFT run. Each path may contain either a full model state
        dict or a PEFT/LoRA adapter directory. Order matches `learning_rates`.
    learning_rates:
        Learning-rate eta_t used to produce each checkpoint. Either a single
        float (broadcast to all checkpoints) or one per checkpoint.
    dataset_size:
        N in Eq. (5). The size of the surrogate training set. Required to
        scale the score correctly.
    group_size:
        GRPO group size G. The estimator generates this many rollouts per
        prompt to mimic GRPO's group-relative advantage. Set to 1 for
        non-grouped RFT (vanilla policy gradient / PPO).
    max_prompt_length:
        Tokenizer truncation budget for the prompt.
    max_response_length:
        Maximum number of new tokens generated per rollout.
    temperature, top_p, top_k:
        Sampling hyper-parameters for rollout generation.
    advantage_eps:
        Numerical floor for std() in GRPO advantage normalisation.
    use_lora_only_grads:
        If True, gradients are computed only over trainable (LoRA) params.
        Strongly recommended for efficiency on large models.
    final_layer_only:
        If True (and `use_lora_only_grads=False`) only the LM head's
        gradients are used (cf. Section 4.3 of the paper).
    dtype:
        Compute dtype for the policy model.
    device:
        Torch device.
    seed:
        Random seed for reproducibility of rollouts.
    """

    base_model: str
    checkpoint_paths: Sequence[str]
    learning_rates: Union[float, Sequence[float]] = 1e-6
    dataset_size: int = 0  # required, validated below
    group_size: int = 8

    max_prompt_length: int = 512
    max_response_length: int = 512

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    advantage_eps: float = 1e-6

    use_lora_only_grads: bool = True
    final_layer_only: bool = False
    is_lora_checkpoint: bool = True

    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 42

    extra_generation_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset_size <= 0:
            raise ValueError(
                "dataset_size (N) must be set to a positive integer."
            )
        if isinstance(self.learning_rates, (int, float)):
            self.learning_rates = [float(self.learning_rates)] * len(
                self.checkpoint_paths
            )
        if len(self.learning_rates) != len(self.checkpoint_paths):
            raise ValueError(
                "len(learning_rates) must equal len(checkpoint_paths)."
            )

    @property
    def num_checkpoints(self) -> int:
        return len(self.checkpoint_paths)
