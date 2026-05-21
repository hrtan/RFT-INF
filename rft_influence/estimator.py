"""High-level orchestrator for the RFT-Inf data-influence estimator.

This module turns the per-sample gradient utilities into a single
ergonomic class. The public surface is intentionally small:

    estimator = RFTInfEstimator(config, reward_fn=...)
    estimator.compute_global_gradients(dataset, save_dir)   # offline
    estimator.load_global_gradients(save_dir)               # cached
    score = estimator.score_sample(prompt, reference)
    scores = estimator.score_dataset(dataset)

Internally each call iterates over the saved checkpoints, switches the
policy weights to the corresponding state, and runs the same gradient
computation pass.
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from tqdm.auto import tqdm

from .config import InfluenceConfig
from .checkpoints import (
    attach_lora_adapter,
    freeze_for_grad_collection,
    load_base_model,
    load_full_state_dict,
    load_tokenizer,
)
from .gradients import (
    FlatGradient,
    SampleGradComputer,
    grad_enabled_eval,
)
from .data import RFTSample

logger = logging.getLogger(__name__)


SampleLike = Union[RFTSample, Tuple[str, str], Dict[str, str]]


def _coerce_sample(s: SampleLike) -> RFTSample:
    if isinstance(s, RFTSample):
        return s
    if isinstance(s, tuple) and len(s) == 2:
        return RFTSample(prompt=s[0], answer=s[1])
    if isinstance(s, dict):
        return RFTSample(
            prompt=s["prompt"],
            answer=s.get("answer") or s.get("reference") or s.get("ground_truth"),
            sample_id=s.get("sample_id"),
        )
    raise TypeError(f"Unsupported sample type: {type(s)}")


class RFTInfEstimator:
    """RFT-Inf data influence estimator (Eq. 5 of the paper).

    Parameters
    ----------
    config:
        :class:`InfluenceConfig` describing model, checkpoints and rollout
        hyper-parameters.
    reward_fn:
        Callable ``reward_fn(completions, references) -> list[float]``.
        ``completions`` are decoded model outputs, ``references`` are
        ground-truth answers. The reward should follow the paper's
        convention (e.g. +1 / -1).
    """

    def __init__(
        self,
        config: InfluenceConfig,
        reward_fn: Callable[[Sequence[str], Sequence[str]], List[float]],
    ):
        self.config = config
        self.reward_fn = reward_fn

        torch.manual_seed(config.seed)

        self._tokenizer = load_tokenizer(config.base_model)
        self._model = load_base_model(
            config.base_model, dtype=config.dtype, device=config.device
        )
        self._global_gradients: Dict[int, FlatGradient] = {}

    # ------------------------------------------------------------------
    # Checkpoint plumbing
    # ------------------------------------------------------------------

    def _activate_checkpoint(self, ckpt_idx: int) -> None:
        """Switch in-memory model to the i-th checkpoint state."""
        path = self.config.checkpoint_paths[ckpt_idx]
        if self.config.is_lora_checkpoint:
            self._model = attach_lora_adapter(
                self._model, path, adapter_name="rftinf_active"
            )
        else:
            load_full_state_dict(self._model, path)
        self._params = freeze_for_grad_collection(
            self._model,
            use_lora_only=self.config.use_lora_only_grads,
            final_layer_only=self.config.final_layer_only,
        )

    def _make_grad_computer(self) -> SampleGradComputer:
        return SampleGradComputer(
            model=self._model,
            tokenizer=self._tokenizer,
            params=self._params,
            reward_fn=self.reward_fn,
            group_size=self.config.group_size,
            max_prompt_length=self.config.max_prompt_length,
            max_response_length=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            advantage_eps=self.config.advantage_eps,
            extra_generation_kwargs=self.config.extra_generation_kwargs,
        )

    # ------------------------------------------------------------------
    # Stage 1: cumulative G_Z^(t) over the surrogate training set
    # ------------------------------------------------------------------

    def compute_global_gradients(
        self,
        dataset: Iterable[SampleLike],
        save_dir: Optional[str] = None,
        progress: bool = True,
    ) -> Dict[int, FlatGradient]:
        """Compute and cache G_Z^(t) for every checkpoint t.

        This is an O(N * C) operation done once. Results are kept in memory
        and optionally serialised under ``save_dir/global_grad_<t>.pt``.
        """
        samples = [_coerce_sample(s) for s in dataset]
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(
                    {**asdict(self.config), "num_samples": len(samples)},
                    f,
                    indent=2,
                    default=str,
                )

        for t in range(self.config.num_checkpoints):
            self._activate_checkpoint(t)
            computer = self._make_grad_computer()

            global_grad: Optional[FlatGradient] = None
            iterator = samples
            if progress:
                iterator = tqdm(
                    samples,
                    desc=f"[G_Z] checkpoint {t + 1}/{self.config.num_checkpoints}",
                )

            with grad_enabled_eval(self._model):
                for s in iterator:
                    grad, _ = computer.compute(s.prompt, s.answer)
                    if global_grad is None:
                        global_grad = FlatGradient(
                            torch.zeros_like(grad.tensor)
                        )
                    global_grad.add_(grad)

            assert global_grad is not None
            self._global_gradients[t] = global_grad
            if save_dir is not None:
                global_grad.save(os.path.join(save_dir, f"global_grad_{t}.pt"))

        return self._global_gradients

    def load_global_gradients(self, save_dir: str) -> Dict[int, FlatGradient]:
        """Load previously cached G_Z^(t) tensors from ``save_dir``."""
        loaded: Dict[int, FlatGradient] = {}
        for t in range(self.config.num_checkpoints):
            path = os.path.join(save_dir, f"global_grad_{t}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing cached global gradient at {path}. "
                    f"Run `compute_global_gradients` first."
                )
            loaded[t] = FlatGradient.load(path, device=self.config.device)
        self._global_gradients = loaded
        return loaded

    # ------------------------------------------------------------------
    # Stage 2: per-sample influence score
    # ------------------------------------------------------------------

    def score_sample(
        self,
        sample: SampleLike,
        *,
        return_components: bool = False,
    ) -> Union[float, Dict[str, object]]:
        """Compute D_hat(z) for a single sample (Eq. 5)."""
        if not self._global_gradients:
            raise RuntimeError(
                "Global gradients are not available. Call "
                "`compute_global_gradients` or `load_global_gradients` first."
            )
        s = _coerce_sample(sample)
        per_ckpt: List[Tuple[float, float]] = []  # (eta, dot)
        score = 0.0

        for t in range(self.config.num_checkpoints):
            self._activate_checkpoint(t)
            computer = self._make_grad_computer()
            G_Z = self._global_gradients[t]

            with grad_enabled_eval(self._model):
                grad, _ = computer.compute(s.prompt, s.answer)
            dot = grad.dot(G_Z)
            eta_t = float(self.config.learning_rates[t])
            per_ckpt.append((eta_t, dot))
            score += 2.0 * eta_t * dot / self.config.dataset_size

        if return_components:
            return {
                "score": score,
                "per_checkpoint": per_ckpt,
                "sample_id": s.sample_id,
            }
        return score

    def score_dataset(
        self,
        dataset: Iterable[SampleLike],
        *,
        progress: bool = True,
    ) -> List[float]:
        """Score every sample in `dataset`. Returns a list of D_hat values.

        The implementation iterates checkpoint-by-checkpoint to amortise
        checkpoint loading: O(C) checkpoint switches instead of O(N*C).
        """
        if not self._global_gradients:
            raise RuntimeError(
                "Global gradients are not available. Call "
                "`compute_global_gradients` or `load_global_gradients` first."
            )
        samples = [_coerce_sample(s) for s in dataset]
        scores = [0.0 for _ in samples]

        for t in range(self.config.num_checkpoints):
            self._activate_checkpoint(t)
            computer = self._make_grad_computer()
            G_Z = self._global_gradients[t]
            eta_t = float(self.config.learning_rates[t])
            iterator = list(enumerate(samples))
            if progress:
                iterator = tqdm(
                    iterator,
                    desc=f"[score] ckpt {t + 1}/{self.config.num_checkpoints}",
                )
            with grad_enabled_eval(self._model):
                for i, s in iterator:
                    grad, _ = computer.compute(s.prompt, s.answer)
                    dot = grad.dot(G_Z)
                    scores[i] += 2.0 * eta_t * dot / self.config.dataset_size
        return scores

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
