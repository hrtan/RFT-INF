"""Per-sample policy-gradient computation for the RFT-Inf estimator.

For a single training sample z = (s, y) and policy pi_theta_t (t-th
checkpoint), the *per-sample* policy gradient used in Eq. (5) is:

    G_z^(t) = (1/G) * sum_{i=1..G}  A_hat(s, a_i) * d/dtheta log pi_theta(a_i|s)

where {a_i} are G rollouts sampled from pi_theta_t and A_hat is the
group-relative advantage of GRPO. For vanilla policy gradient set G=1
and use the (reward - baseline) as advantage.

We don't store gradients as huge per-sample vectors. Instead the gradient
is materialised once, then either:
  - accumulated into a running global gradient buffer, or
  - immediately dotted against a precomputed global gradient.
This keeps memory at O(num_trainable_params) regardless of dataset size.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Flat-gradient helpers
# ---------------------------------------------------------------------------


def _flatten_grads(grads: Sequence[Optional[Tensor]]) -> Tensor:
    """Concatenate a list of (possibly None) gradients into a 1-D tensor."""
    pieces = []
    for g in grads:
        if g is None:
            continue
        pieces.append(g.detach().reshape(-1))
    if not pieces:
        raise RuntimeError("All gradients are None; nothing to flatten.")
    return torch.cat(pieces)


@dataclass
class FlatGradient:
    """Lightweight wrapper around a flat parameter-gradient tensor."""

    tensor: Tensor

    @property
    def numel(self) -> int:
        return self.tensor.numel()

    def to(self, device, dtype=None) -> "FlatGradient":
        t = self.tensor.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return FlatGradient(t)

    def add_(self, other: "FlatGradient") -> "FlatGradient":
        self.tensor.add_(other.tensor)
        return self

    def dot(self, other: "FlatGradient") -> float:
        return torch.dot(
            self.tensor.float(), other.tensor.to(self.tensor.device).float()
        ).item()

    def save(self, path: str) -> None:
        torch.save({"flat_grad": self.tensor.cpu()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FlatGradient":
        obj = torch.load(path, map_location=device)
        return cls(obj["flat_grad"].to(device))

    @classmethod
    def zeros_like(cls, params: Sequence[Tensor], device, dtype=torch.float32):
        n = sum(p.numel() for p in params)
        return cls(torch.zeros(n, device=device, dtype=dtype))


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    prompt_ids: Tensor                # [Lp]
    response_ids: Tensor              # [Lr]
    response_mask: Tensor             # [Lr]  1 if real token else 0 (e.g. pad)
    text: str
    reward: float
    advantage: float = 0.0


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompt: str,
    *,
    group_size: int,
    max_prompt_length: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    extra_kwargs: Optional[dict] = None,
) -> List[Rollout]:
    """Sample `group_size` responses from `model` for `prompt`."""
    inputs = tokenizer(
        [prompt] * group_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)

    gen_kwargs = dict(
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        max_new_tokens=max_response_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    if extra_kwargs:
        gen_kwargs.update(extra_kwargs)

    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    resp_ids = out[:, prompt_len:]
    pad_id = tokenizer.pad_token_id
    response_mask = (resp_ids != pad_id).long()

    rollouts: List[Rollout] = []
    for i in range(group_size):
        text = tokenizer.decode(resp_ids[i], skip_special_tokens=True)
        rollouts.append(
            Rollout(
                prompt_ids=inputs["input_ids"][i].detach(),
                response_ids=resp_ids[i].detach(),
                response_mask=response_mask[i].detach(),
                text=text,
                reward=0.0,
            )
        )
    return rollouts


# ---------------------------------------------------------------------------
# Advantage assignment
# ---------------------------------------------------------------------------


def assign_grpo_advantages(rollouts: List[Rollout], eps: float = 1e-6) -> None:
    """In-place assign GRPO-style group-relative advantages."""
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
    if rewards.numel() == 1:
        adv = rewards - rewards.mean()
    else:
        std = rewards.std(unbiased=False).clamp(min=eps)
        adv = (rewards - rewards.mean()) / std
    for r, a in zip(rollouts, adv.tolist()):
        r.advantage = a


# ---------------------------------------------------------------------------
# Per-sample loss whose gradient equals the policy gradient G_z
# ---------------------------------------------------------------------------


def _policy_loss_for_rollouts(model, rollouts: List[Rollout]) -> Tensor:
    """Build a scalar L such that grad(L, theta) = -G_z (paper's notation).

    We then negate later if needed; for the inner-product <G_z, G_Z> the
    sign is the same on both sides, so we use grad(L) directly throughout.
    """
    losses = []
    for r in rollouts:
        full_ids = torch.cat([r.prompt_ids, r.response_ids], dim=0).unsqueeze(0)
        full_ids = full_ids.to(model.device)
        attn = torch.ones_like(full_ids)
        out = model(input_ids=full_ids, attention_mask=attn)
        logits = out.logits  # [1, T, V]
        # Predict response tokens
        prompt_len = r.prompt_ids.shape[0]
        # logits at position t predict token t+1; response tokens occupy
        # positions [prompt_len, prompt_len + Lr). We need logits at indices
        # [prompt_len-1, prompt_len + Lr - 1) to predict them.
        Lr = r.response_ids.shape[0]
        pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + Lr, :]
        targets = r.response_ids.to(model.device).unsqueeze(0)
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, Lr]
        mask = r.response_mask.to(model.device).float().unsqueeze(0)
        seq_logp = (token_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        # Loss = -A * mean_token logp  (mean within sequence, sum across group)
        losses.append(-r.advantage * seq_logp.squeeze(0))
    return torch.stack(losses).mean()  # divide by G as in paper


# ---------------------------------------------------------------------------
# Public: compute per-sample flat gradient
# ---------------------------------------------------------------------------


@dataclass
class SampleGradComputer:
    """Computes per-sample policy gradients for the RFT-Inf estimator.

    Typical usage::

        comp = SampleGradComputer(model, tokenizer, params, reward_fn,
                                  group_size=8, ...)
        grad = comp.compute(prompt, reference)
        # grad is a FlatGradient living on `model.device`.
    """

    model: torch.nn.Module
    tokenizer: object
    params: List[Tensor]
    reward_fn: Callable[[Sequence[str], Sequence[str]], List[float]]
    group_size: int
    max_prompt_length: int
    max_response_length: int
    temperature: float
    top_p: float
    top_k: int
    advantage_eps: float
    extra_generation_kwargs: Optional[dict] = None

    def _compute_rollouts(self, prompt: str, reference: str) -> List[Rollout]:
        rollouts = generate_rollouts(
            self.model,
            self.tokenizer,
            prompt,
            group_size=self.group_size,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            extra_kwargs=self.extra_generation_kwargs,
        )
        rewards = self.reward_fn([r.text for r in rollouts], [reference] * len(rollouts))
        for r, rv in zip(rollouts, rewards):
            r.reward = float(rv)
        assign_grpo_advantages(rollouts, eps=self.advantage_eps)
        return rollouts

    def compute(
        self,
        prompt: str,
        reference: str,
        *,
        return_rollouts: bool = False,
    ) -> Tuple[FlatGradient, Optional[List[Rollout]]]:
        """Compute G_z = grad of policy loss for the given sample."""
        rollouts = self._compute_rollouts(prompt, reference)
        # If every rollout has zero advantage (e.g. all-correct or all-wrong)
        # the gradient is exactly zero and we can skip the backward pass.
        if all(abs(r.advantage) < 1e-12 for r in rollouts):
            zero = FlatGradient.zeros_like(
                self.params, device=self.model.device, dtype=torch.float32
            )
            return (zero, rollouts if return_rollouts else None)

        for p in self.params:
            if p.grad is not None:
                p.grad = None
        self.model.zero_grad(set_to_none=True)

        loss = _policy_loss_for_rollouts(self.model, rollouts)
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        flat = _flatten_grads(grads).to(torch.float32)
        return (FlatGradient(flat), rollouts if return_rollouts else None)


# ---------------------------------------------------------------------------
# Helper: temporarily put model in eval-but-grad mode
# ---------------------------------------------------------------------------


@contextmanager
def grad_enabled_eval(model):
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        yield
    if was_training:
        model.train()
