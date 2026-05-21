"""Helpers to load surrogate-RFT checkpoints into a policy model.

Two checkpoint flavours are supported:

* PEFT/LoRA adapter directories (recommended; produced by the surrogate
  trainer in this package).
* Full HuggingFace transformers state dirs.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unsupported dtype `{name}`")
    return _DTYPES[name]


def load_tokenizer(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_model(
    base_model: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=resolve_dtype(dtype),
        trust_remote_code=True,
    )
    model.to(device)
    return model


def attach_lora_adapter(model, adapter_path: str, adapter_name: str = "default"):
    """Load a LoRA adapter from `adapter_path` onto `model`.

    On first call wraps the base model with :class:`peft.PeftModel`; on
    subsequent calls reloads the weights of the same named adapter so the
    PEFT module graph is reused (no growth in memory).
    """
    from peft import PeftModel

    if not hasattr(model, "peft_config"):
        return PeftModel.from_pretrained(
            model, adapter_path, adapter_name=adapter_name, is_trainable=True
        )

    # Already a PeftModel - swap weights of the same adapter slot.
    if adapter_name in model.peft_config:
        try:
            model.delete_adapter(adapter_name)
        except Exception:
            pass
    try:
        model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=True)
    except TypeError:
        # Older PEFT versions don't accept `is_trainable` here.
        model.load_adapter(adapter_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    return model


def load_full_state_dict(model, ckpt_path: str):
    """Replace the model weights with those at `ckpt_path` (a HF dir)."""
    if os.path.isdir(ckpt_path):
        ckpt_model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=next(model.parameters()).dtype,
            trust_remote_code=True,
        )
        model.load_state_dict(ckpt_model.state_dict())
        del ckpt_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return model

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model


def freeze_for_grad_collection(model, use_lora_only: bool, final_layer_only: bool):
    """Configure `requires_grad` so that only the wanted parameters get grads.

    Returns the list of parameters whose gradients will be collected.
    """
    if use_lora_only:
        for name, p in model.named_parameters():
            p.requires_grad = "lora_" in name
    elif final_layer_only:
        for name, p in model.named_parameters():
            p.requires_grad = ("lm_head" in name) or ("embed_out" in name)
    else:
        for p in model.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "No parameters require grad. Check `use_lora_only_grads` and the "
            "checkpoint type (LoRA adapter vs full weights)."
        )
    return params
