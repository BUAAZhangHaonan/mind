"""Lightweight dtype parsing helpers."""

from __future__ import annotations

import torch


def resolve_torch_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {value}") from exc
