"""Shared model wrapper helpers."""

from __future__ import annotations

import re

import torch


_YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


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


def parse_yes_no_answer(text: str) -> int | None:
    match = _YES_NO_PATTERN.search(text)
    if match is None:
        return None
    return 1 if match.group(1).lower() == "yes" else 0
