"""Model wrapper factory."""

from __future__ import annotations

from mind.config import ModelConfig

from .wrappers import InternVLWrapper, QwenTextWrapper, QwenVLWrapper


def create_model_wrapper(config: ModelConfig):
    family = config.family.lower()
    if family == "internvl":
        return InternVLWrapper(config)
    if family in {"qwen_vl", "qwen-vl"} or "vl" in config.model_id.lower():
        return QwenVLWrapper(config)
    return QwenTextWrapper(config)
