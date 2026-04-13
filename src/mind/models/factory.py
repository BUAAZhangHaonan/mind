"""Model wrapper factory."""

from __future__ import annotations

from mind.config import ModelConfig

from .wrappers import (
    InternVLWrapper,
    LlavaOnevisionWrapper,
    MolmoWrapper,
    QwenTextWrapper,
    QwenVLWrapper,
)


def create_model_wrapper(config: ModelConfig):
    family = config.family.lower()
    if family == "internvl":
        return InternVLWrapper(config)
    if family in {"llava_onevision", "llava-onevision"}:
        return LlavaOnevisionWrapper(config)
    if family == "molmo":
        return MolmoWrapper(config)
    if family in {"qwen_vl", "qwen-vl"}:
        return QwenVLWrapper(config)
    return QwenTextWrapper(config)
