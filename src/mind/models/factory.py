"""Model wrapper factory."""

from __future__ import annotations

from mind.config import ModelConfig

from .wrappers import (
    InternVLWrapper,
    LlavaOnevisionWrapper,
    MolmoWrapper,
    Qwen25VLWrapper,
    Qwen35VLWrapper,
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
    if family in {"qwen2_5_vl", "qwen2.5_vl"}:
        return Qwen25VLWrapper(config)
    if family in {"qwen3_5", "qwen3.5"}:
        return Qwen35VLWrapper(config)
    if family in {"qwen_vl", "qwen-vl", "qwen3_vl"}:
        return QwenVLWrapper(config)
    if family in {"qwen", "qwen_text"}:
        return QwenTextWrapper(config)
    raise ValueError(f"No model wrapper is implemented for family={config.family!r}")
