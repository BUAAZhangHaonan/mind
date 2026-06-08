"""Local model asset registry for Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
import yaml


REQUIRED_MODEL_ALIASES: tuple[str, ...] = (
    "llava-onevision-qwen2-7b-ov-hf",
    "glm-4.6v-flash",
    "qwen3-vl-8b",
    "internvl3.5-8b",
    "minicpm-v-2_6",
    "gemma-3-12b-it",
    "gemma-4-12b-it",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "phi-4-multimodal-instruct",
    "phi-3.5-vision-instruct",
    "gemma-3-4b-it",
    "molmo-7b-d-0924",
    "minicpm-v-4_5",
    "llava-v1.5-7b",
    "qwen2.5-vl-7b",
)


TruthValue = Literal[True, False, "unknown"]


class DeterministicGenerationConfig(BaseModel):
    do_sample: bool
    temperature: int | float
    max_new_tokens: int


class ThinkingConfig(BaseModel):
    supported: TruthValue
    disabled_by_default: TruthValue
    disable_argument: str | None = None


class PolicyConfig(BaseModel):
    allow_moe: bool
    allow_thinking: bool
    allow_video_only: bool
    allow_audio_only: bool


class AssetModel(BaseModel):
    alias: str
    local_path: str
    hf_model_id: str | None = None
    model_config_path: str
    model_id_or_family_name: str
    family: str
    dtype: str
    trust_remote_code: bool
    attn_implementation: str | None = None
    deterministic_generation: DeterministicGenerationConfig
    thinking: ThinkingConfig
    policy: PolicyConfig
    prompt_template_id: str = "single_image_raw_question_v1"
    prompt_template_text: str = "Single-image prompt receives the normalized question text unchanged."
    hidden_state_index_offset: int | Literal["unknown"] = 1


class AssetRegistry(BaseModel):
    output_root: str
    models: list[AssetModel]


def load_asset_registry(path: str | Path) -> AssetRegistry:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry = AssetRegistry.model_validate(payload)
    aliases = [model.alias for model in registry.models]
    if aliases != list(REQUIRED_MODEL_ALIASES):
        raise ValueError(
            "Asset registry aliases must exactly match the requested Experiment 1 order. "
            f"expected={list(REQUIRED_MODEL_ALIASES)!r} actual={aliases!r}"
        )
    if len(set(aliases)) != len(aliases):
        raise ValueError("Asset registry aliases must be unique.")
    return registry
