"""Typed configuration models for MIND."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    model_id: str
    family: str
    dtype: str = "float16"
    attn_implementation: str | None = None
    trust_remote_code: bool = True


class DatasetConfig(BaseModel):
    name: str
    root: str
    image_root: str | None = None
    splits: list[str] = Field(default_factory=list)
    prompt_template: str
    normalizer: str = "object_yes_no"
    source_dataset: str | None = None
    question_template: str | None = None


class RuntimeConfig(BaseModel):
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 4
    selected_layers: int = 16


class ExperimentConfig(BaseModel):
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
