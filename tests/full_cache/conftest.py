from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mind.models.registry import REQUIRED_MODEL_ALIASES


STAGE0_ACCEPT_MODELS = ("qwen3-vl-8b", "internvl3.5-8b")
SEPARATE_ENV_MODELS = ("gemma-4-12b-it", "molmo-7b-d-0924")

VALID_ROUTES = {
    "accept_existing_stage0",
    "accept_existing_separate_env",
    "extract_default_env",
    "extract_separate_env",
}

VALID_FULL_CACHE_STATUSES = {
    "accepted_existing_stage0",
    "accepted_existing_separate_env",
    "extracted_main_env",
    "extracted_separate_env",
    "failed_extraction",
    "failed_validation",
    "needs_extraction_separate_env",
    "rejected",
}

DATASET_SPECS = (("pope", "popular"), ("repope", "popular"), ("dash-b", "all"))
SYNTHETIC_QUESTION = "Is the red cup still on the table?"


def full_cache_attr(name: str) -> Any:
    module = importlib.import_module("mind.full_cache")
    return getattr(module, name)


def synthetic_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "model_alias": "qwen3-vl-8b",
        "model_family": "qwen3_vl",
        "dataset_name": "pope",
        "source_dataset": "pope",
        "subset": "popular",
        "split": "encoder_train",
        "sample_id": "pope-popular-0001",
        "image_id": "image-0001",
        "image_path": "images/image-0001.jpg",
        "question": SYNTHETIC_QUESTION,
        "label": "yes",
        "object_name": "cup",
        "answer_text": "yes",
        "parsed_answer": 1,
        "selected_layers": [0, 1],
        "layer_vectors": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        "first_token_logits": torch.tensor([0.8, -0.2], dtype=torch.float32),
        "logit_source": "pre_generation_first_token",
        "token_index": -1,
        "prompt_template_id": "synthetic_single_image_raw_question_v1",
    }
    entry.update(overrides)
    return entry


def synthetic_sidecar(
    *,
    cache_origin: str = "stage0",
    extraction_env_name: str | None = None,
    **overrides: object,
) -> dict[str, object]:
    sidecar: dict[str, object] = {
        "schema_version": "mind_full_cache_shard_v1",
        "cache_type": "full_layer_hidden_states",
        "cache_origin": cache_origin,
        "model_alias": "qwen3-vl-8b",
        "model_family": "qwen3_vl",
        "dataset_name": "pope",
        "source_dataset": "pope",
        "subset": "popular",
        "split": "encoder_train",
        "total_layers": 2,
        "selected_layers": [0, 1],
        "num_selected_layers": 2,
        "hidden_dim": 2,
        "token_index": -1,
        "max_new_tokens": 1,
        "dtype": "float32",
        "num_entries": 1,
        "prompt_template_id": "synthetic_single_image_raw_question_v1",
        "logit_source": "pre_generation_first_token",
        "created_at_utc": "2026-06-09T00:00:00Z",
        "generator": "tests.full_cache.synthetic",
        "git_commit": "test-commit",
    }
    if cache_origin == "separate_env":
        sidecar.update(
            {
                "extraction_env_name": extraction_env_name or "mind-full-cache-separate-env",
                "extraction_env_python": "3.11",
                "extraction_env_packages": {"torch": "synthetic"},
            }
        )
    sidecar.update(overrides)
    return sidecar


def write_synthetic_full_cache_root(
    cache_root: Path,
    *,
    model_alias: str = "qwen3-vl-8b",
    cache_origin: str = "stage0",
    extraction_env_name: str | None = None,
    entry_overrides: Mapping[str, object] | None = None,
    sidecar_overrides: Mapping[str, object] | None = None,
    drop_entry_fields: tuple[str, ...] = (),
    drop_sidecar_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    shard_paths: list[str] = []
    sidecar_paths: list[str] = []
    for index, (dataset_name, subset) in enumerate(DATASET_SPECS):
        shard_dir = cache_root / dataset_name / subset
        shard_dir.mkdir(parents=True, exist_ok=True)
        entry = synthetic_entry(
            model_alias=model_alias,
            dataset_name=dataset_name,
            source_dataset=dataset_name,
            subset=subset,
            sample_id=f"{dataset_name}-{subset}-{index:04d}",
            image_id=f"{dataset_name}-{subset}-image-{index:04d}",
            **dict(entry_overrides or {}),
        )
        for field in drop_entry_fields:
            entry.pop(field, None)
        sidecar = synthetic_sidecar(
            cache_origin=cache_origin,
            extraction_env_name=extraction_env_name,
            model_alias=model_alias,
            dataset_name=dataset_name,
            source_dataset=dataset_name,
            subset=subset,
            **dict(sidecar_overrides or {}),
        )
        for field in drop_sidecar_fields:
            sidecar.pop(field, None)
        shard_path = shard_dir / "shard-00000.pt"
        sidecar_path = shard_dir / "shard-00000.pt.json"
        torch.save([entry], shard_path)
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        shard_paths.append(str(shard_path))
        sidecar_paths.append(str(sidecar_path))
    return {"cache_root": str(cache_root), "shard_paths": shard_paths, "sidecar_paths": sidecar_paths}


def synthetic_route_for(alias: str) -> str:
    if alias in STAGE0_ACCEPT_MODELS:
        return "accept_existing_stage0"
    if alias in SEPARATE_ENV_MODELS:
        return "accept_existing_separate_env"
    return "extract_default_env"


def synthetic_route_manifest() -> dict[str, object]:
    return {
        "schema_version": "mind_full_cache_route_manifest_v1",
        "extraction_started": False,
        "models": [
            {
                "model_alias": alias,
                "route": synthetic_route_for(alias),
                "status": "planned",
                "execution_plan": {
                    "steps": ["validate_source", "accept_or_extract", "write_model_manifest"],
                    "requires_extraction": alias not in (*STAGE0_ACCEPT_MODELS, *SEPARATE_ENV_MODELS),
                },
            }
            for alias in REQUIRED_MODEL_ALIASES
        ],
    }


def synthetic_model_manifest(
    alias: str,
    *,
    status: str,
    total_entries: int = 3,
    failed_reason: str = "",
) -> dict[str, object]:
    return {
        "schema_version": "mind_full_cache_model_manifest_v1",
        "model_alias": alias,
        "route": synthetic_route_for(alias),
        "status": status,
        "total_entries": total_entries,
        "num_shards": 3 if status != "failed_validation" else 0,
        "failed_reason": failed_reason,
        "datasets": {
            f"{dataset}/{subset}": {"num_entries": 1 if status != "failed_validation" else 0}
            for dataset, subset in DATASET_SPECS
        },
    }


def synthetic_unified_manifest() -> dict[str, object]:
    statuses = {alias: "extracted_main_env" for alias in REQUIRED_MODEL_ALIASES}
    statuses["qwen3-vl-8b"] = "accepted_existing_stage0"
    statuses["internvl3.5-8b"] = "accepted_existing_stage0"
    statuses["gemma-4-12b-it"] = "accepted_existing_separate_env"
    statuses["molmo-7b-d-0924"] = "failed_validation"
    models = [
        synthetic_model_manifest(
            alias,
            status=statuses[alias],
            total_entries=0 if statuses[alias] == "failed_validation" else 3,
            failed_reason="synthetic schema mismatch" if statuses[alias] == "failed_validation" else "",
        )
        for alias in REQUIRED_MODEL_ALIASES
    ]
    return {
        "schema_version": "mind_full_cache_unified_manifest_v1",
        "models": models,
        "aggregate_counts": {
            "total_models": len(REQUIRED_MODEL_ALIASES),
            "total_entries": sum(int(model["total_entries"]) for model in models),
            "by_status": {status: list(statuses.values()).count(status) for status in set(statuses.values())},
        },
    }
