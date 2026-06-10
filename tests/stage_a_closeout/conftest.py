from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


PANEL_MODELS = tuple(f"model-{index:02d}" for index in range(16))


def write_synthetic_shard(
    root: Path,
    *,
    model_alias: str = "model-00",
    dataset_name: str = "repope",
    subset: str = "popular",
    image_id: str = "image-0001",
    sample_id: str = "sample-0001",
    label: int = 0,
    parsed_answer: int | None = 0,
) -> Path:
    shard_dir = root / dataset_name / subset
    shard_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "sample_id": sample_id,
        "image_id": image_id,
        "image_path": f"images/{image_id}.jpg",
        "question": "Is there a chair in the image?",
        "label": label,
        "object_name": "chair",
        "answer_text": "yes" if parsed_answer == 1 else "no",
        "parsed_answer": parsed_answer,
        "source_dataset": dataset_name,
        "dataset_name": dataset_name,
        "subset": subset,
        "model_alias": model_alias,
        "model_name": model_alias,
        "selected_layers": [0, 1, 2],
        "layer_vectors": torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        ),
        "first_token_logits": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "token_index": -1,
        "prompt_template_id": "synthetic",
    }
    shard_path = shard_dir / "shard-00000.pt"
    torch.save([entry], shard_path)
    (shard_dir / "shard-00000.pt.json").write_text(
        json.dumps(
            {
                "model_alias": model_alias,
                "model_name": model_alias,
                "source_dataset": dataset_name,
                "dataset_name": dataset_name,
                "subset": subset,
                "total_layers": 3,
                "hidden_dim": 2,
                "selected_layers": [0, 1, 2],
                "num_entries": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return shard_path


def write_unified_manifest(
    full_cache_root: Path,
    *,
    models: Iterable[str] = PANEL_MODELS,
    use_source_root_for: set[str] | None = None,
) -> Path:
    use_source_root_for = use_source_root_for or set()
    manifest_models = []
    for alias in models:
        cache_root = full_cache_root / "cache" / alias
        write_synthetic_shard(cache_root, model_alias=alias)
        row = {
            "model_alias": alias,
            "status": "extracted_main_env",
            "validation_status": "passed",
            "total_entries": 1,
            "num_shards": 1,
            "datasets": {"repope/popular": {"num_entries": 1}},
        }
        if alias in use_source_root_for:
            row["source_cache_root"] = str(cache_root)
            row["status"] = "accepted_existing_stage0"
        else:
            row["cache_root"] = str(cache_root)
        manifest_models.append(row)
    manifest = {
        "schema_version": "mind_full_cache_unified_manifest_v1",
        "models": manifest_models,
    }
    path = full_cache_root / "manifests" / "unified_full_cache_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
