from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mind.models.registry import REQUIRED_MODEL_ALIASES


PANEL_MODELS = REQUIRED_MODEL_ALIASES
GLM_MODEL_ALIAS = "glm-4.6v-flash"
EXPECTED_STAGE_B_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
ALLOWED_STAGE_B_OBJECTIVES = ("bce", "supcon", "proxy_anchor")


def stage_b_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f"mind.trajectory.{module_name}")
    return getattr(module, attr_name)


def stage_b_script_attr(script_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f"scripts.{script_name}")
    return getattr(module, attr_name)


def write_unified_manifest(
    full_cache_root: Path,
    *,
    models: Iterable[str] = PANEL_MODELS,
    use_source_root_for: set[str] | None = None,
) -> Path:
    use_source_root_for = use_source_root_for or set()
    manifest_models: list[dict[str, object]] = []
    for alias in models:
        if alias in use_source_root_for:
            cache_root = full_cache_root / "source_cache" / alias
            root_field = "source_cache_root"
            status = "accepted_existing_stage0"
        else:
            cache_root = full_cache_root / "cache" / alias
            root_field = "cache_root"
            status = "extracted_main_env"
        cache_root.mkdir(parents=True, exist_ok=True)
        manifest_models.append(
            {
                "model_alias": alias,
                "status": status,
                "validation_status": "passed",
                "total_entries": 6,
                "num_shards": 3,
                "datasets": {
                    "repope/popular": {"num_entries": 2},
                    "repope/random": {"num_entries": 2},
                    "repope/adversarial": {"num_entries": 2},
                },
                root_field: str(cache_root),
            }
        )

    manifest = {
        "schema_version": "mind_full_cache_unified_manifest_v1",
        "models": manifest_models,
    }
    manifest_path = full_cache_root / "manifests" / "unified_full_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def repope_cal_metric_row(
    k: int,
    *,
    pr_auc: float,
    roc_auc: float,
    dataset_family: str = "repope",
    split: str = "cal",
) -> dict[str, object]:
    return {
        "dataset_family": dataset_family,
        "split": split,
        "metric_split": split,
        "readout": "knn",
        "k": k,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "metric_status": "passed",
    }


def stage_b_metric_row(
    model_alias: str,
    objective: str,
    *,
    pr_auc: float,
    roc_auc: float = 0.75,
    metric_status: str = "passed",
) -> dict[str, object]:
    return {
        "model_alias": model_alias,
        "model_name": model_alias,
        "dataset_family": "repope",
        "split": "test",
        "metric_split": "test",
        "encoder_family": "Sphere-Traj-LSTM",
        "objective": objective,
        "metric_status": metric_status,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
