from __future__ import annotations

import json
from pathlib import Path

from mind.models.registry import REQUIRED_MODEL_ALIASES

from .conftest import VALID_FULL_CACHE_STATUSES, full_cache_attr, synthetic_model_manifest, synthetic_route_manifest


def test_unified_manifest_contains_all_panel_models_and_valid_statuses(tmp_path: Path) -> None:
    build_unified_full_cache_manifest = full_cache_attr("build_unified_full_cache_manifest")
    model_manifests = [
        synthetic_model_manifest(alias, status="extracted_main_env") for alias in REQUIRED_MODEL_ALIASES
    ]
    model_manifests[2] = synthetic_model_manifest("qwen3-vl-8b", status="accepted_existing_stage0")
    model_manifests[3] = synthetic_model_manifest("internvl3.5-8b", status="accepted_existing_stage0")
    model_manifests[6] = synthetic_model_manifest("gemma-4-12b-it", status="accepted_existing_separate_env")
    output = tmp_path / "unified_manifest.json"

    manifest = build_unified_full_cache_manifest(
        route_manifest=synthetic_route_manifest(),
        model_manifests=model_manifests,
        output=output,
    )

    aliases = [model["model_alias"] for model in manifest["models"]]
    assert aliases == list(REQUIRED_MODEL_ALIASES)
    assert set(model["status"] for model in manifest["models"]) <= VALID_FULL_CACHE_STATUSES
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["aggregate_counts"] == manifest["aggregate_counts"]


def test_unified_manifest_aggregate_counts_match_models(tmp_path: Path) -> None:
    build_unified_full_cache_manifest = full_cache_attr("build_unified_full_cache_manifest")
    model_manifests = [
        synthetic_model_manifest(alias, status="extracted_main_env", total_entries=3)
        for alias in REQUIRED_MODEL_ALIASES
    ]
    model_manifests[2] = synthetic_model_manifest("qwen3-vl-8b", status="accepted_existing_stage0", total_entries=3)
    model_manifests[6] = synthetic_model_manifest(
        "gemma-4-12b-it",
        status="accepted_existing_separate_env",
        total_entries=3,
    )

    manifest = build_unified_full_cache_manifest(
        route_manifest=synthetic_route_manifest(),
        model_manifests=model_manifests,
        output=tmp_path / "unified_manifest.json",
    )

    assert manifest["aggregate_counts"]["total_models"] == 16
    assert manifest["aggregate_counts"]["total_entries"] == sum(
        model["total_entries"] for model in manifest["models"]
    )
    for status, count in manifest["aggregate_counts"]["by_status"].items():
        assert count == sum(1 for model in manifest["models"] if model["status"] == status)


def test_unified_manifest_keeps_failed_models_visible(tmp_path: Path) -> None:
    build_unified_full_cache_manifest = full_cache_attr("build_unified_full_cache_manifest")
    model_manifests = [
        synthetic_model_manifest(alias, status="extracted_main_env") for alias in REQUIRED_MODEL_ALIASES
    ]
    failed = synthetic_model_manifest(
        "molmo-7b-d-0924",
        status="failed_validation",
        total_entries=0,
        failed_reason="synthetic schema mismatch",
    )
    model_manifests[12] = failed

    manifest = build_unified_full_cache_manifest(
        route_manifest=synthetic_route_manifest(),
        model_manifests=model_manifests,
        output=tmp_path / "unified_manifest.json",
    )

    models = {model["model_alias"]: model for model in manifest["models"]}
    assert "molmo-7b-d-0924" in models
    assert models["molmo-7b-d-0924"]["status"] == "failed_validation"
    assert models["molmo-7b-d-0924"]["failed_reason"] == "synthetic schema mismatch"


def test_unified_manifest_keeps_qwen35_4b_extracted_separate_env_visible(tmp_path: Path) -> None:
    build_unified_full_cache_manifest = full_cache_attr("build_unified_full_cache_manifest")
    route_manifest = synthetic_route_manifest()
    for model in route_manifest["models"]:
        if model["model_alias"] == "qwen3.5-4b":
            model["route"] = "extract_separate_env"
            model["status"] = "needs_extraction"

    model_manifests = [
        synthetic_model_manifest(alias, status="extracted_main_env") for alias in REQUIRED_MODEL_ALIASES
    ]
    qwen_manifest = synthetic_model_manifest("qwen3.5-4b", status="extracted_separate_env")
    qwen_manifest.update(
        {
            "route": "extract_separate_env",
            "cache_origin": "separate_env",
            "cache_root": "outputs/assets_qwen35_tf5102/full_cache/qwen3.5-4b",
            "extraction_env_name": "mind-gemma4-py311",
        }
    )
    model_manifests[7] = qwen_manifest

    manifest = build_unified_full_cache_manifest(
        route_manifest=route_manifest,
        model_manifests=model_manifests,
        output=tmp_path / "unified_manifest.json",
    )

    models = {model["model_alias"]: model for model in manifest["models"]}
    assert models["qwen3.5-4b"]["route"] == "extract_separate_env"
    assert models["qwen3.5-4b"]["status"] == "extracted_separate_env"
    assert models["qwen3.5-4b"]["cache_root"] == "outputs/assets_qwen35_tf5102/full_cache/qwen3.5-4b"
    assert models["qwen3.5-4b"]["extraction_env_name"] == "mind-gemma4-py311"
