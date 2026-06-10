from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

from mind.models.registry import REQUIRED_MODEL_ALIASES

from .conftest import write_synthetic_full_cache_root


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

full_cache_validate = importlib.import_module("full_cache_validate")
full_cache_report = importlib.import_module("full_cache_report")


CONFIG_PATH = Path("configs/full_cache/model_panel.yaml")


def test_closeout_refreshes_missing_qwen35_9b_manifest_from_main_env_root(tmp_path: Path) -> None:
    output_root = tmp_path / "full_cache"
    model_alias = "qwen3.5-9b"
    cache_root = output_root / "main_env" / "cache" / model_alias
    write_synthetic_full_cache_root(
        cache_root,
        model_alias=model_alias,
        cache_origin="default_env",
    )

    rc = full_cache_validate.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(output_root),
            "--models",
            model_alias,
        ]
    )

    manifest_path = output_root / model_alias / "full_cache_extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_status = json.loads(
        (output_root / "reports" / "full_cache_validation_status.json").read_text(encoding="utf-8")
    )
    row = validation_status["models"][0]

    assert rc == 0
    assert row["cache_root"] == str(cache_root)
    assert "full_cache/full_cache" not in row["cache_root"]
    assert row["validation_status"] == "passed"
    assert manifest["model_alias"] == model_alias
    assert manifest["route"] == "extract_default_env"
    assert manifest["status"] == "extracted_main_env"
    assert manifest["cache_origin"] == "default_env"
    assert manifest["cache_root"] == str(cache_root)
    assert manifest["total_entries"] == 3
    assert manifest["num_shards"] == 3
    assert manifest["validation_status"] == "passed"
    assert manifest["manifest_path"] == str(manifest_path)


def test_closeout_refreshes_stale_phi35_manifest_from_main_env_root(tmp_path: Path) -> None:
    output_root = tmp_path / "full_cache"
    model_alias = "phi-3.5-vision-instruct"
    cache_root = output_root / "main_env" / "cache" / model_alias
    stale_root = output_root / "full_cache" / model_alias
    write_synthetic_full_cache_root(
        cache_root,
        model_alias=model_alias,
        cache_origin="default_env",
    )
    manifest_path = output_root / model_alias / "full_cache_extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "mind_full_cache_model_manifest_v1",
                "model_alias": model_alias,
                "route": "extract_default_env",
                "status": "extracted_default_env",
                "cache_origin": "default_env",
                "cache_root": str(stale_root),
                "total_entries": 16805,
                "num_shards": 134,
                "datasets": {},
                "validation_status": "passed",
                "validation_errors": [],
                "failed_reason": "",
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = full_cache_validate.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(output_root),
            "--models",
            model_alias,
        ]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert manifest["status"] == "extracted_main_env"
    assert manifest["cache_root"] == str(cache_root)
    assert manifest["total_entries"] == 3
    assert manifest["num_shards"] == 3
    assert manifest["validation_status"] == "passed"
    assert manifest["manifest_path"] == str(manifest_path)


def test_closeout_resolves_separate_env_root_from_existing_extraction_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "full_cache"
    model_alias = "molmo-7b-d-0924"
    cache_root = tmp_path / "assets_molmo_tf457" / "full_cache" / model_alias
    write_synthetic_full_cache_root(
        cache_root,
        model_alias=model_alias,
        cache_origin="separate_env",
        extraction_env_name="mind-molmo-py311",
    )
    manifest_path = output_root / model_alias / "full_cache_extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "mind_full_cache_model_manifest_v1",
                "model_alias": model_alias,
                "route": "extract_separate_env",
                "status": "extracted_separate_env",
                "cache_origin": "separate_env",
                "cache_root": str(cache_root),
                "extraction_env_name": "mind-molmo-py311",
                "total_entries": 19867,
                "num_shards": 158,
                "datasets": {},
                "validation_status": "passed",
                "validation_errors": [],
                "failed_reason": "",
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = full_cache_validate.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(output_root),
            "--models",
            model_alias,
        ]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_status = json.loads(
        (output_root / "reports" / "full_cache_validation_status.json").read_text(encoding="utf-8")
    )
    row = validation_status["models"][0]

    assert rc == 0
    assert row["cache_root"] == str(cache_root)
    assert "full_cache/full_cache" not in row["cache_root"]
    assert manifest["status"] == "extracted_separate_env"
    assert manifest["cache_root"] == str(cache_root)
    assert manifest["total_entries"] == 3
    assert manifest["num_shards"] == 3
    assert manifest["extraction_env_name"] == "mind-molmo-py311"


def test_report_rebuilds_default_unified_manifest_from_refreshed_model_manifests(tmp_path: Path) -> None:
    output_root = tmp_path / "full_cache"
    stale_unified = output_root / "manifests" / "unified_full_cache_manifest.json"
    stale_unified.parent.mkdir(parents=True)
    stale_unified.write_text(
        json.dumps(
            {
                "schema_version": "mind_full_cache_unified_manifest_v1",
                "models": [
                    {
                        "schema_version": "mind_full_cache_model_manifest_v1",
                        "model_alias": "qwen3.5-9b",
                        "route": "extract_default_env",
                        "status": "failed_extraction",
                        "total_entries": 0,
                        "num_shards": 0,
                        "failed_reason": "missing full-cache model manifest",
                        "datasets": {},
                    }
                ],
                "aggregate_counts": {
                    "total_models": 1,
                    "total_entries": 0,
                    "by_status": {"failed_extraction": 1},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for alias in REQUIRED_MODEL_ALIASES:
        route = _route_for_alias(alias)
        status = _status_for_route(route)
        manifest_name = "full_cache_acceptance_manifest.json" if route.startswith("accept_existing") else "full_cache_extraction_manifest.json"
        manifest_path = output_root / alias / manifest_name
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "mind_full_cache_model_manifest_v1",
                    "model_alias": alias,
                    "route": route,
                    "status": status,
                    "cache_origin": "stage0" if route == "accept_existing_stage0" else "default_env",
                    "cache_root": str(output_root / "main_env" / "cache" / alias),
                    "total_entries": 3,
                    "num_shards": 3,
                    "datasets": {},
                    "validation_status": "passed",
                    "validation_errors": [],
                    "failed_reason": "",
                    "manifest_path": str(manifest_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    rc = full_cache_report.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--output-root",
            str(output_root),
        ]
    )

    summary = json.loads((output_root / "reports" / "FULL_CACHE_SUMMARY.json").read_text(encoding="utf-8"))
    models = {model["model_alias"]: model for model in summary["models"]}

    assert rc == 0
    assert summary["aggregate_counts"]["total_models"] == 16
    assert "failed_extraction" not in summary["aggregate_counts"]["by_status"]
    assert models["qwen3.5-9b"]["status"] == "extracted_main_env"
    assert models["qwen3.5-9b"]["total_entries"] == 3


def _status_for_route(route: str) -> str:
    if route == "accept_existing_stage0":
        return "accepted_existing_stage0"
    if route == "accept_existing_separate_env":
        return "accepted_existing_separate_env"
    if route == "extract_separate_env":
        return "extracted_separate_env"
    return "extracted_main_env"


def _route_for_alias(alias: str) -> str:
    if alias in {"qwen3-vl-8b", "internvl3.5-8b"}:
        return "accept_existing_stage0"
    if alias == "gemma-4-12b-it":
        return "accept_existing_separate_env"
    if alias in {"molmo-7b-d-0924", "qwen3.5-4b"}:
        return "extract_separate_env"
    return "extract_default_env"
