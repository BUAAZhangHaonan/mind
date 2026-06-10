#!/usr/bin/env python3
"""Accept separate-environment full-cache roots or mark extraction needs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from full_cache_run import (
    DEFAULT_CONFIG,
    cache_output_root_for_model,
    load_panel_config,
    model_manifest_from_validation,
    model_extraction_env_name,
    resolve_output_root,
    route_extraction_env_name,
    route_models,
    route_source_root,
    validate_existing_root,
    write_csv,
    write_model_manifest,
)

from mind import full_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--gemma4-root", type=Path, default=None)
    parser.add_argument("--molmo-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--extraction-env-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_panel_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    default_models = route_models(config, "accept_existing_separate_env") + route_models(config, "extract_separate_env")
    models = requested_models(allowed=default_models, requested=args.models)
    rows = [
        accept_or_mark_model(
            config=config,
            model_alias=model_alias,
            output_root=output_root,
            source_root=args.source_root,
            gemma4_root=args.gemma4_root,
            molmo_root=args.molmo_root,
            extraction_env_name=args.extraction_env_name,
        )
        for model_alias in models
    ]
    write_csv(output_root / "reports" / "accept_separate_env_status.csv", rows)
    for row in rows:
        print(
            "model={model_alias} route={route} status={status} source={source_cache_root} manifest={manifest_path}".format(
                **row
            )
        )
    return 0 if all(row["status"] != "failed_validation" for row in rows) else 2


def requested_models(*, allowed: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    if not requested:
        return list(allowed)
    unknown = [model for model in requested if model not in allowed]
    if unknown:
        raise ValueError(f"models are not separate-env full-cache models: {unknown}")
    return [str(model) for model in requested]


def accept_or_mark_model(
    *,
    config: Mapping[str, Any],
    model_alias: str,
    output_root: Path,
    source_root: Path | None,
    gemma4_root: Path | None,
    molmo_root: Path | None,
    extraction_env_name: str | None,
) -> dict[str, Any]:
    accept_models = set(route_models(config, "accept_existing_separate_env"))
    extract_models = set(route_models(config, "extract_separate_env"))
    if model_alias in accept_models:
        default_root = route_source_root(config, "accept_existing_separate_env")
        env_name = extraction_env_name or model_extraction_env_name(
            config,
            model_alias,
            "accept_existing_separate_env",
            route_default=route_extraction_env_name(config, "accept_existing_separate_env"),
        )
        if env_name is None:
            raise ValueError("extraction_env_name is required for separate-env acceptance")
        return accept_existing_separate_env_model(
            model_alias=model_alias,
            source_root=(
                model_specific_full_cache_source_root(gemma4_root, model_alias)
                if gemma4_root is not None
                else source_root or default_root or Path("outputs/assets_gemma4_tf5102/full_cache")
            ),
            output_root=output_root,
            extraction_env_name=env_name,
        )
    if model_alias in extract_models:
        env_name = extraction_env_name or model_extraction_env_name(
            config,
            model_alias,
            "extract_separate_env",
            route_default=route_extraction_env_name(config, "extract_separate_env"),
        )
        if env_name is None:
            raise ValueError("extraction_env_name is required for separate-env extraction roots")
        if source_root is not None:
            candidate_source_root = source_root
        elif molmo_root is not None and model_alias == "molmo-7b-d-0924":
            candidate_source_root = model_specific_full_cache_source_root(molmo_root, model_alias)
        else:
            candidate_source_root = cache_output_root_for_model(
                config=config,
                output_root=output_root,
                model_alias=model_alias,
                route_name="extract_separate_env",
            )
        candidate_root = resolve_model_cache_root(candidate_source_root, model_alias)
        if not candidate_root.exists():
            manifest = needs_extraction_manifest(
                model_alias=model_alias,
                output_root=output_root,
                source_cache_root=candidate_root,
                extraction_env_name=env_name,
            )
            write_model_manifest(output_root, model_alias, manifest)
            return compact_row(manifest)
        return accept_extracted_separate_env_model(
            model_alias=model_alias,
            source_cache_root=candidate_root,
            output_root=output_root,
            extraction_env_name=env_name,
        )
    raise ValueError(f"unsupported separate-env model: {model_alias}")


def accept_existing_separate_env_model(
    *,
    model_alias: str,
    source_root: Path,
    output_root: Path,
    extraction_env_name: str,
) -> dict[str, Any]:
    model_root = resolve_model_cache_root(source_root, model_alias)
    try:
        report = full_cache.accept_separate_env_cache(
            model_alias=model_alias,
            separate_env_cache_root=model_root,
            output_root=output_root,
            extraction_env_name=extraction_env_name,
        )
        return compact_row(report)
    except full_cache.FullCacheValidationError as error:
        manifest = failed_separate_env_manifest(
            model_alias=model_alias,
            route="accept_existing_separate_env",
            source_cache_root=model_root,
            output_root=output_root,
            validation=error.manifest,
            extraction_env_name=extraction_env_name,
            failed_reason=str(error),
            manifest_name="full_cache_acceptance_manifest.json",
        )
    except (FileNotFoundError, ValueError) as error:
        manifest = failed_separate_env_manifest(
            model_alias=model_alias,
            route="accept_existing_separate_env",
            source_cache_root=model_root,
            output_root=output_root,
            validation={},
            extraction_env_name=extraction_env_name,
            failed_reason=str(error),
            manifest_name="full_cache_acceptance_manifest.json",
        )
    full_cache.write_json_manifest(manifest, manifest["manifest_path"])
    return compact_row(manifest)


def accept_extracted_separate_env_model(
    *,
    model_alias: str,
    source_cache_root: Path,
    output_root: Path,
    extraction_env_name: str,
) -> dict[str, Any]:
    validation = validate_existing_root(
        model_alias=model_alias,
        cache_root=source_cache_root,
        cache_origin="separate_env",
        extraction_env_name=extraction_env_name,
    )
    status = "extracted_separate_env" if validation.get("status") == "passed" else "failed_validation"
    failed_reason = "" if status == "extracted_separate_env" else "; ".join(str(item) for item in validation.get("errors", [])[:3])
    manifest = model_manifest_from_validation(
        model_alias=model_alias,
        route="extract_separate_env",
        status=status,
        cache_origin="separate_env",
        cache_root=source_cache_root,
        validation=validation,
        extraction_env_name=extraction_env_name,
        log_path=None,
        failed_reason=failed_reason,
    )
    manifest_path = write_model_manifest(output_root, model_alias, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return compact_row(manifest)


def resolve_model_cache_root(source_root: Path, model_alias: str) -> Path:
    if source_root.name == model_alias:
        return source_root
    return source_root / model_alias


def model_specific_full_cache_source_root(root: Path, model_alias: str) -> Path:
    if root.name in {"full_cache", model_alias}:
        return root
    return root / "full_cache"


def needs_extraction_manifest(
    *,
    model_alias: str,
    output_root: Path,
    source_cache_root: Path,
    extraction_env_name: str,
) -> dict[str, Any]:
    manifest_path = output_root / model_alias / "full_cache_extraction_manifest.json"
    return {
        "schema_version": full_cache.MODEL_MANIFEST_SCHEMA_VERSION,
        "model_alias": model_alias,
        "route": "extract_separate_env",
        "status": "needs_extraction",
        "cache_origin": "separate_env",
        "source_cache_root": str(source_cache_root),
        "cache_root": str(source_cache_root),
        "extraction_env_name": extraction_env_name,
        "total_entries": 0,
        "num_shards": 0,
        "datasets": {},
        "validation_status": "not_run",
        "validation_errors": [],
        "failed_reason": "",
        "manifest_path": str(manifest_path),
    }


def failed_separate_env_manifest(
    *,
    model_alias: str,
    route: str,
    source_cache_root: Path,
    output_root: Path,
    validation: Mapping[str, Any],
    extraction_env_name: str,
    failed_reason: str,
    manifest_name: str,
) -> dict[str, Any]:
    manifest_path = output_root / model_alias / manifest_name
    return {
        "schema_version": full_cache.MODEL_MANIFEST_SCHEMA_VERSION,
        "model_alias": model_alias,
        "route": route,
        "status": "failed_validation",
        "cache_origin": "separate_env",
        "source_cache_root": str(source_cache_root),
        "cache_root": str(source_cache_root),
        "extraction_env_name": extraction_env_name,
        "copied_tensors": False,
        "source_tensors_mutated": False,
        "total_entries": int(validation.get("total_entries") or 0),
        "num_shards": int(validation.get("num_shards") or 0),
        "datasets": validation.get("datasets", {}),
        "validation_status": validation.get("status", "failed"),
        "validation_errors": validation.get("errors", []),
        "failed_reason": failed_reason,
        "manifest_path": str(manifest_path),
    }


def compact_row(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_alias": report.get("model_alias", ""),
        "route": report.get("route", ""),
        "status": report.get("status", ""),
        "source_cache_root": report.get("source_cache_root", report.get("cache_root", "")),
        "extraction_env_name": report.get("extraction_env_name", ""),
        "copied_tensors": report.get("copied_tensors", False),
        "total_entries": report.get("total_entries", 0),
        "num_shards": report.get("num_shards", 0),
        "validation_status": report.get("validation_status", ""),
        "failed_reason": report.get("failed_reason", ""),
        "manifest_path": report.get("manifest_path", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
