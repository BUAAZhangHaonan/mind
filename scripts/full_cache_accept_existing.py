#!/usr/bin/env python3
"""Accept existing Stage 0 full-cache roots without copying tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from full_cache_run import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    load_panel_config,
    resolve_output_root,
    route_models,
    route_source_root,
    write_csv,
)

from mind import full_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stage0-root", type=Path, default=None)
    parser.add_argument("--stage0-cache-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_panel_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    source_root = resolve_stage0_cache_source_root(
        config=config,
        stage0_root=args.stage0_root,
        stage0_cache_root=args.stage0_cache_root,
    )
    models = requested_models(
        allowed=route_models(config, "accept_existing_stage0"),
        requested=args.models,
        route_name="accept_existing_stage0",
    )
    rows = [
        accept_model(model_alias=model_alias, source_root=source_root, output_root=output_root)
        for model_alias in models
    ]
    write_csv(output_root / "reports" / "accept_existing_stage0_status.csv", rows)
    for row in rows:
        print(
            "model={model_alias} route={route} status={status} source={source_cache_root} manifest={manifest_path}".format(
                **row
            )
        )
    return 0 if all(row["status"] == "accepted_existing_stage0" for row in rows) else 2


def resolve_stage0_cache_source_root(
    *,
    config: Mapping[str, Any],
    stage0_root: Path | None,
    stage0_cache_root: Path | None,
) -> Path:
    if stage0_cache_root is not None:
        return stage0_cache_root
    if stage0_root is not None:
        return stage0_root if stage0_root.name == "cache" else stage0_root / "cache"
    return route_source_root(config, "accept_existing_stage0") or Path("outputs/stage0/cache")


def requested_models(*, allowed: Sequence[str], requested: Sequence[str] | None, route_name: str) -> list[str]:
    if not requested:
        return list(allowed)
    unknown = [model for model in requested if model not in allowed]
    if unknown:
        raise ValueError(f"models are not on route {route_name}: {unknown}")
    return [str(model) for model in requested]


def accept_model(*, model_alias: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    model_root = source_root / model_alias
    try:
        report = full_cache.accept_existing_stage0_cache(
            model_alias=model_alias,
            stage0_cache_root=model_root,
            output_root=output_root,
        )
        return compact_row(report)
    except full_cache.FullCacheValidationError as error:
        manifest = failed_acceptance_manifest(
            model_alias=model_alias,
            source_cache_root=model_root,
            output_root=output_root,
            validation=error.manifest,
            failed_reason=str(error),
        )
    except (FileNotFoundError, ValueError) as error:
        manifest = failed_acceptance_manifest(
            model_alias=model_alias,
            source_cache_root=model_root,
            output_root=output_root,
            validation={},
            failed_reason=str(error),
        )
    full_cache.write_json_manifest(manifest, manifest["manifest_path"])
    return compact_row(manifest)


def failed_acceptance_manifest(
    *,
    model_alias: str,
    source_cache_root: Path,
    output_root: Path,
    validation: Mapping[str, Any],
    failed_reason: str,
) -> dict[str, Any]:
    manifest_path = output_root / model_alias / "full_cache_acceptance_manifest.json"
    return {
        "schema_version": full_cache.MODEL_MANIFEST_SCHEMA_VERSION,
        "model_alias": model_alias,
        "route": "accept_existing_stage0",
        "status": "failed_validation",
        "cache_origin": "stage0",
        "source_cache_root": str(source_cache_root),
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
        "source_cache_root": report.get("source_cache_root", ""),
        "copied_tensors": report.get("copied_tensors", False),
        "total_entries": report.get("total_entries", 0),
        "num_shards": report.get("num_shards", 0),
        "validation_status": report.get("validation_status", ""),
        "failed_reason": report.get("failed_reason", ""),
        "manifest_path": report.get("manifest_path", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
