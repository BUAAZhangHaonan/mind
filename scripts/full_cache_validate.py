#!/usr/bin/env python3
"""Validate every routable MIND Experiment 2 full-cache root."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from full_cache_run import (
    DEFAULT_CONFIG,
    cache_output_root_for_model,
    configured_model_cache_output_root,
    load_panel_config,
    manifest_route_map,
    model_manifest_from_validation,
    model_extraction_env_name,
    resolve_output_root,
    route_extraction_env_name,
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
    parser.add_argument("--models", nargs="+", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_panel_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    routes = manifest_route_map(config)
    models = requested_models(list(routes), args.models)
    rows: list[dict[str, Any]] = []
    validation_dir = output_root / "reports" / "validation"
    for model_alias in models:
        spec = validation_spec(config=config, output_root=output_root, model_alias=model_alias, route=routes[model_alias])
        manifest = validate_existing_root(
            model_alias=model_alias,
            cache_root=spec["cache_root"],
            cache_origin=spec["cache_origin"],
            extraction_env_name=spec["extraction_env_name"],
        )
        output = validation_dir / f"{model_alias}_validation.json"
        full_cache.write_json_manifest(manifest, output)
        refreshed_manifest = refresh_extraction_manifest_from_validation(
            output_root=output_root,
            model_alias=model_alias,
            spec=spec,
            validation=manifest,
        )
        rows.append(
            {
                "model_alias": model_alias,
                "route": spec["route"],
                "cache_root": str(spec["cache_root"]),
                "cache_origin": spec["cache_origin"],
                "extraction_env_name": spec["extraction_env_name"] or "",
                "root_exists": spec["cache_root"].exists(),
                "validation_status": manifest.get("status", ""),
                "total_entries": manifest.get("total_entries", 0),
                "num_shards": manifest.get("num_shards", 0),
                "errors": len(manifest.get("errors", [])),
                "validation_manifest": str(output),
                "refreshed_manifest": str(refreshed_manifest) if refreshed_manifest is not None else "",
            }
        )
    status = {
        "schema_version": "mind_full_cache_validation_status_v1",
        "models": rows,
        "aggregate_counts": {
            "total_models": len(rows),
            "by_validation_status": dict(sorted(Counter(row["validation_status"] for row in rows).items())),
        },
    }
    reports_dir = output_root / "reports"
    full_cache.write_json_manifest(status, reports_dir / "full_cache_validation_status.json")
    write_csv(reports_dir / "full_cache_validation_status.csv", rows)
    for row in rows:
        print(
            "model={model_alias} route={route} validation_status={validation_status} errors={errors} root={cache_root}".format(
                **row
            )
        )
    return 0 if all(row["validation_status"] == "passed" for row in rows) else 2


def requested_models(available: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    if not requested:
        return list(available)
    unknown = [model for model in requested if model not in available]
    if unknown:
        raise ValueError(f"models are not in the full-cache panel: {unknown}")
    return [str(model) for model in requested]


def validation_spec(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    model_alias: str,
    route: str,
) -> dict[str, Any]:
    if route == "accept_existing_stage0":
        source_root = route_source_root(config, "accept_existing_stage0") or Path("outputs/stage0/cache")
        return {
            "route": route,
            "cache_root": source_root / model_alias,
            "cache_origin": "stage0",
            "extraction_env_name": None,
        }
    if route == "accept_existing_separate_env":
        source_root = route_source_root(config, "accept_existing_separate_env") or Path("outputs/assets_gemma4_tf5102/full_cache")
        return {
            "route": route,
            "cache_root": source_root / model_alias,
            "cache_origin": "separate_env",
            "extraction_env_name": route_extraction_env_name(config, "accept_existing_separate_env"),
        }
    if route == "extract_separate_env":
        return {
            "route": route,
            "cache_root": separate_env_extraction_cache_root(
                config=config,
                output_root=output_root,
                model_alias=model_alias,
            ),
            "cache_origin": "separate_env",
            "extraction_env_name": model_extraction_env_name(
                config,
                model_alias,
                "extract_separate_env",
                route_default=route_extraction_env_name(config, "extract_separate_env"),
            ),
        }
    return {
        "route": route,
        "cache_root": cache_output_root_for_model(
            config=config,
            output_root=output_root,
            model_alias=model_alias,
            route_name="extract_main_env",
        ) / model_alias,
        "cache_origin": "default_env",
        "extraction_env_name": route_extraction_env_name(config, "extract_main_env"),
    }


def separate_env_extraction_cache_root(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    model_alias: str,
) -> Path:
    configured_root = configured_model_cache_output_root(config, model_alias)
    if configured_root is not None:
        return model_cache_root(configured_root, model_alias)
    route_root = route_source_root(config, "extract_separate_env")
    if route_root is not None:
        return model_cache_root(route_root, model_alias)
    manifest_root = existing_extraction_manifest_cache_root(output_root, model_alias)
    if manifest_root is not None:
        return manifest_root
    return output_root / "separate_env" / "cache" / model_alias


def model_cache_root(root: Path, model_alias: str) -> Path:
    if root.name == model_alias:
        return root
    return root / model_alias


def existing_extraction_manifest_cache_root(output_root: Path, model_alias: str) -> Path | None:
    manifest_path = output_root / model_alias / "full_cache_extraction_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("cache_root") or payload.get("source_cache_root")
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def refresh_extraction_manifest_from_validation(
    *,
    output_root: Path,
    model_alias: str,
    spec: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path | None:
    status = extracted_status_for_route(str(spec["route"]))
    if status is None or validation.get("status") != "passed":
        return None
    cache_root = Path(str(validation.get("cache_root") or spec["cache_root"]))
    manifest = model_manifest_from_validation(
        model_alias=model_alias,
        route=str(spec["route"]),
        status=status,
        cache_origin=str(spec["cache_origin"]),
        cache_root=cache_root,
        validation=validation,
        extraction_env_name=spec.get("extraction_env_name"),
        log_path=None,
        failed_reason="",
    )
    return write_model_manifest(output_root, model_alias, manifest)


def extracted_status_for_route(route: str) -> str | None:
    if route == "extract_default_env":
        return "extracted_main_env"
    if route == "extract_separate_env":
        return "extracted_separate_env"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
