#!/usr/bin/env python3
"""Build the unified MIND Experiment 2 full-cache manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from full_cache_run import (
    DEFAULT_CONFIG,
    load_panel_config,
    read_json,
    resolve_output_root,
    write_csv,
    write_plan_artifacts,
)

from mind import full_cache
from mind.models.registry import REQUIRED_MODEL_ALIASES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--route-manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_panel_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    route_manifest_path = args.route_manifest or output_root / "manifests" / "model_extraction_routes.json"
    if route_manifest_path.exists():
        route_manifest = read_json(route_manifest_path)
    else:
        route_manifest = write_plan_artifacts(config=config, config_path=args.config, output_root=output_root)
    model_manifests = discover_model_manifests(output_root=output_root, route_manifest=route_manifest)
    manifests_dir = output_root / "manifests"
    unified_path = manifests_dir / "unified_full_cache_manifest.json"
    unified = full_cache.build_unified_full_cache_manifest(
        route_manifest=route_manifest,
        model_manifests=model_manifests,
        output=unified_path,
    )
    write_csv(manifests_dir / "unified_full_cache_manifest.csv", unified_rows(unified))
    write_csv(manifests_dir / "extraction_ledger.csv", ledger_rows(route_manifest=route_manifest, unified_manifest=unified))
    print(f"unified_manifest={unified_path}")
    print(f"models={unified['aggregate_counts']['total_models']}")
    print(f"total_entries={unified['aggregate_counts']['total_entries']}")
    return 0


def discover_model_manifests(*, output_root: Path, route_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    route_by_alias = {
        str(row.get("model_alias")): str(row.get("route"))
        for row in route_manifest.get("models", [])
        if isinstance(row, Mapping) and row.get("model_alias") is not None
    }
    manifests: list[dict[str, Any]] = []
    for alias in REQUIRED_MODEL_ALIASES:
        route = route_by_alias.get(alias, full_cache.route_for_model(alias))
        for path in candidate_manifest_paths(output_root=output_root, model_alias=alias, route=route):
            if not path.exists():
                continue
            manifest = read_json(path)
            manifest.setdefault("manifest_path", str(path))
            manifests.append(manifest)
            break
    return manifests


def candidate_manifest_paths(*, output_root: Path, model_alias: str, route: str) -> list[Path]:
    base = output_root / model_alias
    acceptance = base / "full_cache_acceptance_manifest.json"
    extraction = base / "full_cache_extraction_manifest.json"
    if route.startswith("accept_existing"):
        return [acceptance, extraction]
    return [extraction, acceptance]


def unified_rows(unified_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unified_manifest.get("models", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "model_alias": row.get("model_alias", ""),
                "route": row.get("route", ""),
                "status": row.get("status", ""),
                "total_entries": row.get("total_entries", 0),
                "num_shards": row.get("num_shards", 0),
                "cache_root": row.get("cache_root", ""),
                "source_cache_root": row.get("source_cache_root", ""),
                "manifest_path": row.get("manifest_path", ""),
                "failed_reason": row.get("failed_reason", ""),
            }
        )
    return rows


def ledger_rows(*, route_manifest: Mapping[str, Any], unified_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan_by_alias = {
        str(row.get("model_alias")): row
        for row in route_manifest.get("execution_plan", [])
        if isinstance(row, Mapping) and row.get("model_alias") is not None
    }
    unified_by_alias = {
        str(row.get("model_alias")): row
        for row in unified_manifest.get("models", [])
        if isinstance(row, Mapping) and row.get("model_alias") is not None
    }
    rows: list[dict[str, Any]] = []
    for alias in REQUIRED_MODEL_ALIASES:
        plan = plan_by_alias.get(alias, {})
        model = unified_by_alias.get(alias, {})
        rows.append(
            {
                "model_alias": alias,
                "route": model.get("route", plan.get("route", "")),
                "planned_status": plan.get("status", ""),
                "final_status": model.get("status", ""),
                "action": plan.get("action", ""),
                "command": plan.get("command", ""),
                "total_entries": model.get("total_entries", 0),
                "num_shards": model.get("num_shards", 0),
                "manifest_path": model.get("manifest_path", ""),
                "failed_reason": model.get("failed_reason", ""),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
