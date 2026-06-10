#!/usr/bin/env python3
"""Write MIND Experiment 2 full-cache summary reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from full_cache_build_unified_manifest import discover_model_manifests
from full_cache_run import DEFAULT_CONFIG, load_panel_config, read_json, resolve_output_root, write_plan_artifacts

from mind import full_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--unified-manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_panel_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    unified_path = args.unified_manifest or output_root / "manifests" / "unified_full_cache_manifest.json"
    unified = load_or_build_unified(
        config=config,
        config_path=args.config,
        output_root=output_root,
        unified_path=unified_path,
        rebuild=args.unified_manifest is None,
    )
    report = full_cache.render_full_cache_report(unified)
    reports_dir = output_root / "reports"
    md_path = reports_dir / "FULL_CACHE_SUMMARY.md"
    json_path = reports_dir / "FULL_CACHE_SUMMARY.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report, encoding="utf-8")
    full_cache.write_json_manifest(
        {
            "schema_version": "mind_full_cache_summary_v1",
            "markdown_path": str(md_path),
            "unified_manifest_path": str(unified_path),
            "aggregate_counts": unified.get("aggregate_counts", {}),
            "models": unified.get("models", []),
        },
        json_path,
    )
    print(f"summary_markdown={md_path}")
    print(f"summary_json={json_path}")
    return 0


def load_or_build_unified(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    unified_path: Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    if unified_path.exists() and not rebuild:
        return read_json(unified_path)
    route_manifest_path = output_root / "manifests" / "model_extraction_routes.json"
    if route_manifest_path.exists():
        route_manifest = read_json(route_manifest_path)
    else:
        route_manifest = write_plan_artifacts(config=config, config_path=config_path, output_root=output_root)
    model_manifests = discover_model_manifests(output_root=output_root, route_manifest=route_manifest)
    return full_cache.build_unified_full_cache_manifest(
        route_manifest=route_manifest,
        model_manifests=model_manifests,
        output=unified_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
