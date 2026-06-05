#!/usr/bin/env python3
"""Audit local Experiment 1 model assets without loading model weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.asset_validation import AssetStatus, audit_asset_metadata, build_completion_summary
from mind.models.registry import load_asset_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Reserved for explicit heavy checks. The default audit never loads weights.",
    )
    return parser


def run_audit(*, registry_path: Path, output_root: Path, load_model: bool = False) -> list[dict[str, object]]:
    if load_model:
        raise ValueError("--load-model is not implemented for the lightweight Experiment 1 audit")
    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_asset_registry(registry_path)
    results = [audit_asset_metadata(model).as_dict() for model in registry.models]

    inventory_fields = [
        "alias",
        "local_path",
        "path_exists",
        "path_is_directory",
        "config_exists",
        "processor_tokenizer_assets",
        "model_family_detected",
        "architecture_detected",
        "status",
        "reason",
    ]
    capability_fields = [
        "alias",
        "model_family_detected",
        "architecture_detected",
        "moe_indicators",
        "thinking_detected",
        "thinking_disable_argument",
        "dtype",
        "trust_remote_code_required",
        "local_loading_class_candidate",
        "image_processor_candidate",
        "total_layers",
        "hidden_dim",
        "output_hidden_states_support",
        "generation_api_support",
        "hidden_state_index_offset",
        "prompt_template_id",
        "status",
        "reason",
    ]
    _write_csv(output_root / "asset_inventory.csv", results, inventory_fields)
    _write_csv(output_root / "model_capability_matrix.csv", results, capability_fields)

    unsupported = [
        row
        for row in results
        if row["status"] in {AssetStatus.UNSUPPORTED_BY_POLICY.value, AssetStatus.UNSUPPORTED_BY_WRAPPER.value}
    ]
    blocked = [row for row in results if row["status"] == AssetStatus.BLOCKED.value]
    failed = [row for row in results if row["status"] == AssetStatus.FAILED_VALIDATION.value]
    model_statuses = {str(row["alias"]): str(row["status"]) for row in results}
    model_reasons = {str(row["alias"]): str(row["reason"]) for row in results}
    summary = build_completion_summary(
        model_statuses=model_statuses,
        model_reasons=model_reasons,
        smoke_datasets=[],
        smoke_limit=0,
        tests_run=[],
        git_commit=get_git_commit(),
    )
    manifest = {
        "registry": str(registry_path),
        "output_root": str(output_root),
        "load_model": False,
        "git_commit": get_git_commit(),
        "models": results,
        "summary": summary,
    }
    _write_json(output_root / "model_asset_manifest.json", manifest)
    _write_json(output_root / "unsupported_models.json", unsupported)
    _write_json(output_root / "blocked_models.json", blocked)
    if failed:
        _write_json(output_root / "failed_models.json", failed)
    return results


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = run_audit(
            registry_path=args.registry,
            output_root=args.output_root,
            load_model=args.load_model,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"asset_audit models={len(rows)} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
