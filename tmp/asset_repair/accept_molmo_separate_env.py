#!/usr/bin/env python3
"""Accept Molmo as a separate-environment verified asset.

This script does not copy smoke tensors into the main output root. It only
checks the separate-env smoke and validation artifacts, then writes a manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, normalize_mode


ALIAS = "molmo-7b-d-0924"
EXPECTED_DATASETS = (("pope", "popular"), ("repope", "popular"), ("dash-b", "all"))
REQUIRED_SIDECAR_FIELDS = {
    "model_alias",
    "model_family",
    "total_layers",
    "hidden_dim",
    "hidden_state_index_offset",
    "selected_layers",
    "token_index",
    "prompt_template_id",
    "validation_commit",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "molmo_separate_env_acceptance.json"
    md_path = output_root / "molmo_separate_env_acceptance.md"
    report["manifest_json"] = str(json_path)
    report["manifest_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Molmo Separate-Env Acceptance",
        "",
        f"- status: {report['status']}",
        f"- reason: {report['reason']}",
        f"- source_root: {report['source_root']}",
        f"- copied_tensors: {report['copied_tensors']}",
        "",
        "```json",
        json.dumps(report, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _verified_row(rows: list[dict[str, str]], dataset: str, subset: str) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("model_alias") == ALIAS
            and row.get("dataset") == dataset
            and row.get("subset") == subset
            and row.get("status") == "verified"
        ):
            return row
    return None


def _relative_or_absolute(root: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return root / path


def _check_shard(shard_path: Path, sidecar_path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not sidecar_path.is_file():
        return False, f"missing sidecar: {sidecar_path}", {}
    if not shard_path.is_file():
        return False, f"missing shard: {shard_path}", {}
    sidecar = _read_json(sidecar_path)
    config = sidecar.get("config") if isinstance(sidecar.get("config"), dict) else {}
    merged = {**config, **sidecar}
    missing = sorted(field for field in REQUIRED_SIDECAR_FIELDS if field not in merged or merged.get(field) in (None, "", "unknown"))
    if missing:
        return False, f"sidecar missing required fields: {', '.join(missing)}", merged
    if merged.get("model_alias") != ALIAS:
        return False, f"sidecar model_alias mismatch: {merged.get('model_alias')}", merged
    total_layers = int(merged["total_layers"])
    selected_layers = list(merged["selected_layers"])
    if selected_layers != list(range(total_layers)):
        return False, "sidecar selected_layers are not contiguous full layers", merged

    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or not payload:
        return False, "shard payload is not a non-empty list", merged
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            return False, f"shard entry {index} is not a dict", merged
        layer_vectors = entry.get("layer_vectors")
        if not torch.is_tensor(layer_vectors):
            return False, f"shard entry {index} missing tensor layer_vectors", merged
        if layer_vectors.ndim != 2:
            return False, f"shard entry {index} layer_vectors ndim is not 2", merged
        if layer_vectors.shape[0] != total_layers:
            return False, f"shard entry {index} layer count mismatch", merged
        if not bool(torch.isfinite(layer_vectors).all().item()):
            return False, f"shard entry {index} layer_vectors contain non-finite values", merged
        if list(entry.get("selected_layers", [])) != list(range(total_layers)):
            return False, f"shard entry {index} selected_layers mismatch", merged
    return True, "ok", merged


def accept_molmo_separate_env(*, source_root: Path, output_root: Path, execute: bool = False) -> dict[str, Any]:
    summary = _read_json(source_root / "asset_completion_summary.json")
    smoke_rows = _read_csv(source_root / "smoke_extraction_report.csv")
    validation_rows = _read_csv(source_root / "hidden_state_validation_report.csv")
    report: dict[str, Any] = {
        "model_alias": ALIAS,
        "status": "blocked_remove_from_panel",
        "reason": "",
        "mode": "execute" if execute else "dry_run",
        "source_root": str(source_root),
        "copied_tensors": False,
        "expected_datasets": [f"{dataset}/{subset}" for dataset, subset in EXPECTED_DATASETS],
        "smoke_limit": summary.get("smoke_limit"),
        "deterministic_repeat_check": "covered_by_verified_validation_report",
        "image_sensitivity_canary": "covered_by_verified_validation_report",
        "checked_shards": [],
    }

    model_status = summary.get("model_statuses", {}).get(ALIAS) if isinstance(summary.get("model_statuses"), dict) else None
    verified_models = summary.get("verified_models") if isinstance(summary.get("verified_models"), list) else []
    if model_status != "verified" and ALIAS not in verified_models:
        report["reason"] = "molmo-7b-d-0924 is not verified in separate-env asset_completion_summary.json"
        _write_manifest(output_root, report)
        return report
    if summary.get("smoke_limit") != 2:
        report["reason"] = "separate-env smoke_limit is not 2"
        _write_manifest(output_root, report)
        return report

    for dataset, subset in EXPECTED_DATASETS:
        smoke_row = _verified_row(smoke_rows, dataset, subset)
        if smoke_row is None:
            report["reason"] = f"missing verified smoke row for {dataset}/{subset}"
            _write_manifest(output_root, report)
            return report
        validation_row = _verified_row(validation_rows, dataset, subset)
        if validation_row is None:
            report["reason"] = f"missing verified validation row for {dataset}/{subset}"
            _write_manifest(output_root, report)
            return report

        fallback_shard = source_root / "smoke_cache" / ALIAS / dataset / subset / "shard-00000.pt"
        shard_path = _relative_or_absolute(source_root, smoke_row.get("shard_path") or validation_row.get("shard_path"), fallback_shard)
        sidecar_path = _relative_or_absolute(source_root, smoke_row.get("sidecar_path"), Path(str(shard_path) + ".json"))
        ok, reason, metadata = _check_shard(shard_path, sidecar_path)
        if not ok:
            report["reason"] = reason
            _write_manifest(output_root, report)
            return report
        report["checked_shards"].append(
            {
                "dataset": dataset,
                "subset": subset,
                "shard_path": str(shard_path),
                "sidecar_path": str(sidecar_path),
                "total_layers": metadata["total_layers"],
                "hidden_dim": metadata["hidden_dim"],
            }
        )

    report["status"] = "verified_separate_env"
    report["reason"] = "Molmo separate-env smoke and hidden-state validation artifacts satisfy the acceptance contract"
    _write_manifest(output_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--source-root", type=Path, default=Path("outputs/assets_molmo_tf457"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = normalize_mode(build_parser().parse_args(argv))
    accept_molmo_separate_env(source_root=args.source_root, output_root=args.output_root, execute=bool(args.execute))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
