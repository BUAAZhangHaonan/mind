#!/usr/bin/env python3
"""Accept separate-environment asset validation artifacts into the asset summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.asset_validation import AssetStatus


MOLMO_ALIAS = "molmo-7b-d-0924"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def accept_separate_env(*, model: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    if model != MOLMO_ALIAS:
        raise ValueError(f"separate-env acceptance is implemented only for {MOLMO_ALIAS}, got {model}")
    summary = _read_json(source_root / "asset_completion_summary.json")
    smoke_rows = _read_csv(source_root / "smoke_extraction_report.csv")
    validation_rows = _read_csv(source_root / "hidden_state_validation_report.csv")
    validation_checksums = _read_json(source_root / "validation_checksums.json")
    report: dict[str, Any] = {
        "model_alias": MOLMO_ALIAS,
        "status": AssetStatus.BLOCKED.value,
        "reason": "",
        "source_root": str(source_root),
        "copied_tensors": False,
        "expected_datasets": [f"{dataset}/{subset}" for dataset, subset in EXPECTED_DATASETS],
        "smoke_limit": summary.get("smoke_limit") if isinstance(summary, Mapping) else None,
        "checked_shards": [],
        "checked_determinism": [],
        "checked_image_sensitivity_canary": None,
    }

    model_status = _nested_mapping(summary, "model_statuses").get(MOLMO_ALIAS)
    verified_models = summary.get("verified_models") if isinstance(summary, Mapping) and isinstance(summary.get("verified_models"), list) else []
    if model_status != AssetStatus.VERIFIED.value and MOLMO_ALIAS not in verified_models:
        report["reason"] = "molmo-7b-d-0924 is not verified in separate-env asset_completion_summary.json"
        _write_report(output_root, report)
        return report
    if summary.get("smoke_limit") != 2:
        report["reason"] = "separate-env smoke_limit is not 2"
        _write_report(output_root, report)
        return report

    for dataset, subset in EXPECTED_DATASETS:
        smoke_row = _verified_row(smoke_rows, dataset, subset)
        validation_row = _verified_row(validation_rows, dataset, subset)
        if smoke_row is None:
            report["reason"] = f"missing verified smoke row for {dataset}/{subset}"
            _write_report(output_root, report)
            return report
        if validation_row is None:
            report["reason"] = f"missing verified validation row for {dataset}/{subset}"
            _write_report(output_root, report)
            return report
        ok, reason = _check_dataset_checksums(validation_checksums, dataset, subset)
        if not ok:
            report["reason"] = reason
            _write_report(output_root, report)
            return report
        report["checked_determinism"].append(f"{dataset}/{subset}")
        fallback_shard = source_root / "smoke_cache" / MOLMO_ALIAS / dataset / subset / "shard-00000.pt"
        shard_path = _relative_or_absolute(source_root, smoke_row.get("shard_path") or validation_row.get("shard_path"), fallback_shard)
        sidecar_path = _relative_or_absolute(source_root, smoke_row.get("sidecar_path"), Path(str(shard_path) + ".json"))
        ok, reason, metadata = _check_shard(shard_path, sidecar_path)
        if not ok:
            report["reason"] = reason
            _write_report(output_root, report)
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
    ok, reason, canary = _check_image_sensitivity_canary(validation_checksums)
    if not ok:
        report["reason"] = reason
        _write_report(output_root, report)
        return report
    report["checked_image_sensitivity_canary"] = canary

    report["status"] = AssetStatus.VERIFIED_SEPARATE_ENV.value
    report["reason"] = "Molmo separate-env smoke and hidden-state validation artifacts satisfy the acceptance contract"
    _write_report(output_root, report)
    return report


def _check_shard(shard_path: Path, sidecar_path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not shard_path.is_file():
        return False, f"missing shard: {shard_path}", {}
    if not sidecar_path.is_file():
        return False, f"missing sidecar: {sidecar_path}", {}
    sidecar = _read_json(sidecar_path)
    config = sidecar.get("config") if isinstance(sidecar.get("config"), dict) else {}
    merged = {**config, **sidecar}
    missing = sorted(field for field in REQUIRED_SIDECAR_FIELDS if field not in merged or merged.get(field) in (None, "", "unknown"))
    if missing:
        return False, "sidecar missing required fields: " + ", ".join(missing), merged
    total_layers = int(merged["total_layers"])
    if list(merged["selected_layers"]) != list(range(total_layers)):
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
        if int(layer_vectors.shape[0]) != total_layers:
            return False, f"shard entry {index} layer count mismatch", merged
        if not torch.isfinite(layer_vectors).all().item():
            return False, f"shard entry {index} layer_vectors contain non-finite values", merged
        if list(entry.get("selected_layers", [])) != list(range(total_layers)):
            return False, f"shard entry {index} selected_layers mismatch", merged
    return True, "ok", merged


def _check_dataset_checksums(checksums: Mapping[str, Any], dataset: str, subset: str) -> tuple[bool, str]:
    determinism = checksums.get("determinism") if isinstance(checksums.get("determinism"), Mapping) else {}
    key = f"{MOLMO_ALIAS}/{dataset}/{subset}"
    payload = determinism.get(key)
    if not isinstance(payload, Mapping):
        return False, f"missing Molmo determinism checksum entry for {dataset}/{subset}"
    if payload.get("status") != "verified":
        return False, f"Molmo determinism checksum entry is not verified for {dataset}/{subset}: {payload.get('status')}"
    return True, "ok"


def _check_image_sensitivity_canary(checksums: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    canaries = checksums.get("image_sensitivity_canary") if isinstance(checksums.get("image_sensitivity_canary"), Mapping) else {}
    payload = canaries.get(MOLMO_ALIAS)
    if not isinstance(payload, Mapping):
        return False, "missing Molmo image sensitivity canary checksum entry", {}
    status = payload.get("status")
    if status == "verified":
        return True, "ok", dict(payload)
    if status == "skipped_with_reason" and payload.get("reason"):
        return True, "ok", dict(payload)
    return False, f"Molmo image sensitivity canary is not accepted: {status}", dict(payload)


def _verified_row(rows: list[dict[str, str]], dataset: str, subset: str) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("model_alias") == MOLMO_ALIAS
            and row.get("dataset") == dataset
            and row.get("subset") == subset
            and row.get("status") == AssetStatus.VERIFIED.value
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


def _nested_mapping(payload: object, key: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


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


def _write_report(output_root: Path, report: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "molmo_separate_env_acceptance.json"
    md_path = output_root / "MOLMO_SEPARATE_ENV_ACCEPTANCE.md"
    payload = dict(report)
    payload["manifest_json"] = str(json_path)
    payload["manifest_markdown"] = str(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Molmo Separate-Env Acceptance",
                "",
                f"- status: {payload['status']}",
                f"- reason: {payload['reason']}",
                f"- source_root: {payload['source_root']}",
                "- copied_tensors: false",
                "",
                "This accepts Molmo as `verified_separate_env`; it does not call Molmo main-env verified.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accept_separate_env(model=args.model, source_root=args.source_root, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
