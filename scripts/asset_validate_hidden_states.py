#!/usr/bin/env python3
"""Validate Experiment 1 smoke hidden-state cache shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.asset_validation import (
    AssetStatus,
    ValidationResult,
    build_completion_summary,
    tensor_checksum,
    validate_hidden_state_entries,
    validate_smoke_report_contract,
)
from mind.models.registry import REQUIRED_MODEL_ALIASES


REPORT_FIELDS = (
    "model_alias",
    "dataset",
    "subset",
    "status",
    "reason",
    "shard_path",
    "num_entries",
)
SMOKE_DATASETS = ("pope", "repope", "dash-b")
EXPECTED_DATASET_SUBSETS = (("pope", "popular"), ("repope", "popular"), ("dash-b", "all"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-cache-root", type=Path, required=True)
    return parser


def run_validation(*, output_root: Path, smoke_cache_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    if not smoke_cache_root.is_dir():
        raise FileNotFoundError(f"--smoke-cache-root does not exist or is not a directory: {smoke_cache_root}")
    smoke_rows = read_csv(output_root / "smoke_extraction_report.csv")
    contract = validate_smoke_report_contract(smoke_rows, datasets=SMOKE_DATASETS)
    validation_rows: list[dict[str, object]] = []
    checksums = read_json(output_root / "validation_checksums.json", default={})
    checksums.setdefault("structural", {})

    if contract.status != "verified":
        reason = f"smoke report contract failed: {contract.reason}"
        validation_rows.extend(_contract_failure_rows(reason))
        model_statuses, model_reasons = aggregate_model_statuses(validation_rows)
        summary = build_completion_summary(
            model_statuses=model_statuses,
            model_reasons=model_reasons,
            smoke_datasets=SMOKE_DATASETS,
            smoke_limit=2,
            tests_run=[
                "conda run --no-capture-output -n mind-py311 python -m pytest -q tests/assets",
                "conda run --no-capture-output -n mind-py311 python -m pytest -q tests/stage0 tests/stage_a tests/assets",
            ],
            git_commit=get_git_commit(),
        )
        _write_csv(output_root / "hidden_state_validation_report.csv", validation_rows, REPORT_FIELDS)
        _write_json(output_root / "validation_checksums.json", checksums)
        _write_json(output_root / "asset_completion_summary.json", summary)
        write_markdown_report(output_root / "ASSET_COMPLETION_REPORT.md", summary)
        print(f"asset_validate_hidden_states rows={len(validation_rows)} final_status={summary['final_status']}")
        return 0

    for smoke_row in smoke_rows:
        if smoke_row.get("status") != AssetStatus.VERIFIED.value:
            validation_rows.append(
                {
                    "model_alias": smoke_row.get("model_alias", ""),
                    "dataset": smoke_row.get("dataset", ""),
                    "subset": smoke_row.get("subset", ""),
                    "status": smoke_row.get("status", AssetStatus.BLOCKED.value),
                    "reason": smoke_row.get("reason", ""),
                    "shard_path": smoke_row.get("shard_path", ""),
                    "num_entries": 0,
                }
            )
            continue
        shard_path = Path(str(smoke_row["shard_path"]))
        sidecar_path = Path(str(shard_path) + ".json")
        try:
            expected_shard_path = (
                smoke_cache_root
                / str(smoke_row["model_alias"])
                / str(smoke_row["dataset"])
                / str(smoke_row["subset"])
                / "shard-00000.pt"
            )
            if shard_path.resolve() != expected_shard_path.resolve():
                raise ValueError(f"shard path is not canonical under --smoke-cache-root: {shard_path}")
            entries = torch.load(shard_path, weights_only=False)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            result = validate_hidden_state_entries(entries, sidecar)
            status = result.status
            reason = result.reason
            key = f"{smoke_row['model_alias']}/{smoke_row['dataset']}/{smoke_row['subset']}"
            checksums["structural"][key] = [entry_checksums(entry) for entry in entries]
            checksum_result = validate_checksum_statuses(
                checksums,
                key=key,
                model_alias=str(smoke_row["model_alias"]),
            )
            if status == "verified" and checksum_result.status != "verified":
                status = AssetStatus.FAILED_VALIDATION.value
                reason = checksum_result.reason
        except Exception as error:
            status = AssetStatus.FAILED_VALIDATION.value
            reason = f"validation failed: {type(error).__name__}: {error}"
            entries = []
        validation_rows.append(
            {
                "model_alias": smoke_row.get("model_alias", ""),
                "dataset": smoke_row.get("dataset", ""),
                "subset": smoke_row.get("subset", ""),
                "status": status,
                "reason": reason,
                "shard_path": str(shard_path),
                "num_entries": len(entries),
            }
        )

    model_statuses, model_reasons = aggregate_model_statuses(validation_rows)
    summary = build_completion_summary(
        model_statuses=model_statuses,
        model_reasons=model_reasons,
        smoke_datasets=SMOKE_DATASETS,
        smoke_limit=2,
        tests_run=[
            "conda run --no-capture-output -n mind-py311 python -m pytest -q tests/assets",
            "conda run --no-capture-output -n mind-py311 python -m pytest -q tests/stage0 tests/stage_a tests/assets",
        ],
        git_commit=get_git_commit(),
    )
    _write_csv(output_root / "hidden_state_validation_report.csv", validation_rows, REPORT_FIELDS)
    _write_json(output_root / "validation_checksums.json", checksums)
    _write_json(output_root / "asset_completion_summary.json", summary)
    write_markdown_report(output_root / "ASSET_COMPLETION_REPORT.md", summary)
    print(f"asset_validate_hidden_states rows={len(validation_rows)} final_status={summary['final_status']}")
    return 0


def aggregate_model_statuses(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, str], dict[str, str]]:
    statuses: dict[str, str] = {}
    reasons: dict[str, str] = {}
    severity = {
        AssetStatus.FAILED_VALIDATION.value: 4,
        AssetStatus.BLOCKED.value: 3,
        AssetStatus.NOT_ATTEMPTED_DUE_TO_DEPENDENCY.value: 3,
        AssetStatus.UNSUPPORTED_BY_POLICY.value: 2,
        AssetStatus.UNSUPPORTED_BY_WRAPPER.value: 2,
        AssetStatus.VERIFIED.value: 1,
    }
    for alias in REQUIRED_MODEL_ALIASES:
        alias_rows = [row for row in rows if row.get("model_alias") == alias]
        if not alias_rows:
            statuses[alias] = AssetStatus.NOT_ATTEMPTED_DUE_TO_DEPENDENCY.value
            reasons[alias] = "no smoke validation rows were found"
            continue
        row_pairs = {(str(row.get("dataset")), str(row.get("subset"))) for row in alias_rows}
        expected_pairs = set(EXPECTED_DATASET_SUBSETS)
        if row_pairs == expected_pairs and all(row.get("status") == AssetStatus.VERIFIED.value for row in alias_rows):
            statuses[alias] = AssetStatus.VERIFIED.value
            reasons[alias] = ""
            continue
        worst = max((str(row.get("status")) for row in alias_rows), key=lambda status: severity.get(status, 0))
        statuses[alias] = worst
        reasons[alias] = "; ".join(sorted({str(row.get("reason", "")) for row in alias_rows if row.get("reason")}))
    return statuses, reasons


def _contract_failure_rows(reason: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alias in REQUIRED_MODEL_ALIASES:
        for dataset, subset in EXPECTED_DATASET_SUBSETS:
            rows.append(
                {
                    "model_alias": alias,
                    "dataset": dataset,
                    "subset": subset,
                    "status": AssetStatus.FAILED_VALIDATION.value,
                    "reason": reason,
                    "shard_path": "",
                    "num_entries": 0,
                }
            )
    return rows


def validate_checksum_statuses(
    checksums: Mapping[str, object],
    *,
    key: str,
    model_alias: str,
) -> ValidationResult:
    determinism = _nested_mapping(checksums, "determinism").get(key)
    if not isinstance(determinism, Mapping):
        return ValidationResult(AssetStatus.FAILED_VALIDATION.value, "determinism metadata missing")
    if determinism.get("status") != AssetStatus.VERIFIED.value:
        return ValidationResult(
            AssetStatus.FAILED_VALIDATION.value,
            f"determinism status is not verified: {determinism.get('status')}",
        )
    canary = _nested_mapping(checksums, "image_sensitivity_canary").get(model_alias)
    if not isinstance(canary, Mapping):
        return ValidationResult(AssetStatus.FAILED_VALIDATION.value, "image sensitivity canary metadata missing")
    canary_status = canary.get("status")
    if canary_status == AssetStatus.VERIFIED.value:
        return ValidationResult(AssetStatus.VERIFIED.value)
    if canary_status == "skipped_with_reason" and str(canary.get("reason", "")).strip():
        return ValidationResult(AssetStatus.VERIFIED.value)
    return ValidationResult(
        AssetStatus.FAILED_VALIDATION.value,
        f"image sensitivity canary status is invalid: {canary_status}",
    )


def _nested_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def entry_checksums(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "sample_id": entry.get("sample_id"),
        "layer_vectors_checksum": tensor_checksum(torch.as_tensor(entry["layer_vectors"])),
        "first_token_logits_checksum": tensor_checksum(torch.as_tensor(entry["first_token_logits"])),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path, *, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Asset Completion Report",
        "",
        f"Final status: {summary['final_status']}",
        "",
        f"Total models requested: {summary['total_models_requested']}",
        f"Verified: {summary['num_verified']}",
        f"Blocked: {summary['num_blocked']}",
        f"Unsupported by policy: {summary['num_unsupported_by_policy']}",
        f"Unsupported by wrapper: {summary['num_unsupported_by_wrapper']}",
        f"Failed validation: {summary['num_failed_validation']}",
        f"Not attempted due to dependency: {summary['num_not_attempted_due_to_dependency']}",
        "",
        "Stage A started: false",
        "Full cache extraction started: false",
        "Training started: false",
        "",
        "## Non-Verified Reasons",
    ]
    for group_key in ("blocked_reasons", "unsupported_reasons", "failed_reasons", "not_attempted_due_to_dependency_reasons"):
        reasons = summary.get(group_key, {})
        if isinstance(reasons, Mapping):
            for alias, reason in reasons.items():
                lines.append(f"- {alias}: {reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        return run_validation(output_root=args.output_root, smoke_cache_root=args.smoke_cache_root)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
