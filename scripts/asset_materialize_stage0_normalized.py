#!/usr/bin/env python3
"""Materialize canonical Stage 0 normalized records from existing cache metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)


SOURCE_MODEL = "qwen3-vl-8b"
VALIDATION_MODEL = "internvl3.5-8b"
DATASET_SUBSETS: tuple[tuple[str, str], ...] = (
    ("pope", "popular"),
    ("pope", "random"),
    ("pope", "adversarial"),
    ("repope", "popular"),
    ("repope", "random"),
    ("repope", "adversarial"),
    ("dash-b", "all"),
)
REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
    "source_dataset",
    "split",
    "subset",
)
OPTIONAL_RECORD_FIELDS = ("dataset_name", "prompt_template_id")
CONSISTENCY_FIELDS = (
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
    "source_dataset",
    "subset",
)
REPORT_FIELDS = (
    "dataset_name",
    "subset",
    "output_path",
    "status",
    "num_qwen_records",
    "num_internvl_records",
    "num_written_records",
    "num_mismatches",
    "mismatch_report_path",
    "action",
)
ALLOWED_STATUSES = {
    "written",
    "already_exists_identical",
    "failed_missing_cache",
    "failed_mismatched_records",
    "failed_existing_file_differs",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def materialize_all(*, stage0_root: Path, output_root: Path, overwrite: bool = False) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        materialize_dataset_subset(
            stage0_root=stage0_root,
            output_root=output_root,
            dataset_name=dataset_name,
            subset=subset,
            overwrite=overwrite,
        )
        for dataset_name, subset in DATASET_SUBSETS
    ]
    write_report(output_root / "stage0_normalized_materialization_report.csv", rows)
    manifest = {
        "stage0_root": str(stage0_root),
        "output_root": str(output_root),
        "source_model": SOURCE_MODEL,
        "validation_model": VALIDATION_MODEL,
        "overwrite": bool(overwrite),
        "datasets": rows,
        "status": "passed"
        if all(str(row["status"]) in {"written", "already_exists_identical"} for row in rows)
        else "blocked",
    }
    write_json(output_root / "stage0_normalized_materialization_manifest.json", manifest)
    return rows


def materialize_dataset_subset(
    *,
    stage0_root: Path,
    output_root: Path,
    dataset_name: str,
    subset: str,
    overwrite: bool,
) -> dict[str, object]:
    output_path = normalized_output_path(stage0_root, dataset_name, subset)
    mismatch_report_path = output_root / "stage0_normalized_mismatches" / dataset_name / f"{subset}.jsonl"
    base = {
        "dataset_name": dataset_name,
        "subset": subset,
        "output_path": str(output_path),
        "status": "failed_missing_cache",
        "num_qwen_records": 0,
        "num_internvl_records": 0,
        "num_written_records": 0,
        "num_mismatches": 0,
        "mismatch_report_path": "",
        "action": "none",
    }
    qwen_dir = cache_subset_dir(stage0_root, SOURCE_MODEL, dataset_name, subset)
    internvl_dir = cache_subset_dir(stage0_root, VALIDATION_MODEL, dataset_name, subset)
    if not qwen_dir.is_dir() or not internvl_dir.is_dir():
        return {
            **base,
            "status": "failed_missing_cache",
            "action": f"missing cache dir: {qwen_dir if not qwen_dir.is_dir() else internvl_dir}",
        }

    qwen_records = load_cache_records(qwen_dir)
    internvl_records = load_cache_records(internvl_dir)
    base["num_qwen_records"] = len(qwen_records)
    base["num_internvl_records"] = len(internvl_records)
    if not qwen_records or not internvl_records:
        return {**base, "status": "failed_missing_cache", "action": "no cache records found"}

    mismatches = compare_record_sets(qwen_records, internvl_records)
    if mismatches:
        write_jsonl(mismatch_report_path, mismatches)
        return {
            **base,
            "status": "failed_mismatched_records",
            "num_mismatches": len(mismatches),
            "mismatch_report_path": str(mismatch_report_path),
            "action": "wrote mismatch report",
        }

    materialized = [
        normalize_cache_entry(qwen_records[sample_id])
        for sample_id in sorted(qwen_records, key=sample_sort_key)
    ]
    base["num_written_records"] = len(materialized)
    if output_path.exists():
        existing = load_jsonl(output_path)
        if existing == materialized:
            return {
                **base,
                "status": "already_exists_identical",
                "action": "preserved existing file",
            }
        if not overwrite:
            return {
                **base,
                "status": "failed_existing_file_differs",
                "action": "existing file differs; rerun with --overwrite to replace",
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, materialized)
    return {
        **base,
        "status": "written",
        "action": "overwrote existing file" if output_path.exists() and overwrite else "wrote normalized records",
    }


def cache_subset_dir(stage0_root: Path, model_name: str, dataset_name: str, subset: str) -> Path:
    return stage0_root / "cache" / model_name / dataset_name / subset


def normalized_output_path(stage0_root: Path, dataset_name: str, subset: str) -> Path:
    return stage0_root / "normalized" / dataset_name / f"{subset}.jsonl"


def load_cache_records(cache_dir: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    shard_paths = sorted(cache_dir.glob("shard-*.pt"))
    for shard_path in shard_paths:
        entries = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(entries, list):
            raise ValueError(f"cache shard payload must be a list: {shard_path}")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"cache shard entry must be a mapping: {shard_path}")
            sample_id = str(entry.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"cache entry missing sample_id: {shard_path}")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {cache_dir}")
            records[sample_id] = normalize_cache_entry(entry)
    return records


def compare_record_sets(
    qwen_records: Mapping[str, Mapping[str, object]],
    internvl_records: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    qwen_ids = set(qwen_records)
    internvl_ids = set(internvl_records)
    for sample_id in sorted(qwen_ids - internvl_ids, key=sample_sort_key):
        mismatches.append({"sample_id": sample_id, "mismatch_type": "missing_in_internvl"})
    for sample_id in sorted(internvl_ids - qwen_ids, key=sample_sort_key):
        mismatches.append({"sample_id": sample_id, "mismatch_type": "missing_in_qwen"})
    for sample_id in sorted(qwen_ids & internvl_ids, key=sample_sort_key):
        qwen = qwen_records[sample_id]
        internvl = internvl_records[sample_id]
        for field in CONSISTENCY_FIELDS:
            if qwen.get(field) != internvl.get(field):
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "mismatch_type": "field_mismatch",
                        "field": field,
                        "qwen_value": qwen.get(field),
                        "internvl_value": internvl.get(field),
                    }
                )
    return mismatches


def normalize_cache_entry(entry: Mapping[str, object]) -> dict[str, object]:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"cache entry missing required normalized fields: {', '.join(missing)}")
    row = {field: entry[field] for field in REQUIRED_RECORD_FIELDS}
    for field in OPTIONAL_RECORD_FIELDS:
        if field in entry and entry[field] is not None:
            row[field] = entry[field]
    return row


def sample_sort_key(sample_id: object) -> tuple[int, object]:
    text = str(sample_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: normalized row must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            status = str(row.get("status", ""))
            if status not in ALLOWED_STATUSES:
                raise ValueError(f"invalid materialization status: {status}")
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = materialize_all(
            stage0_root=args.stage0_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    failures = [row for row in rows if str(row["status"]).startswith("failed_")]
    print(
        f"stage0_normalized_materialization datasets={len(rows)} failures={len(failures)} "
        f"output_root={args.output_root}"
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
