"""v2 Stage 0 dataset and cache audit tables."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .dataset import DatasetSpec, NormalizedRecord, load_dataset_records
from .metadata import (
    DATASET_AUDIT_COLUMNS,
    LABEL_BALANCE_COLUMNS,
    OBJECT_NAME_AUDIT_COLUMNS,
    SAMPLE_OVERLAP_AUDIT_COLUMNS,
    UNKNOWN_OBJECT_NAME,
)
from .utils import parse_yes_no_label, write_csv


@dataclass(frozen=True)
class CacheAnswer:
    parsed_answer: int | None
    has_parsed_answer_field: bool
    has_answer_text: bool
    has_first_token_logits: bool


@dataclass(frozen=True)
class AuditResult:
    audit_dir: Path
    dataset_audit_rows: list[dict[str, object | None]]
    label_balance_rows: list[dict[str, object | None]]
    object_name_audit_rows: list[dict[str, object | None]]
    sample_overlap_audit_rows: list[dict[str, object | None]]


def validate_required_datasets(
    specs: Sequence[DatasetSpec],
    required: Iterable[tuple[str, str]],
) -> None:
    by_key = {(spec.dataset_name, spec.subset): spec for spec in specs}
    for dataset_name, subset in required:
        spec = by_key.get((dataset_name, subset))
        if spec is None or not spec.path.exists():
            path_text = "<not configured>" if spec is None else str(spec.path)
            raise FileNotFoundError(
                f"Required dataset is missing: {dataset_name}/{subset} at {path_text}"
            )


def run_audit(
    specs: Sequence[DatasetSpec],
    *,
    output_root: Path | str,
    cache_root: Path | str | None,
) -> AuditResult:
    output_root = Path(output_root)
    audit_dir = output_root / "audit"
    cache_answers = load_cache_answers(Path(cache_root)) if cache_root is not None else None

    records_by_key: dict[tuple[str, str], list[NormalizedRecord]] = {}
    dataset_rows: list[dict[str, object | None]] = []
    label_rows: list[dict[str, object | None]] = []
    object_rows: list[dict[str, object | None]] = []

    for spec in specs:
        if not spec.path.exists():
            dataset_rows.append(_missing_dataset_row(spec))
            label_rows.append(_missing_label_balance_row(spec))
            records_by_key[(spec.dataset_name, spec.subset)] = []
            continue

        records = load_dataset_records(spec)
        records_by_key[(spec.dataset_name, spec.subset)] = records
        dataset_rows.append(_dataset_audit_row(spec, records))
        label_rows.append(_label_balance_row(spec, records, cache_answers))
        object_rows.extend(_object_name_rows(spec, records, cache_answers))

    overlap_rows = _overlap_rows(records_by_key)

    write_csv(audit_dir / "dataset_audit.csv", dataset_rows, DATASET_AUDIT_COLUMNS)
    write_csv(audit_dir / "label_balance.csv", label_rows, LABEL_BALANCE_COLUMNS)
    write_csv(audit_dir / "object_name_audit.csv", object_rows, OBJECT_NAME_AUDIT_COLUMNS)
    write_csv(audit_dir / "sample_overlap_audit.csv", overlap_rows, SAMPLE_OVERLAP_AUDIT_COLUMNS)

    return AuditResult(
        audit_dir=audit_dir,
        dataset_audit_rows=dataset_rows,
        label_balance_rows=label_rows,
        object_name_audit_rows=object_rows,
        sample_overlap_audit_rows=overlap_rows,
    )


def load_cache_answers(cache_root: Path) -> dict[str, CacheAnswer]:
    if not cache_root.exists():
        return {}

    import torch

    answers: dict[str, CacheAnswer] = {}
    for shard_path in sorted(cache_root.rglob("*.pt")):
        payload = torch.load(shard_path, weights_only=False)
        for entry in _iter_cache_entries(payload):
            sample_id = entry.get("sample_id")
            if sample_id is None or not str(sample_id).strip():
                continue
            has_parsed = "parsed_answer" in entry
            parsed = parse_yes_no_label(entry.get("parsed_answer")) if has_parsed else None
            answers[str(sample_id).strip()] = CacheAnswer(
                parsed_answer=parsed,
                has_parsed_answer_field=has_parsed,
                has_answer_text="answer_text" in entry,
                has_first_token_logits="first_token_logits" in entry,
            )
    return answers


def _iter_cache_entries(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if isinstance(payload, Mapping):
        for key in ("entries", "records", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        yield item
                return
        for sample_id, value in payload.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("sample_id", sample_id)
                yield row


def _missing_dataset_row(spec: DatasetSpec) -> dict[str, object | None]:
    return {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "path": str(spec.path),
        "status": "missing",
        "num_records": 0,
        "num_label_yes": 0,
        "num_label_no": 0,
        "num_missing_image_path": 0,
        "num_missing_question": 0,
        "num_unknown_object": 0,
        "unique_objects": 0,
    }


def _dataset_audit_row(spec: DatasetSpec, records: Sequence[NormalizedRecord]) -> dict[str, object | None]:
    labels = Counter(record.label for record in records)
    objects = {record.object_name for record in records}
    return {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "path": str(spec.path),
        "status": "present",
        "num_records": len(records),
        "num_label_yes": labels[1],
        "num_label_no": labels[0],
        "num_missing_image_path": sum(1 for record in records if not record.image_path),
        "num_missing_question": sum(1 for record in records if not record.question),
        "num_unknown_object": sum(1 for record in records if record.object_name == UNKNOWN_OBJECT_NAME),
        "unique_objects": len(objects),
    }


def _missing_label_balance_row(spec: DatasetSpec) -> dict[str, object | None]:
    return {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "status": "missing",
        "num_records": 0,
        "num_gt_yes": 0,
        "num_gt_no": 0,
        "num_parsed_yes": None,
        "num_parsed_no": None,
        "num_parsed_none": None,
        "num_correct": None,
        "num_hallucination": None,
        "hallucination_rate": None,
        "parsed_answer_status": "missing",
    }


def _label_balance_row(
    spec: DatasetSpec,
    records: Sequence[NormalizedRecord],
    cache_answers: Mapping[str, CacheAnswer] | None,
) -> dict[str, object | None]:
    labels = Counter(record.label for record in records)
    row: dict[str, object | None] = {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "status": "present",
        "num_records": len(records),
        "num_gt_yes": labels[1],
        "num_gt_no": labels[0],
    }
    if cache_answers is None:
        row.update(_blank_parsed_metrics("not_available_before_cache"))
        return row

    parsed_values = [_cache_answer_for(record, cache_answers) for record in records]
    if not any(value is not None and value.has_parsed_answer_field for value in parsed_values):
        row.update(_blank_parsed_metrics("not_available_in_cache"))
        return row

    parsed_labels = [None if value is None else value.parsed_answer for value in parsed_values]
    num_hallucination = sum(
        1
        for record, parsed in zip(records, parsed_labels)
        if parsed == 1 and record.label == 0
    )
    row.update(
        {
            "num_parsed_yes": sum(1 for value in parsed_labels if value == 1),
            "num_parsed_no": sum(1 for value in parsed_labels if value == 0),
            "num_parsed_none": sum(1 for value in parsed_labels if value is None),
            "num_correct": sum(
                1
                for record, parsed in zip(records, parsed_labels)
                if parsed is not None and parsed == record.label
            ),
            "num_hallucination": num_hallucination,
            "hallucination_rate": _format_rate(num_hallucination, labels[0]),
            "parsed_answer_status": "available",
        }
    )
    return row


def _blank_parsed_metrics(status: str) -> dict[str, object | None]:
    return {
        "num_parsed_yes": None,
        "num_parsed_no": None,
        "num_parsed_none": None,
        "num_correct": None,
        "num_hallucination": None,
        "hallucination_rate": None,
        "parsed_answer_status": status,
    }


def _object_name_rows(
    spec: DatasetSpec,
    records: Sequence[NormalizedRecord],
    cache_answers: Mapping[str, CacheAnswer] | None,
) -> list[dict[str, object | None]]:
    rows: list[dict[str, object | None]] = []
    for object_name in sorted({record.object_name for record in records}):
        object_records = [record for record in records if record.object_name == object_name]
        row: dict[str, object | None] = {
            "dataset_name": spec.dataset_name,
            "subset": spec.subset,
            "object_name": object_name,
            "num_records": len(object_records),
            "num_correct": None,
            "num_hallucination": None,
        }
        if cache_answers is not None:
            parsed_values = [_cache_answer_for(record, cache_answers) for record in object_records]
            if any(value is not None and value.has_parsed_answer_field for value in parsed_values):
                parsed_labels = [None if value is None else value.parsed_answer for value in parsed_values]
                row["num_correct"] = sum(
                    1
                    for record, parsed in zip(object_records, parsed_labels)
                    if parsed is not None and parsed == record.label
                )
                row["num_hallucination"] = sum(
                    1
                    for record, parsed in zip(object_records, parsed_labels)
                    if parsed == 1 and record.label == 0
                )
        rows.append(row)
    return rows


def _cache_answer_for(
    record: NormalizedRecord,
    cache_answers: Mapping[str, CacheAnswer],
) -> CacheAnswer | None:
    return cache_answers.get(record.sample_id)


def _format_rate(numerator: int, denominator: int) -> str | None:
    if denominator <= 0:
        return None
    return f"{numerator / denominator:.6f}".rstrip("0").rstrip(".")


def _overlap_rows(
    records_by_key: Mapping[tuple[str, str], Sequence[NormalizedRecord]],
) -> list[dict[str, object | None]]:
    rows: list[dict[str, object | None]] = []
    for (left_dataset, left_subset), left_records in sorted(records_by_key.items()):
        if left_dataset != "pope" or not left_records:
            continue
        for (right_dataset, right_subset), right_records in sorted(records_by_key.items()):
            if right_dataset != "repope" or not right_records:
                continue
            for overlap_key, left_values, right_values in _overlap_value_sets(left_records, right_records):
                overlap_count = len(left_values & right_values)
                if overlap_count <= 0:
                    continue
                rows.append(
                    {
                        "overlap_key": overlap_key,
                        "left_dataset": left_dataset,
                        "right_dataset": right_dataset,
                        "left_subset": left_subset,
                        "right_subset": right_subset,
                        "overlap_count": overlap_count,
                    }
                )
    return rows


def _overlap_value_sets(
    left_records: Sequence[NormalizedRecord],
    right_records: Sequence[NormalizedRecord],
) -> Iterable[tuple[str, set[object], set[object]]]:
    yield "sample_id", _value_set(left_records, "sample_id"), _value_set(right_records, "sample_id")
    yield "image_id", _value_set(left_records, "image_id"), _value_set(right_records, "image_id")
    yield "image_path", _value_set(left_records, "image_path"), _value_set(right_records, "image_path")
    yield (
        "image_id_object_name",
        {(record.image_id, record.object_name) for record in left_records if record.image_id},
        {(record.image_id, record.object_name) for record in right_records if record.image_id},
    )
    yield (
        "sample_id_object_name",
        {(record.sample_id, record.object_name) for record in left_records if record.sample_id},
        {(record.sample_id, record.object_name) for record in right_records if record.sample_id},
    )


def _value_set(records: Sequence[NormalizedRecord], field_name: str) -> set[object]:
    return {
        getattr(record, field_name)
        for record in records
        if getattr(record, field_name)
    }
