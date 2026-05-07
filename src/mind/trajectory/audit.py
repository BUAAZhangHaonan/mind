"""v2 Stage 0 dataset and cache audit tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .dataset import DatasetSpec, NormalizedRecord, load_dataset_records
from .metadata import (
    CACHE_LABEL_BALANCE_COLUMNS,
    DATASET_AUDIT_COLUMNS,
    LABEL_BALANCE_COLUMNS,
    OBJECT_NAME_AUDIT_COLUMNS,
    SAMPLE_OVERLAP_AUDIT_COLUMNS,
    UNKNOWN_OBJECT_NAME,
)
from .utils import parse_yes_no_label, write_csv

CacheKey = tuple[str, str, str, str]
RecordCacheKey = tuple[str, str, str]


@dataclass(frozen=True)
class CacheAnswer:
    model_name: str | None
    dataset_name: str | None
    subset: str | None
    sample_id: str
    parsed_answer: int | None
    has_parsed_answer_field: bool
    has_answer_text: bool
    has_first_token_logits: bool


@dataclass(frozen=True)
class CacheAnswerIndex:
    by_identity: Mapping[CacheKey, CacheAnswer]
    by_record_identity: Mapping[RecordCacheKey, tuple[CacheAnswer, ...]]
    legacy_by_sample_id: Mapping[str, tuple[CacheAnswer, ...]]


@dataclass(frozen=True)
class AuditResult:
    audit_dir: Path
    dataset_audit_rows: list[dict[str, object | None]]
    label_balance_rows: list[dict[str, object | None]]
    cache_label_balance_rows: list[dict[str, object | None]]
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
    present_by_key: dict[tuple[str, str], bool] = {}
    dataset_rows: list[dict[str, object | None]] = []
    label_rows: list[dict[str, object | None]] = []
    cache_label_rows: list[dict[str, object | None]] = []
    object_rows: list[dict[str, object | None]] = []

    for spec in specs:
        key = (spec.dataset_name, spec.subset)
        if not spec.path.exists():
            dataset_rows.append(_missing_dataset_row(spec))
            records_by_key[key] = []
            present_by_key[key] = False
            continue

        records = load_dataset_records(spec)
        records_by_key[key] = records
        present_by_key[key] = True
        dataset_rows.append(_dataset_audit_row(spec, records))

    ambiguous_sample_ids = _ambiguous_sample_ids(records_by_key.values())
    for spec in specs:
        key = (spec.dataset_name, spec.subset)
        records = records_by_key[key]
        if not present_by_key[key]:
            label_rows.append(_missing_label_balance_row(spec))
            continue
        label_rows.append(_label_balance_row(spec, records, cache_answers, ambiguous_sample_ids))
        object_rows.extend(_object_name_rows(spec, records, cache_answers, ambiguous_sample_ids))

    overlap_rows = _overlap_rows(records_by_key)

    write_csv(audit_dir / "dataset_audit.csv", dataset_rows, DATASET_AUDIT_COLUMNS)
    write_csv(audit_dir / "label_balance.csv", label_rows, LABEL_BALANCE_COLUMNS)
    if cache_answers is not None:
        cache_label_rows = _cache_label_balance_rows(
            specs,
            records_by_key,
            cache_answers,
            ambiguous_sample_ids,
            model_names=None,
        )
        write_csv(
            audit_dir / "cache_label_balance.csv",
            cache_label_rows,
            CACHE_LABEL_BALANCE_COLUMNS,
        )
    write_csv(audit_dir / "object_name_audit.csv", object_rows, OBJECT_NAME_AUDIT_COLUMNS)
    write_csv(audit_dir / "sample_overlap_audit.csv", overlap_rows, SAMPLE_OVERLAP_AUDIT_COLUMNS)

    return AuditResult(
        audit_dir=audit_dir,
        dataset_audit_rows=dataset_rows,
        label_balance_rows=label_rows,
        cache_label_balance_rows=cache_label_rows,
        object_name_audit_rows=object_rows,
        sample_overlap_audit_rows=overlap_rows,
    )


def build_cache_label_balance_rows(
    specs: Sequence[DatasetSpec],
    *,
    cache_root: Path | str,
    model_names: Sequence[str] | None = None,
) -> list[dict[str, object | None]]:
    cache_answers = load_cache_answers(Path(cache_root))
    records_by_key = {
        (spec.dataset_name, spec.subset): load_dataset_records(spec)
        for spec in specs
        if spec.path.exists()
    }
    ambiguous_sample_ids = _ambiguous_sample_ids(records_by_key.values())
    return _cache_label_balance_rows(
        specs,
        records_by_key,
        cache_answers,
        ambiguous_sample_ids,
        model_names=model_names,
    )


def _cache_label_balance_rows(
    specs: Sequence[DatasetSpec],
    records_by_key: Mapping[tuple[str, str], Sequence[NormalizedRecord]],
    cache_answers: CacheAnswerIndex,
    ambiguous_sample_ids: set[str],
    *,
    model_names: Sequence[str] | None,
) -> list[dict[str, object | None]]:
    rows: list[dict[str, object | None]] = []
    scoped_models = list(model_names or _cache_model_names(cache_answers))
    if scoped_models:
        for model_name in scoped_models:
            for spec in specs:
                records = records_by_key.get((spec.dataset_name, spec.subset))
                if records is None:
                    rows.append(_missing_cache_label_balance_row(spec, model_name=model_name))
                    continue
                rows.append(
                    _cache_label_balance_row(
                        spec,
                        records,
                        cache_answers,
                        ambiguous_sample_ids,
                        model_name=model_name,
                    )
                )
        return rows

    for spec in specs:
        records = records_by_key.get((spec.dataset_name, spec.subset))
        if records is None:
            rows.append(_missing_cache_label_balance_row(spec, model_name=None))
            continue
        rows.append(
            _cache_label_balance_row(
                spec,
                records,
                cache_answers,
                ambiguous_sample_ids,
                model_name=None,
            )
        )
    return rows


def _cache_model_names(cache_answers: CacheAnswerIndex) -> list[str]:
    return sorted(
        {
            answer.model_name
            for answer in cache_answers.by_identity.values()
            if answer.model_name is not None
        }
    )


def load_cache_answers(cache_root: Path) -> CacheAnswerIndex:
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")
    if not cache_root.is_dir():
        raise NotADirectoryError(f"Cache root is not a directory: {cache_root}")

    import torch

    by_identity: dict[CacheKey, CacheAnswer] = {}
    by_record_identity: dict[RecordCacheKey, list[CacheAnswer]] = defaultdict(list)
    legacy_by_sample_id: dict[str, list[CacheAnswer]] = defaultdict(list)
    for shard_path in sorted(cache_root.rglob("*.pt")):
        if not shard_path.is_file():
            continue
        sidecar = _load_cache_sidecar(shard_path)
        payload = torch.load(shard_path, weights_only=False)
        for entry in _iter_cache_entries(payload):
            sample_id = _identity_text(entry.get("sample_id"))
            if sample_id is None:
                continue
            model_name = _identity_text(entry.get("model_name")) or _identity_text(sidecar.get("model_name"))
            dataset_name = (
                _identity_text(entry.get("dataset_name") or entry.get("source_dataset"))
                or _identity_text(sidecar.get("dataset_name") or sidecar.get("source_dataset"))
            )
            subset = (
                _identity_text(entry.get("subset") or entry.get("split"))
                or _identity_text(sidecar.get("subset") or sidecar.get("split"))
            )
            has_parsed = "parsed_answer" in entry
            parsed = parse_yes_no_label(entry.get("parsed_answer")) if has_parsed else None
            answer = CacheAnswer(
                model_name=model_name,
                dataset_name=dataset_name,
                subset=subset,
                sample_id=sample_id,
                parsed_answer=parsed,
                has_parsed_answer_field=has_parsed,
                has_answer_text="answer_text" in entry,
                has_first_token_logits="first_token_logits" in entry,
            )
            if model_name is not None and dataset_name is not None and subset is not None:
                by_identity[(model_name, dataset_name, subset, sample_id)] = answer
                by_record_identity[(dataset_name, subset, sample_id)].append(answer)
            elif dataset_name is not None and subset is not None:
                by_record_identity[(dataset_name, subset, sample_id)].append(answer)
            elif dataset_name is None and subset is None:
                legacy_by_sample_id[sample_id].append(answer)
    return CacheAnswerIndex(
        by_identity=by_identity,
        by_record_identity={
            key: tuple(answers)
            for key, answers in by_record_identity.items()
        },
        legacy_by_sample_id={
            sample_id: tuple(answers)
            for sample_id, answers in legacy_by_sample_id.items()
        },
    )


def _load_cache_sidecar(shard_path: Path) -> Mapping[str, object]:
    sidecar_path = Path(str(shard_path) + ".json")
    if not sidecar_path.exists():
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _identity_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        "unique_images": 0,
        "num_duplicate_sample_id": 0,
        "num_null_required_fields": 0,
        "num_invalid_label": 0,
    }


def _dataset_audit_row(spec: DatasetSpec, records: Sequence[NormalizedRecord]) -> dict[str, object | None]:
    labels = Counter(record.label for record in records)
    objects = {record.object_name for record in records}
    sample_counts = Counter(record.sample_id for record in records if record.sample_id)
    image_keys = {_image_key(record) for record in records if _image_key(record)}
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
        "unique_images": len(image_keys),
        "num_duplicate_sample_id": sum(count - 1 for count in sample_counts.values() if count > 1),
        "num_null_required_fields": sum(_null_required_field_count(record) for record in records),
        "num_invalid_label": labels[None],
    }


def _missing_label_balance_row(
    spec: DatasetSpec,
    *,
    model_name: str | None = None,
) -> dict[str, object | None]:
    row: dict[str, object | None] = {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "status": "missing",
        "num_records": 0,
        "num_gt_yes": 0,
        "num_gt_no": 0,
        "num_invalid_label": 0,
        "num_parsed_yes": None,
        "num_parsed_no": None,
        "num_parsed_none": None,
        "num_correct": None,
        "num_hallucination": None,
        "hallucination_rate": None,
        "parsed_answer_status": "missing",
    }
    if model_name is not None:
        row["model_name"] = model_name
    return row


def _missing_cache_label_balance_row(
    spec: DatasetSpec,
    *,
    model_name: str | None,
) -> dict[str, object | None]:
    return {
        "model_name": model_name,
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "num_entries": 0,
        "num_gt_yes": 0,
        "num_gt_no": 0,
        "num_parsed_yes": 0,
        "num_parsed_no": 0,
        "num_parsed_none": 0,
        "num_correct": 0,
        "num_hard_hallucination": 0,
        "num_false_negative_error": 0,
        "num_primary_population": 0,
        "hallucination_rate_in_primary_population": None,
    }


def _cache_label_balance_row(
    spec: DatasetSpec,
    records: Sequence[NormalizedRecord],
    cache_answers: CacheAnswerIndex,
    ambiguous_sample_ids: set[str],
    *,
    model_name: str | None,
) -> dict[str, object | None]:
    labels = Counter(record.label for record in records)
    parsed_labels = [
        None
        if (answer := _cache_answer_for(record, cache_answers, ambiguous_sample_ids, model_name=model_name))
        is None
        else answer.parsed_answer
        for record in records
    ]
    num_hard_hallucination = sum(
        1
        for record, parsed in zip(records, parsed_labels)
        if record.label == 0 and parsed == 1
    )
    num_false_negative_error = sum(
        1
        for record, parsed in zip(records, parsed_labels)
        if record.label == 1 and parsed == 0
    )
    primary_population = labels[0]
    return {
        "model_name": model_name,
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "num_entries": len(records),
        "num_gt_yes": labels[1],
        "num_gt_no": labels[0],
        "num_parsed_yes": sum(1 for value in parsed_labels if value == 1),
        "num_parsed_no": sum(1 for value in parsed_labels if value == 0),
        "num_parsed_none": sum(1 for value in parsed_labels if value is None),
        "num_correct": sum(
            1
            for record, parsed in zip(records, parsed_labels)
            if parsed is not None and parsed == record.label
        ),
        "num_hard_hallucination": num_hard_hallucination,
        "num_false_negative_error": num_false_negative_error,
        "num_primary_population": primary_population,
        "hallucination_rate_in_primary_population": _format_rate(
            num_hard_hallucination,
            primary_population,
        ),
    }


def _label_balance_row(
    spec: DatasetSpec,
    records: Sequence[NormalizedRecord],
    cache_answers: CacheAnswerIndex | None,
    ambiguous_sample_ids: set[str],
    *,
    model_name: str | None = None,
) -> dict[str, object | None]:
    labels = Counter(record.label for record in records)
    row: dict[str, object | None] = {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "status": "present",
        "num_records": len(records),
        "num_gt_yes": labels[1],
        "num_gt_no": labels[0],
        "num_invalid_label": labels[None],
    }
    if model_name is not None:
        row["model_name"] = model_name
    if cache_answers is None:
        row.update(_blank_parsed_metrics("not_available_before_cache"))
        return row

    parsed_values = [
        _cache_answer_for(record, cache_answers, ambiguous_sample_ids, model_name=model_name)
        for record in records
    ]
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
    cache_answers: CacheAnswerIndex | None,
    ambiguous_sample_ids: set[str],
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
            parsed_values = [
                _cache_answer_for(record, cache_answers, ambiguous_sample_ids, model_name=None)
                for record in object_records
            ]
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
    cache_answers: CacheAnswerIndex,
    ambiguous_sample_ids: set[str],
    *,
    model_name: str | None,
) -> CacheAnswer | None:
    if model_name is not None:
        return cache_answers.by_identity.get(
            (model_name, record.dataset_name, record.subset, record.sample_id)
        )
    exact_answers = cache_answers.by_record_identity.get(
        (record.dataset_name, record.subset, record.sample_id),
        (),
    )
    if len(exact_answers) == 1:
        return exact_answers[0]
    if record.sample_id in ambiguous_sample_ids:
        return None
    legacy_answers = cache_answers.legacy_by_sample_id.get(record.sample_id, ())
    if len(legacy_answers) == 1:
        return legacy_answers[0]
    return None


def _ambiguous_sample_ids(record_groups: Iterable[Sequence[NormalizedRecord]]) -> set[str]:
    counts: Counter[str] = Counter()
    for records in record_groups:
        counts.update(record.sample_id for record in records if record.sample_id)
    return {sample_id for sample_id, count in counts.items() if count > 1}


def _image_key(record: NormalizedRecord) -> str:
    return record.image_id or record.image_path


def _null_required_field_count(record: NormalizedRecord) -> int:
    return (
        int(not record.sample_id)
        + int(not record.image_path)
        + int(not record.question)
        + int(record.label is None)
    )


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
