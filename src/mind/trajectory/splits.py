"""Deterministic grouped split construction for v2 Stage 0."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import random
from typing import Mapping, Sequence

from .utils import read_jsonl

SPLIT_NAMES = ("encoder_train", "bank", "cal", "test")
DEFAULT_RATIOS = (0.50, 0.20, 0.10, 0.20)
DEFAULT_SEED = 20260506
PRESERVED_FIELDS = (
    "dataset_name",
    "subset",
    "sample_id",
    "image_id",
    "image_path",
    "object_name",
    "label",
    "question",
)


def build_split_manifest(
    *,
    dataset_name: str,
    subset: str,
    input_records: Path | str,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    group_key: str = "image_id",
) -> dict[str, object]:
    """Build a split manifest from one normalized dataset/subset JSONL file."""

    input_path = Path(input_records)
    ratio_values = _validate_ratios(ratios)
    rows = read_jsonl(input_path)
    _validate_single_source(rows, dataset_name=dataset_name, subset=subset)
    grouped = _group_rows(rows, group_key=group_key)
    group_to_split = _assign_groups(grouped, ratios=ratio_values, seed=seed)
    assignments = [
        _assignment_row(
            row,
            split=group_to_split[_required_text(row.get(group_key))],
            dataset_name=dataset_name,
            subset=subset,
        )
        for row in rows
    ]

    image_validation = _overlap_validation(assignments, key="image_id")
    sample_validation = _overlap_validation(assignments, key="sample_id")
    _raise_for_overlap("image_id", image_validation)
    _raise_for_overlap("sample_id", sample_validation)

    return {
        "seed": int(seed),
        "group_key": group_key,
        "split_names": list(SPLIT_NAMES),
        "ratios": list(ratio_values),
        "dataset_name": dataset_name,
        "subset": subset,
        "input_records": str(input_path),
        "counts_per_split": _counts_per_split(assignments),
        "label_counts_per_split": _field_counts_per_split(assignments, "label"),
        "object_counts_per_split": _field_counts_per_split(assignments, "object_name"),
        "image_id_overlap_validation": image_validation,
        "sample_id_overlap_validation": sample_validation,
        "assignments": assignments,
    }


def write_split_manifest(manifest: Mapping[str, object], output: Path | str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float, float]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"expected {len(SPLIT_NAMES)} ratios, got {len(ratios)}")
    values = tuple(float(ratio) for ratio in ratios)
    if any(not math.isfinite(ratio) or ratio < 0 for ratio in values):
        raise ValueError("ratios must be finite non-negative values")
    total = sum(values)
    if total <= 0:
        raise ValueError("ratios must sum to a positive value")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1.0, got {total:.6f}")
    return (values[0], values[1], values[2], values[3])


def _validate_single_source(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_name: str,
    subset: str,
) -> None:
    for index, row in enumerate(rows, start=1):
        sample_id = _optional_text(row.get("sample_id")) or f"row {index}"
        row_dataset = _first_text(row, ("dataset_name", "source_dataset", "dataset"))
        if row_dataset is not None and row_dataset != dataset_name:
            raise ValueError(
                f"mismatched dataset_name for {sample_id}: expected {dataset_name}, got {row_dataset}"
            )
        row_subset = _first_text(row, ("subset", "source_subset"))
        if row_subset is not None and row_subset != subset:
            raise ValueError(
                f"mismatched subset for {sample_id}: expected {subset}, got {row_subset}"
            )


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        group_value = _optional_text(row.get(group_key))
        if group_value is None:
            sample_id = _optional_text(row.get("sample_id")) or f"row {index}"
            raise ValueError(f"missing or blank group key '{group_key}' for {sample_id}")
        grouped[group_value].append(row)
    return dict(grouped)


def _assign_groups(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    ratios: Sequence[float],
    seed: int,
) -> dict[str, str]:
    group_keys = sorted(grouped)
    random.Random(seed).shuffle(group_keys)
    split_group_counts = _largest_remainder_counts(len(group_keys), ratios)
    group_to_split: dict[str, str] = {}
    offset = 0
    for split_name, count in zip(SPLIT_NAMES, split_group_counts, strict=True):
        for group_key in group_keys[offset : offset + count]:
            group_to_split[group_key] = split_name
        offset += count
    return group_to_split


def _largest_remainder_counts(total: int, ratios: Sequence[float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _assignment_row(
    row: Mapping[str, object],
    *,
    split: str,
    dataset_name: str,
    subset: str,
) -> dict[str, object]:
    return {
        "split": split,
        "dataset_name": dataset_name,
        "subset": subset,
        "sample_id": row.get("sample_id", ""),
        "image_id": row.get("image_id", ""),
        "image_path": row.get("image_path", ""),
        "object_name": row.get("object_name", ""),
        "label": row.get("label", ""),
        "question": row.get("question", ""),
    }


def _counts_per_split(assignments: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["split"]) for row in assignments)
    return {split_name: counts[split_name] for split_name in SPLIT_NAMES}


def _field_counts_per_split(
    assignments: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in SPLIT_NAMES}
    for row in assignments:
        split = str(row["split"])
        counts[split][_required_text(row.get(field))] += 1
    return {
        split_name: dict(sorted(counts[split_name].items()))
        for split_name in SPLIT_NAMES
    }


def _overlap_validation(
    assignments: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> dict[str, object]:
    split_by_value: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        value = _optional_text(row.get(key))
        if value is None:
            raise ValueError(f"missing or blank {key} in split assignment")
        split_by_value[value].add(str(row["split"]))
    overlaps = {
        value: sorted(splits)
        for value, splits in sorted(split_by_value.items())
        if len(splits) > 1
    }
    return {"valid": not overlaps, "overlaps": overlaps}


def _raise_for_overlap(key: str, validation: Mapping[str, object]) -> None:
    if validation.get("valid") is True:
        return
    overlaps = validation.get("overlaps", {})
    raise ValueError(f"{key} overlap detected between splits: {overlaps}")


def _first_text(row: Mapping[str, object], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = _optional_text(row.get(key))
        if value is not None:
            return value
    return None


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: object | None) -> str:
    text = _optional_text(value)
    if text is None:
        return ""
    return text
