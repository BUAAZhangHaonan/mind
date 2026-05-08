"""Stage A POPE-family split construction."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

from .splits import DEFAULT_RATIOS, DEFAULT_SEED, SPLIT_NAMES
from .stage_a_population import PopulationClass, classify_entry


STAGE_A_PRIMARY_DATASET = "pope"
STAGE_A_PRIMARY_SUBSETS = ("popular", "random", "adversarial")


def build_pope_family_split(
    entries: Iterable[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    group_key: str = "image_id",
) -> dict[str, object]:
    """Build one Stage A split across POPE-family subsets by image id."""

    rows = [dict(entry) for entry in entries]
    _validate_stage_a_primary_scope(rows)
    ratio_values = _validate_ratios(ratios)
    grouped = _group_rows(rows, group_key=group_key)
    group_to_split = _assign_groups(grouped, ratios=ratio_values, seed=seed)
    assignments = [
        _assignment_row(
            row,
            split=group_to_split[_required_text(row.get(group_key))],
            group_key=group_key,
        )
        for row in rows
    ]

    return {
        "stage": "stage_a",
        "split_scope": "pope_family",
        "seed": int(seed),
        "group_key": group_key,
        "split_names": list(SPLIT_NAMES),
        "ratios": list(ratio_values),
        "num_entries": len(assignments),
        "num_image_ids": len(grouped),
        "counts_per_split": _counts_per_split(assignments),
        "counts_per_model": _field_counts(assignments, "model_name"),
        "counts_per_dataset": _field_counts(assignments, "dataset_name"),
        "counts_per_subset": _field_counts(assignments, "subset"),
        "label_counts_per_split": _field_counts_per_split(assignments, "label"),
        "primary_population_counts_per_split": _primary_counts_per_split(assignments),
        "hard_hallucination_counts_per_split": _hard_hallucination_counts_per_split(assignments),
        "object_counts_per_split": _field_counts_per_split(assignments, "object_name"),
        "stage0_split_conflict_report": _stage0_split_conflict_report(rows, group_key=group_key),
        "image_id_overlap_validation": _overlap_validation(assignments, key="image_id"),
        "sample_id_overlap_validation": _overlap_validation(assignments, key="sample_id"),
        "assignments": assignments,
    }


def _validate_stage_a_primary_scope(rows: Sequence[Mapping[str, object]]) -> None:
    invalid_datasets = sorted(
        {
            _scope_value(row.get("dataset_name"))
            for row in rows
            if _scope_value(row.get("dataset_name")) != STAGE_A_PRIMARY_DATASET
        }
    )
    invalid_subsets = sorted(
        {
            _scope_value(row.get("subset"))
            for row in rows
            if _scope_value(row.get("subset")) not in STAGE_A_PRIMARY_SUBSETS
        }
    )
    messages = []
    if invalid_datasets:
        messages.append(
            "Stage A primary scope requires dataset_name='pope'; found: "
            + ", ".join(invalid_datasets)
        )
    if invalid_subsets:
        messages.append(
            "Stage A primary scope requires subset in "
            + ", ".join(STAGE_A_PRIMARY_SUBSETS)
            + "; found: "
            + ", ".join(invalid_subsets)
        )
    if messages:
        raise ValueError("; ".join(messages))


def write_family_split_manifest(manifest: Mapping[str, object], output: Path | str) -> None:
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


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        value = _optional_text(row.get(group_key))
        if value is None:
            sample_id = _optional_text(row.get("sample_id")) or f"row {index}"
            raise ValueError(f"missing or blank group key '{group_key}' for {sample_id}")
        grouped[value].append(row)
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
        for group in group_keys[offset : offset + count]:
            group_to_split[group] = split_name
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
    group_key: str,
) -> dict[str, object]:
    stage0_split = _stage0_split_value(row)
    assignment: dict[str, object] = {
        "split": split,
        "model_name": row.get("model_name", ""),
        "dataset_name": row.get("dataset_name", row.get("source_dataset", "")),
        "source_dataset": row.get("source_dataset", row.get("dataset_name", "")),
        "subset": row.get("subset", ""),
        "sample_id": row.get("sample_id", ""),
        "image_id": row.get(group_key, ""),
        "image_path": row.get("image_path", ""),
        "object_name": row.get("object_name", ""),
        "label": row.get("label", ""),
        "parsed_answer": row.get("parsed_answer", ""),
        "question": row.get("question", ""),
    }
    if stage0_split is not None:
        assignment["stage0_split"] = stage0_split
    return assignment


def _stage0_split_conflict_report(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
) -> dict[str, object]:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = _optional_text(row.get(group_key))
        split = _stage0_split_value(row)
        if group is None or split is None:
            continue
        splits_by_group[group].add(split)

    conflicts = {
        group: sorted(values)
        for group, values in sorted(splits_by_group.items())
        if len(values) > 1
    }
    if conflicts:
        action = "ignored Stage 0 per-subset assignments and built a new POPE-family split"
    else:
        action = "built a new POPE-family split"
    return {
        "num_conflicting_image_ids": len(conflicts),
        "conflicts": conflicts,
        "stage_a_action": action,
    }


def _stage0_split_value(row: Mapping[str, object]) -> str | None:
    explicit = _optional_text(row.get("stage0_split"))
    if explicit is not None:
        return explicit
    split = _optional_text(row.get("split"))
    if split in SPLIT_NAMES:
        return split
    return None


def _counts_per_split(assignments: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["split"]) for row in assignments)
    return {split_name: counts[split_name] for split_name in SPLIT_NAMES}


def _field_counts(
    assignments: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, int]:
    counts = Counter(_required_text(row.get(field)) for row in assignments)
    return dict(sorted(counts.items()))


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


def _primary_counts_per_split(assignments: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for row in assignments:
        if classify_entry(row) in {PopulationClass.CORRECT, PopulationClass.HARD_HALLUCINATION}:
            counts[str(row["split"])] += 1
    return counts


def _hard_hallucination_counts_per_split(assignments: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for row in assignments:
        if classify_entry(row) == PopulationClass.HARD_HALLUCINATION:
            counts[str(row["split"])] += 1
    return counts


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


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: object | None) -> str:
    return _optional_text(value) or ""


def _scope_value(value: object | None) -> str:
    return _optional_text(value) or "<blank>"
