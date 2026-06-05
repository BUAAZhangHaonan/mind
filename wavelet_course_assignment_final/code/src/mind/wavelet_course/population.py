"""Population labels, grouped splits, and audits for wavelet-course runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import random
from typing import Any

from .utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_SEED,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SUBSETS,
    SPLIT_NAMES,
    largest_remainder_counts,
    optional_text,
    parse_yes_no_label,
    read_json_object,
    require_binary_label,
    require_equal,
    require_field,
    require_non_empty,
    require_text,
    validate_ratios,
)


class PopulationClass(str, Enum):
    CORRECT = "correct"
    HARD_HALLUCINATION = "hard_hallucination"
    FALSE_NEGATIVE = "false_negative"
    PARSED_NONE = "parsed_none"
    INVALID_LABEL = "invalid_label"


@dataclass(frozen=True)
class WaveletPopulation:
    primary_entries: list[dict[str, Any]]
    labels: list[int]
    assignments: dict[str, str]
    audit_rows: list[dict[str, Any]]
    split_source: str


def classify_entry(entry: Mapping[str, Any]) -> PopulationClass:
    label = parse_yes_no_label(entry.get("label"))
    parsed_answer = parse_yes_no_label(entry.get("parsed_answer"))
    if label is None:
        return PopulationClass.INVALID_LABEL
    if parsed_answer is None:
        return PopulationClass.PARSED_NONE
    if parsed_answer == label:
        return PopulationClass.CORRECT
    if label == 0 and parsed_answer == 1:
        return PopulationClass.HARD_HALLUCINATION
    return PopulationClass.FALSE_NEGATIVE


def primary_label(entry: Mapping[str, Any]) -> int | None:
    entry_class = classify_entry(entry)
    if entry_class == PopulationClass.CORRECT:
        return 0
    if entry_class == PopulationClass.HARD_HALLUCINATION:
        return 1
    return None


def population_key(entry: Mapping[str, Any]) -> str:
    """Stable key for rows whose sample_id can repeat across subsets."""

    context = "population key"
    values = [
        require_text(entry.get("model_name"), field="model_name", context=context),
        require_text(entry.get("dataset_name"), field="dataset_name", context=context),
        require_text(entry.get("subset"), field="subset", context=context),
        require_text(entry.get("sample_id"), field="sample_id", context=context),
    ]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def build_wavelet_population(
    entries: Iterable[Mapping[str, Any]],
    *,
    manifest_dir: Path | str = Path("outputs/stage0/manifests"),
    dataset_name: str = DEFAULT_DATASET_NAME,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> WaveletPopulation:
    rows = [dict(entry) for entry in entries]
    require_non_empty(rows, name="entries")
    _validate_population_scope(rows, dataset_name=dataset_name, subsets=subsets)
    assignments, split_source = load_or_build_split_assignments(
        rows,
        manifest_dir=manifest_dir,
        subsets=subsets,
        seed=seed,
        ratios=ratios,
    )
    primary_entries: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in rows:
        label = primary_label(row)
        if label is None:
            continue
        key = population_key(row)
        enriched = dict(row)
        enriched["wavelet_population_key"] = key
        enriched["wavelet_split"] = assignments[key]
        enriched["wavelet_label"] = label
        primary_entries.append(enriched)
        labels.append(label)
    audit_rows = build_population_audit_rows(rows, assignments=assignments, subsets=subsets)
    return WaveletPopulation(
        primary_entries=primary_entries,
        labels=labels,
        assignments=assignments,
        audit_rows=audit_rows,
        split_source=split_source,
    )


def load_or_build_split_assignments(
    entries: Sequence[Mapping[str, Any]],
    *,
    manifest_dir: Path | str,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[dict[str, str], str]:
    existing, reason = _load_existing_split_assignments(
        entries,
        manifest_dir=manifest_dir,
        subsets=subsets,
    )
    if existing is not None:
        _validate_grouped_split(entries, existing)
        return existing, "stage0_split_manifest"
    built = build_grouped_split_assignments(entries, seed=seed, ratios=ratios)
    _validate_grouped_split(entries, built)
    return built, f"constructed:{reason}"


def build_grouped_split_assignments(
    entries: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, str]:
    ratio_values = validate_ratios(ratios, split_names=SPLIT_NAMES)
    grouped: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(entries, start=1):
        context = f"split row {index}"
        image_id = require_text(row.get("image_id"), field="image_id", context=context)
        grouped[image_id].append(population_key(row))
    group_keys = sorted(grouped)
    random.Random(int(seed)).shuffle(group_keys)
    counts = largest_remainder_counts(len(group_keys), ratio_values)
    group_to_split: dict[str, str] = {}
    offset = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for image_id in group_keys[offset : offset + count]:
            group_to_split[image_id] = split_name
        offset += count
    return {
        key: group_to_split[image_id]
        for image_id, keys in grouped.items()
        for key in keys
    }


def build_population_audit_rows(
    entries: Sequence[Mapping[str, Any]],
    *,
    assignments: Mapping[str, str],
    subsets: Sequence[str] = DEFAULT_SUBSETS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        subset_rows = [row for row in entries if row.get("subset") == subset]
        rows.append(_audit_row(subset, subset_rows, assignments=assignments))
    return rows


def _audit_row(
    subset: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    class_counts = Counter(classify_entry(row).value for row in rows)
    gt_counts = Counter(_label_name(parse_yes_no_label(row.get("label"))) for row in rows)
    parsed_counts = Counter(_label_name(parse_yes_no_label(row.get("parsed_answer"))) for row in rows)
    split_label_counts = _split_primary_counts(rows, assignments=assignments)
    audit: dict[str, Any] = {
        "subset": subset,
        "total": len(rows),
        "gt_yes": gt_counts["yes"],
        "gt_no": gt_counts["no"],
        "parsed_yes": parsed_counts["yes"],
        "parsed_no": parsed_counts["no"],
        "correct": class_counts[PopulationClass.CORRECT.value],
        "hard_hallucination": class_counts[PopulationClass.HARD_HALLUCINATION.value],
        "false_negative": class_counts[PopulationClass.FALSE_NEGATIVE.value],
        "parsed_none": class_counts[PopulationClass.PARSED_NONE.value],
        "invalid_label": class_counts[PopulationClass.INVALID_LABEL.value],
        "primary_pos": class_counts[PopulationClass.HARD_HALLUCINATION.value],
        "primary_neg": class_counts[PopulationClass.CORRECT.value],
    }
    for split_name in SPLIT_NAMES:
        audit[f"{split_name}_pos"] = split_label_counts[split_name][1]
        audit[f"{split_name}_neg"] = split_label_counts[split_name][0]
    return audit


def _load_existing_split_assignments(
    entries: Sequence[Mapping[str, Any]],
    *,
    manifest_dir: Path | str,
    subsets: Sequence[str],
) -> tuple[dict[str, str] | None, str]:
    required_keys = {population_key(row) for row in entries}
    keys_by_subset_sample: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in entries:
        subset = require_text(row.get("subset"), field="subset", context="split input")
        sample_id = require_text(row.get("sample_id"), field="sample_id", context="split input")
        keys_by_subset_sample[(subset, sample_id)].add(population_key(row))
    assignments: dict[str, str] = {}
    seen_split_names: set[str] = set()
    missing_files: list[str] = []
    for subset in subsets:
        path = Path(manifest_dir) / f"split_manifest_repope_{subset}.json"
        if not path.exists():
            missing_files.append(str(path))
            continue
        payload = read_json_object(path)
        raw_assignments = payload.get("assignments")
        if not isinstance(raw_assignments, list):
            raise ValueError(f"{path}: field 'assignments' must be a list")
        for index, raw_row in enumerate(raw_assignments, start=1):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"{path}: assignment {index} must be a mapping")
            sample_id = optional_text(raw_row.get("sample_id"))
            split = optional_text(raw_row.get("split"))
            if sample_id is None or split is None:
                raise ValueError(f"{path}: assignment {index} missing sample_id or split")
            keys = keys_by_subset_sample.get((subset, sample_id), set())
            if not keys:
                continue
            seen_split_names.add(split)
            for key in keys:
                assignments[key] = split
    if missing_files:
        return None, "missing_split_manifest"
    if set(SPLIT_NAMES) - seen_split_names:
        return None, "split_manifest_not_train_validation_test"
    if set(assignments) != required_keys:
        return None, "split_manifest_incomplete_sample_ids"
    invalid = sorted(set(assignments.values()) - set(SPLIT_NAMES))
    if invalid:
        return None, f"split_manifest_has_invalid_split_names:{invalid}"
    return assignments, "complete"


def _validate_population_scope(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    subsets: Sequence[str],
) -> None:
    subset_set = set(subsets)
    for index, row in enumerate(rows, start=1):
        context = f"population row {index}"
        require_field(row, "dataset_name", context=context)
        require_field(row, "subset", context=context)
        require_equal(row["dataset_name"], dataset_name, field="dataset_name", context=context)
        if row["subset"] not in subset_set:
            raise ValueError(f"{context}: unexpected subset {row['subset']!r}")
        require_binary_label(row.get("label"), field="label", context=context)


def _validate_grouped_split(
    entries: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> None:
    split_by_image: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(entries, start=1):
        context = f"split validation row {index}"
        image_id = require_text(row.get("image_id"), field="image_id", context=context)
        key = population_key(row)
        split = assignments.get(key)
        if split not in SPLIT_NAMES:
            raise ValueError(f"{context}: missing or invalid split for population_key={key!r}")
        split_by_image[image_id].add(split)
    overlaps = {
        image_id: sorted(splits)
        for image_id, splits in split_by_image.items()
        if len(splits) > 1
    }
    if overlaps:
        raise ValueError(f"image_id split overlap detected: {overlaps}")


def _split_primary_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    assignments: Mapping[str, str],
) -> dict[str, Counter[int]]:
    counts: dict[str, Counter[int]] = {split_name: Counter() for split_name in SPLIT_NAMES}
    for row in rows:
        label = primary_label(row)
        if label is None:
            continue
        split = assignments[population_key(row)]
        counts[split][label] += 1
    return counts


def _label_name(label: int | None) -> str:
    if label == 1:
        return "yes"
    if label == 0:
        return "no"
    return "none"
