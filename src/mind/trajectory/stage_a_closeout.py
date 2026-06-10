"""Stage A closeout helpers over the unified full-cache panel."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable, Iterator, Mapping, Sequence

import torch

from .splits import DEFAULT_RATIOS, DEFAULT_SEED, SPLIT_NAMES
from .stage_a_population import PopulationClass, classify_entry, summarize_population


EXPECTED_PANEL_SIZE = 16
CLOSEOUT_VARIANTS = (
    "Raw-Static",
    "Sphere-Static",
    "Raw-Traj-MeanPool",
    "Sphere-Traj-MeanPool",
    "Raw-Traj-LSTM",
    "Sphere-Traj-LSTM",
)
CLOSEOUT_READOUTS = ("Diag-Classifier", "Diag-KNN")
FAMILY_SUBSETS = {
    "pope": ("popular", "random", "adversarial"),
    "repope": ("popular", "random", "adversarial"),
    "dash-b": ("all",),
}


@dataclass(frozen=True)
class CloseoutPanelManifest:
    """Validated view of the Experiment 2 unified full-cache manifest."""

    path: Path
    models: list[dict[str, object]]
    payload: dict[str, object]


def load_closeout_panel_manifest(
    full_cache_root: Path | str,
    *,
    expected_panel_size: int = EXPECTED_PANEL_SIZE,
) -> CloseoutPanelManifest:
    """Load the unified full-cache manifest used as Stage A closeout input."""

    root = Path(full_cache_root)
    manifest_path = root / "manifests" / "unified_full_cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unified manifest must be a JSON object: {manifest_path}")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("unified manifest must contain a models list")
    if len(models) != int(expected_panel_size):
        raise ValueError(
            f"Stage A closeout requires {expected_panel_size} panel models; found {len(models)}"
        )
    aliases: list[str] = []
    rows: list[dict[str, object]] = []
    for row in models:
        if not isinstance(row, Mapping):
            raise ValueError("each unified manifest model row must be an object")
        model_row = dict(row)
        alias = str(model_row.get("model_alias", "")).strip()
        if not alias:
            raise ValueError("each unified manifest model row must have model_alias")
        aliases.append(alias)
        status = str(model_row.get("status", ""))
        validation_status = str(model_row.get("validation_status", ""))
        if validation_status and validation_status != "passed":
            raise ValueError(f"{alias} validation_status is not passed: {validation_status}")
        if status in {"failed_extraction", "failed_validation", "blocked"}:
            raise ValueError(f"{alias} is not available for closeout: {status}")
        resolve_model_cache_root(model_row, root)
        rows.append(model_row)
    duplicates = sorted(alias for alias, count in Counter(aliases).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate models in unified manifest: " + ", ".join(duplicates))
    return CloseoutPanelManifest(path=manifest_path, models=rows, payload=payload)


def resolve_model_cache_root(model_row: Mapping[str, object], full_cache_root: Path | str) -> Path:
    """Resolve the physical cache root from a unified manifest model row."""

    for field in ("cache_root", "source_cache_root"):
        value = model_row.get(field)
        if value not in (None, ""):
            return _resolve_path(Path(full_cache_root), Path(str(value)))
    raise ValueError(f"missing cache root for {model_row.get('model_alias', '<unknown>')}")


def iter_full_cache_shards(
    cache_root: Path | str,
    *,
    dataset_name: str,
    subsets: Sequence[str],
) -> Iterator[Path]:
    """Yield full-cache shard paths for one model/dataset family."""

    root = Path(cache_root)
    for subset in subsets:
        subset_dir = root / dataset_name / subset
        for shard_path in sorted(subset_dir.glob("*.pt")):
            yield shard_path


def stream_full_cache_entries(
    model_row: Mapping[str, object],
    full_cache_root: Path | str,
    *,
    dataset_family: str,
    include_tensors: bool = True,
) -> Iterator[dict[str, object]]:
    """Stream full-cache entries for one model and dataset family."""

    family = _normalize_family(dataset_family)
    cache_root = resolve_model_cache_root(model_row, full_cache_root)
    alias = str(model_row["model_alias"])
    for shard_path in iter_full_cache_shards(
        cache_root,
        dataset_name=family,
        subsets=FAMILY_SUBSETS[family],
    ):
        payload = torch.load(shard_path, weights_only=False, map_location="cpu")
        for entry in _iter_payload_entries(payload):
            row = dict(entry)
            row.setdefault("model_alias", alias)
            row.setdefault("model_name", alias)
            row.setdefault("dataset_name", family)
            row.setdefault("source_dataset", family)
            row.setdefault("subset", shard_path.parent.name)
            if not include_tensors:
                row.pop("layer_vectors", None)
                row.pop("first_token_logits", None)
            yield row


def build_closeout_family_split(
    entries: Iterable[Mapping[str, object]],
    *,
    family: str,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    group_key: str = "image_id",
) -> dict[str, object]:
    """Build grouped family-level Stage A closeout splits."""

    normalized_family = _normalize_family(family)
    rows = [dict(entry) for entry in entries]
    _validate_family_scope(rows, normalized_family)
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
        "stage": "stage_a_closeout",
        "split_scope": f"{normalized_family}_family" if normalized_family != "dash-b" else "dash-b",
        "dataset_family": normalized_family,
        "seed": int(seed),
        "group_key": group_key,
        "split_names": list(SPLIT_NAMES),
        "ratios": list(ratio_values),
        "allowed_subsets": list(FAMILY_SUBSETS[normalized_family]),
        "num_entries": len(assignments),
        "num_image_ids": len(grouped),
        "counts_per_split": _counts_per_split(assignments),
        "counts_per_model": _field_counts(assignments, "model_name"),
        "counts_per_dataset": _field_counts(assignments, "dataset_name"),
        "counts_per_subset": _field_counts(assignments, "subset"),
        "primary_population_counts_per_split": _primary_counts_per_split(assignments),
        "hard_hallucination_counts_per_split": _hard_hallucination_counts_per_split(assignments),
        "stage0_split_conflict_report": _stage0_split_conflict_report(rows, group_key=group_key),
        "image_id_overlap_validation": _overlap_validation(assignments, key="image_id"),
        "sample_id_overlap_validation": _overlap_validation(assignments, key="sample_id"),
        "assignments": assignments,
    }


def write_split_manifest(manifest: Mapping[str, object], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_csv_rows(path: Path | str, rows: Sequence[Mapping[str, object]]) -> None:
    """Write dictionaries as CSV using the union of row keys."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def render_closeout_summary_markdown(summary: Mapping[str, object]) -> str:
    """Render the tracked Stage A closeout summary without scientific claims."""

    verdict = summary.get("sphere_closeout_verdict", {})
    if not isinstance(verdict, Mapping):
        verdict = {}
    lines = [
        "# Stage A Closeout Summary",
        "",
        "This report closes the representation pretest phase. It does not validate the final MIND detector.",
        "",
        "## Status",
        "",
        f"- stage_a_closed: {str(summary.get('stage_a_closed', False)).lower()}",
        f"- stage_b_started: {str(summary.get('stage_b_started', False)).lower()}",
        f"- panel_models: {len(summary.get('panel_models', []))}",
        f"- sphere_verdict: {verdict.get('verdict', 'unknown')}",
        "",
        "## Canonical Tables",
        "",
        f"- metrics_long: {summary.get('metrics_long_path', '')}",
        f"- repope_classifier: {summary.get('repope_classifier_table_path', '')}",
        f"- repope_knn: {summary.get('repope_knn_table_path', '')}",
        f"- pope_secondary: {summary.get('pope_secondary_table_path', '')}",
        f"- dash_b_secondary: {summary.get('dash_b_secondary_table_path', '')}",
        f"- per_model_summary: {summary.get('per_model_summary_path', '')}",
        "",
        "## Model Status",
        "",
        "| model | status | reason |",
        "| --- | --- | --- |",
    ]
    failed = summary.get("failed_models", {})
    failed = failed if isinstance(failed, Mapping) else {}
    evaluated = set(str(model) for model in summary.get("evaluated_models", []))
    for model in summary.get("panel_models", []):
        model_name = str(model)
        if model_name in failed:
            lines.append(f"| {model_name} | failed | {failed[model_name]} |")
        elif model_name in evaluated:
            lines.append(f"| {model_name} | evaluated |  |")
        else:
            lines.append(f"| {model_name} | missing | no metric row or failure row |")
    lines.extend(
        [
            "",
            "## Verdict Inputs",
            "",
            f"- supporting_models: {', '.join(map(str, verdict.get('supporting_models', [])))}",
            f"- contradicting_models: {', '.join(map(str, verdict.get('contradicting_models', [])))}",
            "",
            "Stage A is closed after Raw-Traj-LSTM is added and the closeout summary is written.",
            "Later stages must not reopen Stage A except if the frozen theory note is explicitly revised.",
        ]
    )
    return "\n".join(lines) + "\n"


def decide_sphere_closeout_verdict(
    metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Classify Sphere-Traj-LSTM relative to Raw-Traj-LSTM on RePOPE test classifier."""

    by_model: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in metric_rows:
        if (
            row.get("dataset_family") == "repope"
            and row.get("readout") == "Diag-Classifier"
            and row.get("eval_split") == "test"
            and row.get("eval_scope") == "pooled"
            and row.get("metric_status", "passed") == "passed"
            and row.get("variant") in {"Raw-Traj-LSTM", "Sphere-Traj-LSTM"}
        ):
            by_model[str(row["model_name"])][str(row["variant"])] = row

    comparisons: list[dict[str, object]] = []
    for model_name, rows in sorted(by_model.items()):
        raw = rows.get("Raw-Traj-LSTM")
        sphere = rows.get("Sphere-Traj-LSTM")
        if raw is None or sphere is None:
            continue
        raw_pr = float(raw["pr_auc"])
        sphere_pr = float(sphere["pr_auc"])
        comparisons.append(
            {
                "model_name": model_name,
                "raw_pr_auc": raw_pr,
                "sphere_pr_auc": sphere_pr,
                "delta_pr_auc": sphere_pr - raw_pr,
                "winner": "sphere" if sphere_pr > raw_pr else "raw" if raw_pr > sphere_pr else "tie",
            }
        )
    if not comparisons:
        return {
            "verdict": "neutral",
            "reason": "no complete RePOPE Diag-Classifier comparison rows were available",
            "supporting_models": [],
            "contradicting_models": [],
            "comparisons": [],
        }

    deltas = [float(row["delta_pr_auc"]) for row in comparisons]
    mean_delta = sum(deltas) / len(deltas)
    sphere_wins = sum(1 for row in comparisons if row["winner"] == "sphere")
    raw_wins = sum(1 for row in comparisons if row["winner"] == "raw")
    if mean_delta >= 0.0 and sphere_wins >= raw_wins:
        verdict = "beneficial"
    elif mean_delta < -0.02 and raw_wins > sphere_wins:
        verdict = "harmful"
    else:
        verdict = "neutral"
    return {
        "verdict": verdict,
        "mean_delta_pr_auc": mean_delta,
        "sphere_wins": sphere_wins,
        "raw_wins": raw_wins,
        "ties": len(comparisons) - sphere_wins - raw_wins,
        "supporting_models": [
            str(row["model_name"])
            for row in comparisons
            if row["winner"] in {"sphere", "tie"}
        ],
        "contradicting_models": [
            str(row["model_name"]) for row in comparisons if row["winner"] == "raw"
        ],
        "comparisons": comparisons,
    }


def summarize_closeout_status(
    *,
    panel_models: Sequence[str],
    metric_rows: Sequence[Mapping[str, object]],
    failures: Mapping[str, str],
) -> dict[str, object]:
    """Summarize closeout status without hiding failed panel models."""

    evaluated = sorted({str(row.get("model_name", "")) for row in metric_rows if row.get("model_name")})
    failed = dict(sorted((str(model), str(reason)) for model, reason in failures.items()))
    missing = sorted(set(panel_models) - set(evaluated) - set(failed))
    return {
        "panel_models": list(panel_models),
        "evaluated_models": evaluated,
        "failed_models": failed,
        "missing_models": missing,
        "all_panel_models_present": not missing,
        "stage_b_started": False,
    }


def population_audit_row(
    entries: Sequence[Mapping[str, object]],
    *,
    model_name: str,
    dataset_family: str,
    subset: str = "pooled",
    split: str | None = None,
) -> dict[str, object]:
    summary = summarize_population(entries)
    row = {
        "model_name": model_name,
        "dataset_family": dataset_family,
        "subset": subset,
        "num_entries": summary["num_entries"],
        "num_correct": summary["num_correct"],
        "num_hard_hallucination": summary["num_hard_hallucination"],
        "num_false_negative_error": summary["num_false_negative_error"],
        "num_parsed_none": summary["num_parsed_none"],
        "num_primary_population": summary["num_primary_population"],
        "hallucination_rate_in_primary_population": summary[
            "hallucination_rate_in_primary_population"
        ],
    }
    if split is not None:
        row["split"] = split
    return row


def _resolve_path(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base.parent.parent / path).resolve() if str(path).startswith("outputs/") else (base / path)


def _iter_payload_entries(payload: object) -> Iterator[Mapping[str, object]]:
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


def _normalize_family(family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized not in FAMILY_SUBSETS:
        raise ValueError("unsupported Stage A closeout dataset family: " + str(family))
    return normalized


def _validate_family_scope(rows: Sequence[Mapping[str, object]], family: str) -> None:
    allowed_subsets = set(FAMILY_SUBSETS[family])
    invalid = []
    for row in rows:
        dataset = str(row.get("dataset_name", row.get("source_dataset", "")))
        subset = str(row.get("subset", ""))
        if dataset != family or subset not in allowed_subsets:
            invalid.append(f"{dataset}/{subset}")
    if invalid:
        raise ValueError("entries outside closeout family scope: " + ", ".join(sorted(set(invalid))))


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float, float]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"expected {len(SPLIT_NAMES)} ratios")
    values = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("ratios must be finite non-negative values")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("ratios must sum to 1.0")
    return values  # type: ignore[return-value]


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_required_text(row.get(group_key))].append(row)
    return dict(grouped)


def _assign_groups(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    ratios: Sequence[float],
    seed: int,
) -> dict[str, str]:
    keys = sorted(grouped)
    random.Random(int(seed)).shuffle(keys)
    counts = _largest_remainder_counts(len(keys), ratios)
    result: dict[str, str] = {}
    offset = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for key in keys[offset : offset + count]:
            result[key] = split_name
        offset += count
    return result


def _largest_remainder_counts(total: int, ratios: Sequence[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _assignment_row(
    row: Mapping[str, object],
    *,
    split: str,
    group_key: str,
) -> dict[str, object]:
    return {
        "split": split,
        "model_name": row.get("model_name", row.get("model_alias", "")),
        "model_alias": row.get("model_alias", row.get("model_name", "")),
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


def _counts_per_split(assignments: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["split"]) for row in assignments)
    return {split_name: counts[split_name] for split_name in SPLIT_NAMES}


def _field_counts(assignments: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts = Counter(_required_text(row.get(field)) for row in assignments)
    return dict(sorted(counts.items()))


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


def _stage0_split_conflict_report(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
) -> dict[str, object]:
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = str(row.get(group_key, ""))
        split = str(row.get("split", ""))
        if group and split in SPLIT_NAMES:
            split_by_group[group].add(split)
    conflicts = {
        group: sorted(values)
        for group, values in split_by_group.items()
        if len(values) > 1
    }
    return {
        "num_conflicting_image_ids": len(conflicts),
        "conflicts": dict(sorted(conflicts.items())),
        "stage_a_closeout_action": "built a new family-level split",
    }


def _overlap_validation(assignments: Sequence[Mapping[str, object]], *, key: str) -> dict[str, object]:
    splits_by_key: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        splits_by_key[_required_text(row.get(key))].add(str(row["split"]))
    conflicts = {
        item: sorted(splits)
        for item, splits in splits_by_key.items()
        if len(splits) > 1
    }
    return {
        "status": "passed" if not conflicts else "failed",
        "num_conflicts": len(conflicts),
        "conflicts": dict(sorted(conflicts.items())),
    }


def _required_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("required text field is missing or blank")
    return text
