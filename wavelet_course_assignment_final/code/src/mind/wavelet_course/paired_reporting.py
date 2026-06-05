"""Paired reporting helpers for wavelet-course v2 runs.

The functions in this module operate on ``list[dict]`` rows and intentionally
avoid pandas. A failed method run is still a completed reporting row: it must
remain in ``metrics_long.csv`` and in the paired wide view.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = ("Teacher", "Ours")
DEFAULT_PAIR_KEY_FIELDS = (
    "run_id",
    "model_name",
    "dataset_name",
    "subset_scope",
    "seed",
    "block",
    "pair_id",
    "classifier",
    "learning_rate",
)
PAIRED_CONFIG_COMPARE_FIELD_GROUPS = (
    ("transform", ("transform",)),
    ("wavelet", ("wavelet",)),
    ("level", ("level",)),
    ("scales", ("cwt_scales", "scales")),
    ("threshold", ("threshold",)),
    ("window_mode", ("window_mode",)),
    ("window_strategy", ("window_strategy",)),
    ("window_size", ("window_size",)),
    ("stride", ("stride",)),
    ("feature_protocol", ("feature_protocol",)),
    ("classifier", ("classifier",)),
    ("sequence_model", ("sequence_model",)),
    ("learning_rate", ("learning_rate",)),
    ("seed", ("seed",)),
    ("split", ("split", "wavelet_split", "split_name", "data_split")),
)
LOWER_IS_BETTER_METRICS = frozenset({"fpr_at_95pct_tpr"})
LONG_METRIC_ALIASES = {
    "pr_auc": "test_pr_auc",
    "average_precision": "test_average_precision",
    "roc_auc": "test_roc_auc",
}
DEFAULT_COMPARE_METRICS = (
    "pr_auc",
    "average_precision",
    "roc_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "fpr_at_95pct_tpr",
)
LONG_FIELD_ORDER = (
    "run_id",
    "model_name",
    "dataset_name",
    "subset_scope",
    "seed",
    "block",
    "pair_id",
    "row_id",
    "source",
    "signal_builder",
    "config_name",
    "method_family",
    "classifier",
    "sequence_model",
    "learning_rate",
    "max_epochs",
    "patience",
    "transform",
    "feature_protocol",
    "wavelet",
    "level",
    "window_strategy",
    "window_size",
    "stride",
    "mode",
    "cwt_scales",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_pos",
    "val_pos",
    "test_pos",
    "feature_shape",
    "feature_seconds",
    "train_eval_seconds",
    "total_seconds",
    "best_epoch",
    "best_validation_pr_auc",
    "epochs_ran",
    "early_stopped",
    "converged",
    "max_epoch_reached",
    "pr_auc",
    "test_pr_auc",
    "average_precision",
    "test_average_precision",
    "roc_auc",
    "test_roc_auc",
    "best_val_threshold",
    "test_f1",
    "test_precision",
    "test_recall",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "fpr_at_95pct_tpr",
    "status",
    "failure_reason",
)
SOURCE_WIDE_FIELDS = (
    "config_name",
    "row_id",
    "signal_builder",
    "status",
    "failure_reason",
    "feature_shape",
    "feature_seconds",
    "train_eval_seconds",
    "total_seconds",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_pos",
    "val_pos",
    "test_pos",
    "learning_rate",
    "max_epochs",
    "patience",
    "best_epoch",
    "best_validation_pr_auc",
    "epochs_ran",
    "early_stopped",
    "converged",
    "max_epoch_reached",
)
BEST_BY_BLOCK_FIELDS = (
    "block",
    "metric",
    "paired_rows",
    "comparable_pairs",
    "not_comparable_pairs",
    "selection_scope",
    "long_rows",
    "success_rows",
    "failure_rows",
    "best_source",
    "best_pair_id",
    "best_config_name",
    "best_classifier",
    "best_value",
    "teacher_best_pair_id",
    "teacher_best_config_name",
    "teacher_best_classifier",
    "teacher_best_value",
    "ours_best_pair_id",
    "ours_best_config_name",
    "ours_best_classifier",
    "ours_best_value",
)
PAIRWISE_WINRATE_FIELDS = (
    "block",
    "metric",
    "num_pairs",
    "num_both_success",
    "num_teacher_success_ours_fail",
    "num_ours_success_teacher_fail",
    "num_both_fail",
    "ours_wins_by_pr_auc",
    "teacher_wins_by_pr_auc",
    "mean_delta_pr_auc",
    "median_delta_pr_auc",
    "mean_delta_f1",
    "median_delta_f1",
    "total_pairs",
    "comparable_pairs",
    "ours_wins",
    "teacher_wins",
    "ties",
    "not_comparable_pairs",
    "ours_winrate",
    "teacher_winrate",
    "tie_rate",
)
FAILURE_FIELD_ORDER = (
    "run_id",
    "model_name",
    "dataset_name",
    "subset_scope",
    "seed",
    "block",
    "pair_id",
    "source",
    "config_name",
    "classifier",
    "status",
    "failure_reason",
)
WIDE_BASE_FIELDS = (
    "run_id",
    "model_name",
    "dataset_name",
    "subset_scope",
    "seed",
    "block",
    "pair_id",
    "classifier",
    "learning_rate",
    "shared_config_summary",
    "paired_status",
    "paired_failure_reason",
)
SOURCE_ALIASES = {
    "teacher": "Teacher",
    "teacher_bagua": "Teacher",
    "teacher-bagua": "Teacher",
    "teacher_bagua_wavelet": "Teacher",
    "ours": "Ours",
    "ours_wavelet": "Ours",
    "ours-wavelet": "Ours",
    "ours_wavelet_v2": "Ours",
}
BLOCK_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
SUMMARY_SPLITS = ("train", "validation", "test")


def assert_paired_completeness(
    rows: list[dict[str, Any]],
    *,
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    expected_blocks: Sequence[str] | None = None,
    expected_pair_ids: Sequence[str] | None = None,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
) -> list[dict[str, Any]]:
    """Return normalized row copies only if every paired key has exact sources.

    A failure status is valid and does not make a row incomplete. Missing rows,
    duplicate rows for the same source, unknown sources, or missing expected
    blocks/pairs raise ``ValueError`` before any derived report is built.
    """

    normalized = _normalized_rows(rows)
    if not normalized:
        raise ValueError("paired metrics rows must not be empty")

    expected_source_tuple = tuple(_canonical_source(source) for source in expected_sources)
    expected_source_set = set(expected_source_tuple)
    if not expected_source_set:
        raise ValueError("expected_sources must not be empty")
    if len(expected_source_set) != len(expected_source_tuple):
        raise ValueError(f"expected_sources contains duplicates: {list(expected_sources)}")

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in normalized:
        source = _canonical_source(row.get("source"))
        if source not in expected_source_set:
            raise ValueError(f"unknown paired source {row.get('source')!r}; expected {sorted(expected_source_set)}")
        key = _pair_key(row, pair_key_fields)
        source_rows = grouped.setdefault(key, {})
        if source in source_rows:
            raise ValueError(f"duplicate paired metrics row for key={key!r}, source={source!r}")
        source_rows[source] = row

    for key, source_rows in grouped.items():
        present = set(source_rows)
        if present != expected_source_set:
            missing = sorted(expected_source_set - present)
            extra = sorted(present - expected_source_set)
            raise ValueError(
                f"paired metrics key={key!r} must have exact sources {sorted(expected_source_set)}; "
                f"missing={missing}, extra={extra}"
            )
        _assert_paired_config_consistency(key, source_rows, expected_source_tuple)

    if expected_blocks is not None:
        present_blocks = {str(row.get("block", "")) for row in normalized if str(row.get("block", ""))}
        missing_blocks = sorted(set(str(block) for block in expected_blocks) - present_blocks, key=_block_sort_key)
        if missing_blocks:
            raise ValueError(f"paired metrics missing blocks: {missing_blocks}")

    if expected_pair_ids is not None:
        present_pairs = {str(_pair_id(row)) for row in normalized if str(_pair_id(row))}
        expected_pairs = set(str(pair_id) for pair_id in expected_pair_ids)
        missing_pairs = sorted(expected_pairs - present_pairs)
        extra_pairs = sorted(present_pairs - expected_pairs)
        if missing_pairs:
            raise ValueError(f"paired metrics missing pair_ids: {missing_pairs}")
        if extra_pairs:
            raise ValueError(f"paired metrics contains unknown pair_ids: {extra_pairs}")

    return normalized


def build_metrics_wide_paired_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: Sequence[str] = DEFAULT_COMPARE_METRICS,
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
    primary_metric: str = "pr_auc",
) -> list[dict[str, Any]]:
    """Build one wide Teacher/Ours row per paired key without dropping failures."""

    metric_names = _metric_names(metrics, primary_metric)
    normalized = assert_paired_completeness(
        rows,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    source_order = tuple(_canonical_source(source) for source in expected_sources)
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in normalized:
        grouped[_pair_key(row, pair_key_fields)][_canonical_source(row.get("source"))] = row

    wide_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=_paired_key_sort_key):
        source_rows = grouped[key]
        representative = _representative_row(source_rows, source_order)
        wide: dict[str, Any] = {
            field: _key_field_value(representative, field)
            for field in pair_key_fields
            if _include_key_field(normalized, field)
        }
        wide.setdefault("block", representative.get("block", ""))
        wide.setdefault("pair_id", _pair_id(representative))
        wide["shared_config_summary"] = _shared_config_summary(representative)
        statuses = [_status(source_rows[source]) for source in source_order]
        wide["paired_status"] = _paired_status(statuses)
        wide["paired_failure_reason"] = _paired_failure_reason(source_rows, source_order)

        for source in source_order:
            prefix = _source_prefix(source)
            row = source_rows[source]
            for field in SOURCE_WIDE_FIELDS:
                wide[f"{prefix}_{field}"] = row.get(field, "")
            for metric in metric_names:
                wide[f"{prefix}_{metric}"] = row.get(metric, "")

        if {"Teacher", "Ours"} <= set(source_order):
            teacher = source_rows["Teacher"]
            ours = source_rows["Ours"]
            for metric in metric_names:
                teacher_value = _finite_float(teacher.get(metric))
                ours_value = _finite_float(ours.get(metric))
                wide[f"delta_{metric}"] = "" if teacher_value is None or ours_value is None else ours_value - teacher_value
                wide[f"winner_{metric}"] = _metric_winner(metric, teacher_value, ours_value)
            wide["teacher_f1"] = wide.get("teacher_test_f1", "")
            wide["ours_f1"] = wide.get("ours_test_f1", "")
            wide["delta_f1"] = wide.get("delta_test_f1", "")
            wide["speedup_or_slowdown"] = _runtime_ratio(
                teacher.get("total_seconds"),
                ours.get("total_seconds"),
            )
        wide_rows.append(wide)
    return wide_rows


def build_best_by_block_rows(
    rows: list[dict[str, Any]],
    *,
    primary_metric: str = "pr_auc",
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
) -> list[dict[str, Any]]:
    """Return best successful source rows per block for the primary metric."""

    normalized = assert_paired_completeness(
        rows,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    source_order = tuple(_canonical_source(source) for source in expected_sources)
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in normalized:
        grouped[_pair_key(row, pair_key_fields)][_canonical_source(row.get("source"))] = row
    blocks = sorted({str(row.get("block", "")) for row in normalized}, key=_block_sort_key)
    output: list[dict[str, Any]] = []
    for block in blocks:
        block_groups = [
            source_rows
            for key, source_rows in sorted(grouped.items(), key=lambda item: _paired_key_sort_key(item[0]))
            if str(_representative_row(source_rows, source_order).get("block", "")) == block
        ]
        block_rows = [row for source_rows in block_groups for row in source_rows.values()]
        comparable_groups = [
            source_rows
            for source_rows in block_groups
            if _is_comparable_source_group(source_rows, source_order, primary_metric)
        ]
        comparable_rows = [
            source_rows[source]
            for source_rows in comparable_groups
            for source in source_order
        ]
        source_bests: dict[str, dict[str, Any] | None] = {}
        for canonical in source_order:
            candidates = [
                source_rows[canonical]
                for source_rows in comparable_groups
            ]
            source_bests[canonical] = _best_row(candidates, primary_metric)
        all_best = _best_row(comparable_rows, primary_metric)
        result = {
            "block": block,
            "metric": primary_metric,
            "paired_rows": len(block_groups),
            "comparable_pairs": len(comparable_groups),
            "not_comparable_pairs": len(block_groups) - len(comparable_groups),
            "selection_scope": "comparable_both_success",
            "long_rows": len(block_rows),
            "success_rows": sum(1 for row in block_rows if _is_success(row)),
            "failure_rows": sum(1 for row in block_rows if _is_failure(row)),
            "best_source": _canonical_source(all_best.get("source")) if all_best else "",
            "best_pair_id": _pair_id(all_best) if all_best else "",
            "best_config_name": all_best.get("config_name", "") if all_best else "",
            "best_classifier": all_best.get("classifier", "") if all_best else "",
            "best_value": all_best.get(primary_metric, "") if all_best else "",
        }
        for canonical in source_order:
            prefix = _source_prefix(canonical)
            best = source_bests.get(canonical)
            result[f"{prefix}_best_pair_id"] = _pair_id(best) if best else ""
            result[f"{prefix}_best_config_name"] = best.get("config_name", "") if best else ""
            result[f"{prefix}_best_classifier"] = best.get("classifier", "") if best else ""
            result[f"{prefix}_best_value"] = best.get(primary_metric, "") if best else ""
        output.append(result)
    return output


def build_pairwise_winrate_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: Sequence[str] = ("pr_auc",),
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
) -> list[dict[str, Any]]:
    """Return per-block and overall Teacher/Ours pairwise win rates."""

    source_order = tuple(_canonical_source(source) for source in expected_sources)
    if {"Teacher", "Ours"} - set(source_order):
        raise ValueError("pairwise winrate requires Teacher and Ours sources")
    wide_rows = build_metrics_wide_paired_rows(
        rows,
        metrics=metrics,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    blocks = sorted({str(row.get("block", "")) for row in wide_rows}, key=_block_sort_key)
    output: list[dict[str, Any]] = []
    for block in [*blocks, "overall"]:
        block_rows = wide_rows if block == "overall" else [row for row in wide_rows if str(row.get("block", "")) == block]
        for metric in metrics:
            output.append(_winrate_row(block, str(metric), block_rows))
    return output


def build_failure_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-success rows with normalized source labels and all original fields."""

    failures: list[dict[str, Any]] = []
    for row in _normalized_rows(rows):
        if _is_failure(row):
            failures.append(dict(row))
    return failures


def write_metrics_long_csv(rows: list[dict[str, Any]], output: Path | str) -> Path:
    """Write long metrics rows exactly once, including failed rows."""

    aliased_rows = [_with_long_metric_aliases(row) for row in rows]
    return _write_csv(aliased_rows, output, preferred_fields=LONG_FIELD_ORDER)


def write_metrics_wide_paired_csv(
    rows: list[dict[str, Any]],
    output: Path | str,
    *,
    metrics: Sequence[str] = DEFAULT_COMPARE_METRICS,
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
    primary_metric: str = "pr_auc",
) -> Path:
    """Build and write ``metrics_wide_paired.csv`` from long rows."""

    wide_rows = build_metrics_wide_paired_rows(
        rows,
        metrics=metrics,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
        primary_metric=primary_metric,
    )
    return _write_csv(
        wide_rows,
        output,
        preferred_fields=_wide_field_order(metrics, pair_key_fields, primary_metric),
    )


def write_best_by_block_csv(
    rows: list[dict[str, Any]],
    output: Path | str,
    *,
    primary_metric: str = "pr_auc",
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
) -> Path:
    """Build and write ``best_by_block.csv`` from long rows."""

    best_rows = build_best_by_block_rows(
        rows,
        primary_metric=primary_metric,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    return _write_csv(best_rows, output, preferred_fields=BEST_BY_BLOCK_FIELDS)


def write_pairwise_winrate_csv(
    rows: list[dict[str, Any]],
    output: Path | str,
    *,
    metrics: Sequence[str] = ("pr_auc",),
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
) -> Path:
    """Build and write ``pairwise_winrate.csv`` from long rows."""

    winrate_rows = build_pairwise_winrate_rows(
        rows,
        metrics=metrics,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    return _write_csv(winrate_rows, output, preferred_fields=PAIRWISE_WINRATE_FIELDS)


def write_failure_report_csv(rows: list[dict[str, Any]], output: Path | str) -> Path:
    """Write ``failure_report.csv`` without hiding failed paired rows."""

    failure_rows = build_failure_report_rows(rows)
    return _write_csv(failure_rows, output, preferred_fields=FAILURE_FIELD_ORDER)


def write_summary_md(
    rows: list[dict[str, Any]],
    output: Path | str,
    *,
    config: Mapping[str, Any] | None = None,
    preflight: Mapping[str, Any] | None = None,
    metrics_wide_paired_rows: list[dict[str, Any]] | None = None,
    best_by_block_rows: list[dict[str, Any]] | None = None,
    pairwise_winrate_rows: list[dict[str, Any]] | None = None,
    failure_report_rows: list[dict[str, Any]] | None = None,
    report_paths: Mapping[str, Path | str] | None = None,
    primary_metric: str = "pr_auc",
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
) -> Path:
    """Write paired v2 ``summary.md`` with pairing sections and rationale."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    wide_rows = (
        metrics_wide_paired_rows
        if metrics_wide_paired_rows is not None
        else build_metrics_wide_paired_rows(rows, expected_sources=expected_sources, pair_key_fields=pair_key_fields)
    )
    best_rows = (
        best_by_block_rows
        if best_by_block_rows is not None
        else build_best_by_block_rows(
            rows,
            primary_metric=primary_metric,
            expected_sources=expected_sources,
            pair_key_fields=pair_key_fields,
        )
    )
    winrate_rows = (
        pairwise_winrate_rows
        if pairwise_winrate_rows is not None
        else build_pairwise_winrate_rows(
            rows,
            metrics=(primary_metric,),
            expected_sources=expected_sources,
            pair_key_fields=pair_key_fields,
        )
    )
    failure_rows = failure_report_rows if failure_report_rows is not None else build_failure_report_rows(rows)
    normalized = assert_paired_completeness(
        rows,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    paths = dict(report_paths or {})

    lines = [
        "# Paired Wavelet V2 Summary",
        "",
        "## Experiment Overview",
        "",
        "- narrative: v2 paired extension; v1 not overwritten.",
        *_summary_v2_extension_lines(config, preflight, expected_sources),
        "- outputs: v2 paired reports are written separately from the v1 output root.",
        "",
        "## Task and Population",
        "",
        *_summary_preflight_lines(config, preflight),
        "",
        "## Why Paired Comparison",
        "",
        "- comparison: paired Teacher/Ours rows",
        f"- long_rows: {len(normalized)}",
        f"- paired_rows: {len(wide_rows)}",
        f"- sources: {', '.join(_canonical_source(source) for source in expected_sources)}",
        "- completeness: every pair key has one Teacher row and one Ours row",
        "- failed rows: preserved in metrics_long.csv, metrics_wide_paired.csv, failure_report.csv, and this summary",
        "",
        "## Method Definitions",
        "",
        "- Teacher: hidden-dimension signal rows from the teacher-side wavelet comparison.",
        "- Ours: semantic trace rows from the layer-ordered wavelet summary comparison.",
        "- Both sources share the same block, pair_id, classifier, split, seed, model, and dataset before they are compared.",
        "",
        "## Wavelet Selection Rationale",
        "",
        "- wavelet rationale: hidden dimensions are unordered coordinates, but layers have a stable computation order.",
        "- Hidden dimension axis: a hidden coordinate order is not a physical sensor axis, so it should not be read as stable time or frequency.",
        "- Layer axis: layer order is meaningful computation depth, so Ours applies wavelet summaries to semantic traces over layers.",
        "- Paired grid: Teacher and Ours are compared only when they share the same block, pair_id, classifier, and run context.",
        "",
        "## Paired Results",
        "",
        *_summary_completion_lines(normalized, wide_rows, failure_rows),
        "",
        "### Output Files",
        "",
        *_summary_output_lines(paths),
        "",
        "### Paired Metric Rows",
        "",
        *_summary_full_paired_metric_lines(wide_rows, primary_metric),
        "",
        "The tables below report block winners and pairwise win rate from the same paired wide rows.",
        "",
        "### Best By Block",
        "",
        *_summary_best_by_block_lines(best_rows),
        "",
        "### Pairwise Win Rate",
        "",
        *_summary_winrate_lines(winrate_rows, primary_metric),
        "",
        "### Failures",
        "",
        *_summary_failure_lines(failure_rows),
        "",
        "## Interpretation",
        "",
        *_summary_interpretation_lines(best_rows, winrate_rows, failure_rows, primary_metric),
        "",
        "## Limitations",
        "",
        "- limitations: these counts describe comparable paired rows, not an unpaired sweep.",
        *_summary_limitations_lines(wide_rows, failure_rows),
        "",
        "## Conclusion",
        "",
        _summary_conclusion(best_rows, failure_rows, primary_metric),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_paired_reports(
    rows: list[dict[str, Any]],
    output_dir: Path | str,
    *,
    config: Mapping[str, Any] | None = None,
    preflight: Mapping[str, Any] | None = None,
    metrics: Sequence[str] = DEFAULT_COMPARE_METRICS,
    winrate_metrics: Sequence[str] = ("pr_auc",),
    primary_metric: str = "pr_auc",
    expected_sources: Sequence[str] = DEFAULT_SOURCES,
    expected_blocks: Sequence[str] | None = None,
    expected_pair_ids: Sequence[str] | None = None,
    pair_key_fields: Sequence[str] = DEFAULT_PAIR_KEY_FIELDS,
    metrics_ledger_path: Path | str | None = None,
) -> dict[str, Path]:
    """Validate paired completeness and write all paired report artifacts."""

    normalized = assert_paired_completeness(
        rows,
        expected_sources=expected_sources,
        expected_blocks=expected_blocks,
        expected_pair_ids=expected_pair_ids,
        pair_key_fields=pair_key_fields,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_long": directory / "metrics_long.csv",
        "metrics_wide_paired": directory / "metrics_wide_paired.csv",
        "best_by_block": directory / "best_by_block.csv",
        "pairwise_winrate": directory / "pairwise_winrate.csv",
        "failure_report": directory / "failure_report.csv",
        "summary": directory / "summary.md",
    }
    if metrics_ledger_path is not None:
        paths["metrics_ledger"] = Path(metrics_ledger_path)
    wide_rows = build_metrics_wide_paired_rows(
        rows,
        metrics=metrics,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
        primary_metric=primary_metric,
    )
    best_rows = build_best_by_block_rows(
        rows,
        primary_metric=primary_metric,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    winrate_rows = build_pairwise_winrate_rows(
        rows,
        metrics=winrate_metrics,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    failure_rows = build_failure_report_rows(rows)

    write_metrics_long_csv(normalized, paths["metrics_long"])
    _write_csv(
        wide_rows,
        paths["metrics_wide_paired"],
        preferred_fields=_wide_field_order(metrics, pair_key_fields, primary_metric),
    )
    _write_csv(best_rows, paths["best_by_block"], preferred_fields=BEST_BY_BLOCK_FIELDS)
    _write_csv(winrate_rows, paths["pairwise_winrate"], preferred_fields=PAIRWISE_WINRATE_FIELDS)
    _write_csv(failure_rows, paths["failure_report"], preferred_fields=FAILURE_FIELD_ORDER)
    write_summary_md(
        rows,
        paths["summary"],
        config=config,
        preflight=preflight,
        metrics_wide_paired_rows=wide_rows,
        best_by_block_rows=best_rows,
        pairwise_winrate_rows=winrate_rows,
        failure_report_rows=failure_rows,
        report_paths=paths,
        primary_metric=primary_metric,
        expected_sources=expected_sources,
        pair_key_fields=pair_key_fields,
    )
    return paths


def write_paired_metrics_csv(rows: list[dict[str, Any]], output: Path | str) -> Path:
    """Backward-compatible long CSV writer for paired metrics rows."""

    return write_metrics_long_csv(rows, output)


def write_paired_summary_md(
    rows: list[dict[str, Any]],
    output: Path | str,
    **kwargs: Any,
) -> Path:
    """Alias for callers that prefer an explicit paired summary function name."""

    return write_summary_md(rows, output, **kwargs)


def _normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["source"] = _row_source(copied)
        if "pair_id" not in copied and "paired_config_name" in copied:
            copied["pair_id"] = copied.get("paired_config_name", "")
        normalized.append(copied)
    return normalized


def _row_source(row: Mapping[str, Any]) -> str:
    for field in ("source", "paired_source", "pair_source", "method_source"):
        value = row.get(field)
        if str(value or "").strip():
            return _canonical_source(value)
    family = str(row.get("method_family", "")).strip()
    if family:
        return _canonical_source(family)
    raise ValueError("paired metrics row is missing source")


def _canonical_source(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("source must be a non-empty string")
    return SOURCE_ALIASES.get(text.lower(), text)


def _pair_key(row: Mapping[str, Any], pair_key_fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(_key_field_value(row, field) for field in pair_key_fields)


_MISSING_CONFIG_VALUE = object()


def _assert_paired_config_consistency(
    key: tuple[Any, ...],
    source_rows: Mapping[str, Mapping[str, Any]],
    source_order: Sequence[str],
) -> None:
    drift_parts: list[str] = []
    for label, fields in PAIRED_CONFIG_COMPARE_FIELD_GROUPS:
        values: dict[str, Any] = {}
        for source in source_order:
            value = _config_group_value(source_rows[source], fields)
            if value is not _MISSING_CONFIG_VALUE:
                values[source] = value
        if not values:
            continue
        if len(values) != len(source_order) or len(set(values.values())) != 1:
            source_values = ", ".join(
                f"{source}={_config_value_text(values.get(source, _MISSING_CONFIG_VALUE))}"
                for source in source_order
            )
            drift_parts.append(f"{label}({source_values})")
    if drift_parts:
        raise ValueError(f"paired config drift for key={key!r}: {'; '.join(drift_parts)}")


def _config_group_value(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    values: list[Any] = []
    for field in fields:
        if field not in row:
            continue
        value = row.get(field)
        if _is_missing_config_value(value):
            continue
        values.append(_normalize_config_value(value))
    if not values:
        return _MISSING_CONFIG_VALUE
    first = values[0]
    if any(value != first for value in values[1:]):
        field_text = "/".join(fields)
        raise ValueError(f"paired config row has conflicting aliases for {field_text}: {values!r}")
    return first


def _is_missing_config_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                return _normalize_config_value(json.loads(text))
            except json.JSONDecodeError:
                return text
        return text
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _normalize_config_value(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normalize_config_value(item) for item in value)
    return value


def _config_value_text(value: Any) -> str:
    if value is _MISSING_CONFIG_VALUE:
        return "<missing>"
    if isinstance(value, tuple):
        return json.dumps(value)
    return str(value)


def _key_field_value(row: Mapping[str, Any], field: str) -> Any:
    if field == "pair_id":
        return _pair_id(row)
    return row.get(field, "")


def _pair_id(row: Mapping[str, Any] | None) -> Any:
    if row is None:
        return ""
    return row.get("pair_id", row.get("paired_config_name", ""))


def _include_key_field(rows: list[dict[str, Any]], field: str) -> bool:
    if field in {"block", "pair_id", "classifier"}:
        return True
    return any(str(_key_field_value(row, field)) for row in rows)


def _metric_names(metrics: Sequence[str], primary_metric: str) -> tuple[str, ...]:
    names: list[str] = []
    for metric in [primary_metric, *list(metrics)]:
        text = str(metric)
        if text and text not in names:
            names.append(text)
    return tuple(names)


def _representative_row(
    source_rows: Mapping[str, dict[str, Any]],
    source_order: Sequence[str],
) -> dict[str, Any]:
    for source in source_order:
        if source in source_rows:
            return source_rows[source]
    return next(iter(source_rows.values()))


def _status(row: Mapping[str, Any]) -> str:
    value = str(row.get("status", "") or "").strip()
    return value if value else "unknown"


def _is_success(row: Mapping[str, Any]) -> bool:
    return _status(row).lower() == "success"


def _is_failure(row: Mapping[str, Any]) -> bool:
    status = _status(row).lower()
    if status == "success":
        return bool(str(row.get("failure_reason", "") or "").strip())
    return status != "unknown" or bool(str(row.get("failure_reason", "") or "").strip())


def _paired_status(statuses: Sequence[str]) -> str:
    lowered = [status.lower() for status in statuses]
    if all(status == "success" for status in lowered):
        return "success"
    if all(status != "success" for status in lowered):
        return "failure"
    return "partial_failure"


def _paired_failure_reason(
    source_rows: Mapping[str, Mapping[str, Any]],
    source_order: Sequence[str],
) -> str:
    parts: list[str] = []
    for source in source_order:
        row = source_rows[source]
        if _is_success(row):
            continue
        reason = str(row.get("failure_reason", "") or "").strip() or _status(row)
        parts.append(f"{source}: {reason}")
    return "; ".join(parts)


def _source_prefix(source: str) -> str:
    return _canonical_source(source).lower().replace("-", "_")


def _finite_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _metric_winner(metric: str, teacher_value: float | None, ours_value: float | None) -> str:
    if teacher_value is None or ours_value is None:
        return ""
    if _metric_is_better(metric, ours_value, teacher_value):
        return "Ours"
    if _metric_is_better(metric, teacher_value, ours_value):
        return "Teacher"
    return "tie"


def _best_row(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        value = _finite_float(row.get(metric))
        if value is None:
            continue
        scored.append((value, str(row.get("config_name", "")), row))
    if not scored:
        return None
    scored.sort(key=lambda item: (_metric_rank(metric, item[0]), item[1]))
    return scored[0][2]


def _metric_is_better(metric: str, candidate: float, current: float) -> bool:
    if metric in LOWER_IS_BETTER_METRICS:
        return candidate < current
    return candidate > current


def _metric_rank(metric: str, value: float) -> float:
    if metric in LOWER_IS_BETTER_METRICS:
        return value
    return -value


def _is_comparable_source_group(
    source_rows: Mapping[str, Mapping[str, Any]],
    source_order: Sequence[str],
    metric: str,
) -> bool:
    for source in source_order:
        row = source_rows.get(source)
        if row is None or not _is_success(row):
            return False
        if _finite_float(row.get(metric)) is None:
            return False
    return True


def _winrate_row(block: str, metric: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ours_wins = 0
    teacher_wins = 0
    ties = 0
    comparable = 0
    both_success = 0
    teacher_success_ours_fail = 0
    ours_success_teacher_fail = 0
    both_fail = 0
    for row in rows:
        teacher_success = str(row.get("teacher_status", "")).lower() == "success"
        ours_success = str(row.get("ours_status", "")).lower() == "success"
        if teacher_success and ours_success:
            both_success += 1
        elif teacher_success and not ours_success:
            teacher_success_ours_fail += 1
        elif ours_success and not teacher_success:
            ours_success_teacher_fail += 1
        else:
            both_fail += 1
        if not teacher_success or not ours_success:
            continue
        teacher_value = _finite_float(row.get(f"teacher_{metric}"))
        ours_value = _finite_float(row.get(f"ours_{metric}"))
        if teacher_value is None or ours_value is None:
            continue
        comparable += 1
        winner = _metric_winner(metric, teacher_value, ours_value)
        if winner == "Ours":
            ours_wins += 1
        elif winner == "Teacher":
            teacher_wins += 1
        else:
            ties += 1
    total = len(rows)
    pr_auc_summary = _delta_summary(rows, "pr_auc")
    f1_summary = _delta_summary(rows, "test_f1")
    return {
        "block": block,
        "metric": metric,
        "num_pairs": total,
        "num_both_success": both_success,
        "num_teacher_success_ours_fail": teacher_success_ours_fail,
        "num_ours_success_teacher_fail": ours_success_teacher_fail,
        "num_both_fail": both_fail,
        "ours_wins_by_pr_auc": _count_wins(rows, "Ours", "pr_auc"),
        "teacher_wins_by_pr_auc": _count_wins(rows, "Teacher", "pr_auc"),
        "mean_delta_pr_auc": pr_auc_summary["mean"],
        "median_delta_pr_auc": pr_auc_summary["median"],
        "mean_delta_f1": f1_summary["mean"],
        "median_delta_f1": f1_summary["median"],
        "total_pairs": total,
        "comparable_pairs": comparable,
        "ours_wins": ours_wins,
        "teacher_wins": teacher_wins,
        "ties": ties,
        "not_comparable_pairs": total - comparable,
        "ours_winrate": _rate(ours_wins, comparable),
        "teacher_winrate": _rate(teacher_wins, comparable),
        "tie_rate": _rate(ties, comparable),
    }


def _rate(count: int, total: int) -> str:
    if total <= 0:
        return ""
    return f"{count / total:.6f}"


def _count_wins(rows: Sequence[Mapping[str, Any]], winner: str, metric: str) -> int:
    count = 0
    for row in rows:
        if str(row.get("teacher_status", "")).lower() != "success":
            continue
        if str(row.get("ours_status", "")).lower() != "success":
            continue
        teacher_value = _finite_float(row.get(f"teacher_{metric}"))
        ours_value = _finite_float(row.get(f"ours_{metric}"))
        if teacher_value is None or ours_value is None:
            continue
        if _metric_winner(metric, teacher_value, ours_value) == winner:
            count += 1
    return count


def _delta_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float | str]:
    deltas: list[float] = []
    for row in rows:
        if str(row.get("teacher_status", "")).lower() != "success":
            continue
        if str(row.get("ours_status", "")).lower() != "success":
            continue
        teacher_value = _finite_float(row.get(f"teacher_{metric}"))
        ours_value = _finite_float(row.get(f"ours_{metric}"))
        if teacher_value is None or ours_value is None:
            continue
        deltas.append(ours_value - teacher_value)
    if not deltas:
        return {"mean": "", "median": ""}
    ordered = sorted(deltas)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {"mean": sum(deltas) / len(deltas), "median": median}


def _with_long_metric_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    aliased = dict(row)
    for canonical, alias in LONG_METRIC_ALIASES.items():
        canonical_present = canonical in aliased and not _is_empty_report_value(aliased.get(canonical))
        alias_present = alias in aliased and not _is_empty_report_value(aliased.get(alias))
        if canonical_present and not alias_present:
            aliased[alias] = aliased.get(canonical)
        elif alias_present and not canonical_present:
            aliased[canonical] = aliased.get(alias)
    return aliased


def _shared_config_summary(row: Mapping[str, Any]) -> str:
    fields = (
        "transform",
        "wavelet",
        "level",
        "cwt_scales",
        "threshold",
        "window_mode",
        "window_strategy",
        "window_size",
        "stride",
        "feature_protocol",
        "classifier",
        "sequence_model",
        "learning_rate",
    )
    parts: list[str] = []
    for field in fields:
        value = row.get(field)
        if _is_empty_report_value(value):
            continue
        parts.append(f"{field}={_csv_value(value)}")
    return "; ".join(parts)


def _runtime_ratio(teacher_seconds: Any, ours_seconds: Any) -> float | str:
    teacher_value = _finite_float(teacher_seconds)
    ours_value = _finite_float(ours_seconds)
    if teacher_value is None or ours_value is None or ours_value <= 0:
        return ""
    return teacher_value / ours_value


def _is_empty_report_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _write_csv(
    rows: list[dict[str, Any]],
    output: Path | str,
    *,
    preferred_fields: Sequence[str],
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows, preferred_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
    return path


def _fieldnames(rows: Sequence[Mapping[str, Any]], preferred_fields: Sequence[str]) -> list[str]:
    names: list[str] = []
    for field in preferred_fields:
        if field not in names:
            names.append(str(field))
    for row in rows:
        for field in row:
            if field not in names:
                names.append(str(field))
    return names


def _wide_field_order(
    metrics: Sequence[str],
    pair_key_fields: Sequence[str],
    primary_metric: str = "pr_auc",
) -> tuple[str, ...]:
    fields: list[str] = [field for field in pair_key_fields if field in WIDE_BASE_FIELDS]
    for field in WIDE_BASE_FIELDS:
        if field not in fields:
            fields.append(field)
    metric_names = _metric_names(metrics, primary_metric)
    for source in DEFAULT_SOURCES:
        prefix = _source_prefix(source)
        for field in SOURCE_WIDE_FIELDS:
            fields.append(f"{prefix}_{field}")
        for metric in metric_names:
            fields.append(f"{prefix}_{metric}")
    for metric in metric_names:
        fields.append(f"delta_{metric}")
        fields.append(f"winner_{metric}")
    fields.extend(("teacher_f1", "ours_f1", "delta_f1"))
    fields.append("speedup_or_slowdown")
    return tuple(fields)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _paired_key_sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(item) for item in key)


def _block_sort_key(block: object) -> tuple[int, str]:
    text = str(block)
    return BLOCK_ORDER.get(text, 99), text


def _summary_output_lines(paths: Mapping[str, Path | str]) -> list[str]:
    if not paths:
        return [
            "- metrics_ledger.csv",
            "- metrics_long.csv",
            "- metrics_wide_paired.csv",
            "- best_by_block.csv",
            "- pairwise_winrate.csv",
            "- failure_report.csv",
            "- summary.md",
        ]
    labels = (
        ("metrics_ledger", "metrics_ledger.csv"),
        ("metrics_long", "metrics_long.csv"),
        ("metrics_wide_paired", "metrics_wide_paired.csv"),
        ("best_by_block", "best_by_block.csv"),
        ("pairwise_winrate", "pairwise_winrate.csv"),
        ("failure_report", "failure_report.csv"),
        ("summary", "summary.md"),
    )
    return [f"- {display}: {paths.get(key, display)}" for key, display in labels]


def _summary_v2_extension_lines(
    config: Mapping[str, Any] | None,
    preflight: Mapping[str, Any] | None,
    expected_sources: Sequence[str],
) -> list[str]:
    cfg = _safe_mapping(config)
    paired = _safe_mapping(cfg.get("paired_wavelet_v2"))
    population_grid = _safe_mapping(cfg.get("population_grid"))
    population_summary = _safe_mapping(_safe_mapping(preflight).get("population_summary"))
    source = (
        str(population_grid.get("source", "") or population_summary.get("split_source", "")).strip()
        or "not_provided"
    )
    lines = [
        "- extension: paired Teacher/Ours wavelet-course v2",
        f"- v1_preservation: {source}",
        f"- expected_sources: {', '.join(_canonical_source(source) for source in expected_sources)}",
    ]
    description = str(paired.get("description", "")).strip()
    if description:
        lines.append(f"- description: {description}")
    return lines


def _summary_preflight_lines(
    config: Mapping[str, Any] | None,
    preflight: Mapping[str, Any] | None,
) -> list[str]:
    pf = _safe_mapping(preflight)
    if not pf:
        return ["- No preflight metadata was provided."]

    cfg = _safe_mapping(config)
    cache = _safe_mapping(pf.get("cache_audit"))
    population = _safe_mapping(pf.get("population_summary"))
    split_validation = _safe_mapping(pf.get("split_validation"))
    paired_grid = _safe_mapping(pf.get("paired_grid_audit"))
    sample_grid = _safe_mapping(pf.get("sample_grid_audit"))
    lines: list[str] = []

    task = _task_text(cfg)
    if task:
        lines.append(f"- task: {task}")
    lines.append(f"- cache_accepted: {str(bool(cache.get('accepted', False))).lower()}")
    lines.append(f"- cache_entries: {cache.get('num_entries', 0)}")
    subset_counts = _count_mapping_text(cache.get("subsets"))
    if subset_counts:
        lines.append(f"- task_subset_counts: {subset_counts}")
    lines.append(f"- primary_population: {population.get('num_primary_population', 0)}")
    lines.append(f"- hard_hallucinations: {population.get('num_hard_hallucination', 0)}")
    lines.append(f"- correct: {population.get('num_correct', 0)}")
    if population.get("split_source"):
        lines.append(f"- split_source: {population.get('split_source')}")
    lines.extend(_summary_split_count_lines(split_validation))
    lines.extend(_summary_paired_grid_lines(paired_grid))
    lines.extend(_summary_sample_grid_lines(sample_grid))
    metrics_ledger_path = str(pf.get("metrics_ledger_path", "") or "").strip()
    if metrics_ledger_path:
        lines.append(f"- metrics_ledger.csv: {metrics_ledger_path}")
    return lines


def _task_text(config: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field in ("model_name", "dataset_name"):
        value = str(config.get(field, "")).strip()
        if value:
            parts.append(f"{field}={value}")
    subsets = config.get("subsets")
    if isinstance(subsets, Sequence) and not isinstance(subsets, (str, bytes)):
        text = ",".join(str(item) for item in subsets)
        if text:
            parts.append(f"subsets={text}")
    return " ".join(parts)


def _summary_split_count_lines(split_validation: Mapping[str, Any]) -> list[str]:
    counts = _safe_mapping(split_validation.get("counts"))
    lines: list[str] = []
    for split in SUMMARY_SPLITS:
        split_counts = _safe_mapping(counts.get(split))
        if split_counts:
            lines.append(
                f"- split_{split}: pos={split_counts.get('pos', 0)} neg={split_counts.get('neg', 0)}"
            )
    return lines


def _summary_paired_grid_lines(paired_grid: Mapping[str, Any]) -> list[str]:
    if not paired_grid:
        return []
    lines = [
        f"- paired_grid_path: {paired_grid.get('paired_grid_path', '')}",
        f"- paired_grid_rows: {paired_grid.get('num_pair_rows', 0)}",
        f"- paired_grid_pair_ids: {paired_grid.get('num_pair_ids', 0)}",
    ]
    blocks = _sequence_text(paired_grid.get("requested_blocks"))
    if blocks:
        lines.append(f"- paired_grid_blocks: {blocks}")
    return lines


def _summary_sample_grid_lines(sample_grid: Mapping[str, Any]) -> list[str]:
    if not sample_grid:
        return []
    path = str(
        sample_grid.get("sample_grid_path", "")
        or sample_grid.get("selected_sample_grid_path", "")
    )
    rows = sample_grid.get("num_rows", sample_grid.get("selected_num_rows", 0))
    row_hash = sample_grid.get("row_order_hash", sample_grid.get("selected_row_order_hash", ""))
    return [
        f"- sample_grid_path: {path}",
        f"- sample_grid_rows: {rows}",
        f"- sample_grid_row_order_hash: {row_hash}",
    ]


def _summary_limitations_lines(
    wide_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    partial_rows = sum(1 for row in wide_rows if str(row.get("paired_status", "")) != "success")
    return [
        "- paired comparisons use only exact Teacher/Ours grid matches.",
        "- failed configs remain in metrics_long.csv, metrics_wide_paired.csv, and failure_report.csv.",
        f"- non_comparable_paired_rows: {partial_rows}",
        f"- failed_config_rows: {len(failure_rows)}",
    ]


def _summary_completion_lines(
    long_rows: Sequence[Mapping[str, Any]],
    wide_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    success_count = sum(1 for row in long_rows if _is_success(row))
    return [
        f"- metrics_long_rows: {len(long_rows)}",
        f"- metrics_wide_paired_rows: {len(wide_rows)}",
        f"- success_rows: {success_count}",
        f"- non_success_rows: {len(failure_rows)}",
        "- paired_completeness: passed",
    ]


def _summary_full_paired_metric_lines(
    rows: Sequence[Mapping[str, Any]],
    primary_metric: str,
) -> list[str]:
    if not rows:
        return ["- No paired wide rows were recorded."]
    teacher_metric = f"teacher_{primary_metric}"
    ours_metric = f"ours_{primary_metric}"
    delta_metric = f"delta_{primary_metric}"
    winner_metric = f"winner_{primary_metric}"
    lines = [
        "| block | pair_id | classifier | teacher_status | ours_status | teacher_value | ours_value | delta | winner | paired_failure_reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {block} | {pair} | {classifier} | {teacher_status} | {ours_status} | {teacher_value} | "
            "{ours_value} | {delta} | {winner} | {reason} |".format(
                block=_md_cell(row.get("block", "")),
                pair=_md_cell(_pair_id(row)),
                classifier=_md_cell(row.get("classifier", "")),
                teacher_status=_md_cell(row.get("teacher_status", "")),
                ours_status=_md_cell(row.get("ours_status", "")),
                teacher_value=_md_cell(row.get(teacher_metric, "")),
                ours_value=_md_cell(row.get(ours_metric, "")),
                delta=_md_cell(row.get(delta_metric, "")),
                winner=_md_cell(row.get(winner_metric, "")),
                reason=_md_cell(row.get("paired_failure_reason", "")),
            )
        )
    return lines


def _summary_best_by_block_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["- No successful block-level metric rows were available."]
    lines = [
        "| block | selection_scope | comparable_pairs | not_comparable_pairs | best_source | best_pair_id | best_config_name | best_value | failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {block} | {scope} | {comparable} | {not_comparable} | {source} | {pair} | {config} | {value} | {failures} |".format(
                block=_md_cell(row.get("block", "")),
                scope=_md_cell(row.get("selection_scope", "")),
                comparable=_md_cell(row.get("comparable_pairs", "")),
                not_comparable=_md_cell(row.get("not_comparable_pairs", "")),
                source=_md_cell(row.get("best_source", "")),
                pair=_md_cell(row.get("best_pair_id", "")),
                config=_md_cell(row.get("best_config_name", "")),
                value=_md_cell(row.get("best_value", "")),
                failures=_md_cell(row.get("failure_rows", "")),
            )
        )
    return lines


def _summary_winrate_lines(rows: Sequence[Mapping[str, Any]], primary_metric: str) -> list[str]:
    selected = [row for row in rows if str(row.get("metric", "")) == primary_metric]
    if not selected:
        return [f"- No pairwise win-rate rows were available for {primary_metric}."]
    lines = [
        "| block | comparable_pairs | ours_wins | teacher_wins | ties | not_comparable_pairs | ours_winrate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected:
        lines.append(
            "| {block} | {comparable} | {ours} | {teacher} | {ties} | {not_comparable} | {winrate} |".format(
                block=_md_cell(row.get("block", "")),
                comparable=_md_cell(row.get("comparable_pairs", "")),
                ours=_md_cell(row.get("ours_wins", "")),
                teacher=_md_cell(row.get("teacher_wins", "")),
                ties=_md_cell(row.get("ties", "")),
                not_comparable=_md_cell(row.get("not_comparable_pairs", "")),
                winrate=_md_cell(row.get("ours_winrate", "")),
            )
        )
    return lines


def _summary_failure_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["- None recorded."]
    lines = [
        "| block | pair_id | source | config_name | status | failure_reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {block} | {pair} | {source} | {config} | {status} | {reason} |".format(
                block=_md_cell(row.get("block", "")),
                pair=_md_cell(_pair_id(row)),
                source=_md_cell(row.get("source", "")),
                config=_md_cell(row.get("config_name", "")),
                status=_md_cell(row.get("status", "")),
                reason=_md_cell(row.get("failure_reason", "")),
            )
        )
    return lines


def _summary_interpretation_lines(
    best_rows: Sequence[Mapping[str, Any]],
    winrate_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    primary_metric: str,
) -> list[str]:
    best = _best_summary_row(best_rows, primary_metric)
    if best is None:
        best_line = f"- paired best: no comparable paired row had a numeric {primary_metric}."
    else:
        best_line = (
            f"- paired best: block {best.get('block', '')}, source {best.get('best_source', '')}, "
            f"pair {best.get('best_pair_id', '')}, value {best.get('best_value', '')}."
        )

    overall = next(
        (
            row
            for row in winrate_rows
            if str(row.get("metric", "")) == primary_metric and str(row.get("block", "")) == "overall"
        ),
        None,
    )
    if overall is None:
        winrate_line = f"- winrate: no pairwise winrate row was available for {primary_metric}."
    else:
        winrate_line = (
            f"- winrate: Ours wins {overall.get('ours_wins', 0)}, "
            f"Teacher wins {overall.get('teacher_wins', 0)}, ties {overall.get('ties', 0)}, "
            f"comparable pairs {overall.get('comparable_pairs', 0)}, "
            f"not comparable {overall.get('not_comparable_pairs', 0)}, "
            f"Ours winrate {overall.get('ours_winrate', '')}."
        )

    return [
        best_line,
        winrate_line,
        f"- failure counts: {len(failure_rows)} non-success config rows are retained.",
    ]


def _summary_conclusion(
    best_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    primary_metric: str,
) -> str:
    best = _best_summary_row(best_rows, primary_metric)
    if best is None:
        base = f"No block has a comparable paired row with a numeric {primary_metric} yet."
    else:
        base = (
            f"The strongest block-level {primary_metric} row is block {best.get('block', '')}, "
            f"{best.get('best_source', '')} on {best.get('best_pair_id', '')}."
        )
    if failure_rows:
        return f"{base} {len(failure_rows)} non-success rows remain part of the paired report."
    return base


def _best_summary_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        value = _finite_float(row.get("best_value"))
        if value is not None:
            scored.append((value, row))
    if not scored:
        return None
    scored.sort(key=lambda item: _metric_rank(metric, item[0]))
    return scored[0][1]


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _count_mapping_text(value: Any) -> str:
    counts = _safe_mapping(value)
    parts = [f"{key}={counts[key]}" for key in sorted(counts)]
    return ", ".join(parts)


def _sequence_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ", ".join(str(item) for item in value)
    return ""


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


__all__ = [
    "BEST_BY_BLOCK_FIELDS",
    "DEFAULT_COMPARE_METRICS",
    "DEFAULT_PAIR_KEY_FIELDS",
    "DEFAULT_SOURCES",
    "FAILURE_FIELD_ORDER",
    "LONG_FIELD_ORDER",
    "PAIRWISE_WINRATE_FIELDS",
    "WIDE_BASE_FIELDS",
    "assert_paired_completeness",
    "build_best_by_block_rows",
    "build_failure_report_rows",
    "build_metrics_wide_paired_rows",
    "build_pairwise_winrate_rows",
    "write_best_by_block_csv",
    "write_failure_report_csv",
    "write_metrics_long_csv",
    "write_metrics_wide_paired_csv",
    "write_paired_metrics_csv",
    "write_paired_reports",
    "write_paired_summary_md",
    "write_pairwise_winrate_csv",
    "write_summary_md",
]
