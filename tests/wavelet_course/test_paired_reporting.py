from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest


def _paired_metric_rows() -> list[dict[str, object]]:
    common = {
        "run_id": "paired_wavelet_v2",
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subset_scope": "popular,random,adversarial",
        "seed": 20260506,
        "block": "A",
        "pair_id": "A_none_stat28_full",
        "classifier": "logreg",
        "transform": "none",
        "feature_protocol": "stat28",
        "window_strategy": "full",
        "train_samples": 6,
        "val_samples": 2,
        "test_samples": 2,
        "train_pos": 3,
        "val_pos": 1,
        "test_pos": 1,
    }
    return [
        {
            **common,
            "source": "Teacher",
            "row_id": "A_none_stat28_full::Teacher",
            "signal_builder": "teacher_hidden_dim_signal",
            "config_name": "A_none_stat28_full::Teacher::logreg",
            "status": "success",
            "failure_reason": "",
            "pr_auc": 0.55,
        },
        {
            **common,
            "source": "Ours",
            "row_id": "A_none_stat28_full::Ours",
            "signal_builder": "ours_semantic_trace_signal",
            "config_name": "A_none_stat28_full::Ours::logreg",
            "status": "failure",
            "failure_reason": "xgboost_not_installed",
            "pr_auc": "",
        },
    ]


def _successful_pair_rows(
    *,
    pair_id: str = "A_dwt_db2_l2",
    block: str = "A",
    teacher_value: float = 0.55,
    ours_value: float = 0.75,
    metric: str = "pr_auc",
    classifier: str = "logreg",
) -> list[dict[str, object]]:
    common = {
        "run_id": "paired_wavelet_v2",
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subset_scope": "popular,random,adversarial",
        "seed": 20260506,
        "block": block,
        "pair_id": pair_id,
        "classifier": classifier,
        "transform": "dwt",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "db2",
        "level": 2,
        "threshold": "universal_soft",
        "sequence_model": "",
        "window_mode": "global",
        "window_strategy": "full",
        "window_size": "",
        "stride": "",
        "cwt_scales": [1.0, 2.0, 3.0],
        "status": "success",
        "failure_reason": "",
    }
    return [
        {
            **common,
            "source": "Teacher",
            "row_id": f"{pair_id}::Teacher",
            "signal_builder": "teacher_hidden_dim_signal",
            "config_name": f"{pair_id}::Teacher::{classifier}",
            metric: teacher_value,
        },
        {
            **common,
            "source": "Ours",
            "row_id": f"{pair_id}::Ours",
            "signal_builder": "ours_semantic_trace_signal",
            "config_name": f"{pair_id}::Ours::{classifier}",
            metric: ours_value,
        },
    ]


def test_paired_metrics_long_csv_preserves_failure_rows(tmp_path: Path) -> None:
    from mind.wavelet_course.paired_reporting import write_paired_metrics_csv

    output = tmp_path / "metrics_long.csv"
    write_paired_metrics_csv(
        [
            {
                "run_id": "paired_wavelet_v2",
                "pair_id": "A_none_stat28_full",
                "source": "Teacher",
                "status": "success",
                "failure_reason": "",
                "pr_auc": 0.5,
                "average_precision": 0.51,
                "roc_auc": 0.52,
            },
            {
                "run_id": "paired_wavelet_v2",
                "pair_id": "A_none_stat28_full",
                "source": "Ours",
                "status": "failure",
                "failure_reason": "xgboost_not_installed",
                "pr_auc": "",
                "average_precision": "",
                "roc_auc": "",
            },
        ],
        output,
    )

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert [row["source"] for row in rows] == ["Teacher", "Ours"]
    assert rows[1]["status"] == "failure"
    assert rows[1]["failure_reason"] == "xgboost_not_installed"
    assert rows[0]["test_pr_auc"] == "0.5"
    assert rows[0]["test_average_precision"] == "0.51"
    assert rows[0]["test_roc_auc"] == "0.52"
    assert rows[1]["test_pr_auc"] == ""


def test_sequence_status_fields_are_part_of_long_and_wide_reporting_contract() -> None:
    from mind.wavelet_course.paired_reporting import (
        DEFAULT_PAIR_KEY_FIELDS,
        LONG_FIELD_ORDER,
        SOURCE_WIDE_FIELDS,
    )

    for field in (
        "learning_rate",
        "max_epochs",
        "patience",
        "best_epoch",
        "early_stopped",
        "converged",
        "max_epoch_reached",
    ):
        assert field in LONG_FIELD_ORDER
    for field in (
        "learning_rate",
        "best_epoch",
        "early_stopped",
        "converged",
        "max_epoch_reached",
    ):
        assert field in SOURCE_WIDE_FIELDS
    assert "learning_rate" in DEFAULT_PAIR_KEY_FIELDS


def test_paired_wide_report_preserves_failures_and_compares_success_rows() -> None:
    from mind.wavelet_course.paired_reporting import build_metrics_wide_paired_rows

    rows = [
        {
            "run_id": "paired_wavelet_v2",
            "model_name": "qwen3-vl-8b",
            "dataset_name": "repope",
            "subset_scope": "popular,random,adversarial",
            "seed": 20260506,
            "block": "A",
            "pair_id": "A_none_stat28_full",
            "source": "Teacher",
            "classifier": "logreg",
            "config_name": "teacher_A_none_stat28_full_logreg",
            "status": "success",
            "failure_reason": "",
            "pr_auc": 0.55,
        },
        {
            "run_id": "paired_wavelet_v2",
            "model_name": "qwen3-vl-8b",
            "dataset_name": "repope",
            "subset_scope": "popular,random,adversarial",
            "seed": 20260506,
            "block": "A",
            "pair_id": "A_none_stat28_full",
            "source": "Ours",
            "classifier": "logreg",
            "config_name": "ours_A_none_stat28_full_logreg",
            "status": "success",
            "failure_reason": "",
            "pr_auc": 0.75,
        },
        {
            "run_id": "paired_wavelet_v2",
            "model_name": "qwen3-vl-8b",
            "dataset_name": "repope",
            "subset_scope": "popular,random,adversarial",
            "seed": 20260506,
            "block": "D",
            "pair_id": "D_swt_db2_l2_summary",
            "source": "Teacher",
            "classifier": "xgb",
            "config_name": "teacher_D_swt_db2_l2_summary_xgb",
            "status": "success",
            "failure_reason": "",
            "pr_auc": 0.60,
        },
        {
            "run_id": "paired_wavelet_v2",
            "model_name": "qwen3-vl-8b",
            "dataset_name": "repope",
            "subset_scope": "popular,random,adversarial",
            "seed": 20260506,
            "block": "D",
            "pair_id": "D_swt_db2_l2_summary",
            "source": "Ours",
            "classifier": "xgb",
            "config_name": "ours_D_swt_db2_l2_summary_xgb",
            "status": "failure",
            "failure_reason": "xgboost_not_installed",
            "pr_auc": "",
        },
    ]

    wide_rows = build_metrics_wide_paired_rows(rows, metrics=("pr_auc",))

    assert [row["pair_id"] for row in wide_rows] == [
        "A_none_stat28_full",
        "D_swt_db2_l2_summary",
    ]
    assert wide_rows[0]["paired_status"] == "success"
    assert wide_rows[0]["shared_config_summary"]
    assert wide_rows[0]["teacher_pr_auc"] == 0.55
    assert wide_rows[0]["ours_pr_auc"] == 0.75
    assert wide_rows[0]["delta_pr_auc"] == pytest.approx(0.20)
    assert wide_rows[0]["winner_pr_auc"] == "Ours"
    assert "speedup_or_slowdown" in wide_rows[0]
    assert wide_rows[1]["paired_status"] == "partial_failure"
    assert wide_rows[1]["ours_failure_reason"] == "xgboost_not_installed"
    assert wide_rows[1]["delta_pr_auc"] == ""


@pytest.mark.parametrize(
    ("field", "teacher_value", "ours_value"),
    [
        ("transform", "dwt", "swt"),
        ("wavelet", "db2", "haar"),
        ("level", 2, 3),
        ("cwt_scales", [1.0, 2.0], [1.0, 3.0]),
        ("threshold", "none", "universal_soft"),
        ("window_mode", "global", "win4_s4"),
        ("window_strategy", "full", "sliding"),
        ("window_size", 4, 6),
        ("stride", 4, 3),
        ("feature_protocol", "stat28", "raw_sequence"),
        ("classifier", "logreg", "rf"),
        ("sequence_model", "", "lstm_projected"),
        ("seed", 20260506, 20260507),
        ("wavelet_split", "train", "test"),
    ],
)
def test_paired_completeness_rejects_shared_config_drift(
    field: str,
    teacher_value: object,
    ours_value: object,
) -> None:
    from mind.wavelet_course.paired_reporting import assert_paired_completeness

    rows = _successful_pair_rows()
    rows[0][field] = teacher_value
    rows[1][field] = ours_value

    pair_key_fields: Sequence[str] = (
        "run_id",
        "model_name",
        "dataset_name",
        "subset_scope",
        "block",
        "pair_id",
    )
    with pytest.raises(ValueError, match="paired config drift"):
        assert_paired_completeness(rows, pair_key_fields=pair_key_fields)


def test_fpr_at_95pct_tpr_is_lower_is_better_for_winners_and_best_rows() -> None:
    from mind.wavelet_course.paired_reporting import (
        build_best_by_block_rows,
        build_metrics_wide_paired_rows,
        build_pairwise_winrate_rows,
    )

    rows = _successful_pair_rows(
        pair_id="A_dwt_db2_l2_fpr",
        teacher_value=0.20,
        ours_value=0.10,
        metric="fpr_at_95pct_tpr",
    )

    wide_rows = build_metrics_wide_paired_rows(rows, metrics=("fpr_at_95pct_tpr",))
    winrate_rows = build_pairwise_winrate_rows(rows, metrics=("fpr_at_95pct_tpr",))
    best_rows = build_best_by_block_rows(rows, primary_metric="fpr_at_95pct_tpr")

    assert wide_rows[0]["delta_fpr_at_95pct_tpr"] == pytest.approx(-0.10)
    assert wide_rows[0]["winner_fpr_at_95pct_tpr"] == "Ours"
    assert winrate_rows[-1]["ours_wins"] == 1
    assert winrate_rows[-1]["teacher_wins"] == 0
    assert best_rows[0]["best_source"] == "Ours"
    assert best_rows[0]["best_value"] == 0.10


def test_best_by_block_uses_only_comparable_both_success_pairs() -> None:
    from mind.wavelet_course.paired_reporting import build_best_by_block_rows

    partial = _successful_pair_rows(
        pair_id="A_partial_high_teacher",
        teacher_value=0.99,
        ours_value=0.10,
    )
    partial[1]["status"] = "failure"
    partial[1]["failure_reason"] = "xgboost_not_installed"
    partial[1]["pr_auc"] = ""
    comparable = _successful_pair_rows(
        pair_id="A_comparable_lower_but_valid",
        teacher_value=0.60,
        ours_value=0.70,
    )

    best_rows = build_best_by_block_rows([*partial, *comparable], primary_metric="pr_auc")

    assert best_rows == [
        {
            **best_rows[0],
            "paired_rows": 2,
            "comparable_pairs": 1,
            "not_comparable_pairs": 1,
            "selection_scope": "comparable_both_success",
            "best_source": "Ours",
            "best_pair_id": "A_comparable_lower_but_valid",
            "best_value": 0.70,
            "teacher_best_pair_id": "A_comparable_lower_but_valid",
            "ours_best_pair_id": "A_comparable_lower_but_valid",
        }
    ]


def test_paired_reports_preserve_failed_configs_in_all_failure_outputs(tmp_path: Path) -> None:
    from mind.wavelet_course.paired_reporting import write_paired_reports

    paths = write_paired_reports(_paired_metric_rows(), tmp_path)

    long_rows = list(csv.DictReader(paths["metrics_long"].open(newline="", encoding="utf-8")))
    wide_rows = list(csv.DictReader(paths["metrics_wide_paired"].open(newline="", encoding="utf-8")))
    best_rows = list(csv.DictReader(paths["best_by_block"].open(newline="", encoding="utf-8")))
    winrate_rows = list(csv.DictReader(paths["pairwise_winrate"].open(newline="", encoding="utf-8")))
    failure_rows = list(csv.DictReader(paths["failure_report"].open(newline="", encoding="utf-8")))
    summary = paths["summary"].read_text(encoding="utf-8")

    assert [row["config_name"] for row in long_rows] == [
        "A_none_stat28_full::Teacher::logreg",
        "A_none_stat28_full::Ours::logreg",
    ]
    assert wide_rows[0]["ours_config_name"] == "A_none_stat28_full::Ours::logreg"
    assert wide_rows[0]["ours_status"] == "failure"
    assert wide_rows[0]["paired_status"] == "partial_failure"
    assert failure_rows == [
        {
            **failure_rows[0],
            "config_name": "A_none_stat28_full::Ours::logreg",
            "failure_reason": "xgboost_not_installed",
            "status": "failure",
        }
    ]
    assert best_rows[0]["paired_rows"] == "1"
    assert best_rows[0]["comparable_pairs"] == "0"
    assert best_rows[0]["not_comparable_pairs"] == "1"
    assert best_rows[0]["selection_scope"] == "comparable_both_success"
    assert best_rows[0]["best_source"] == ""
    assert winrate_rows == [
        {
            **winrate_rows[0],
            "block": "A",
            "metric": "pr_auc",
            "num_pairs": "1",
            "total_pairs": "1",
            "num_both_success": "0",
            "num_teacher_success_ours_fail": "1",
            "num_ours_success_teacher_fail": "0",
            "num_both_fail": "0",
            "comparable_pairs": "0",
            "ours_wins_by_pr_auc": "0",
            "teacher_wins_by_pr_auc": "0",
            "ours_wins": "0",
            "teacher_wins": "0",
            "ties": "0",
            "not_comparable_pairs": "1",
            "mean_delta_pr_auc": "",
            "median_delta_pr_auc": "",
            "mean_delta_f1": "",
            "median_delta_f1": "",
            "ours_winrate": "",
            "teacher_winrate": "",
            "tie_rate": "",
        },
        {
            **winrate_rows[1],
            "block": "overall",
            "metric": "pr_auc",
            "num_pairs": "1",
            "total_pairs": "1",
            "num_both_success": "0",
            "num_teacher_success_ours_fail": "1",
            "num_ours_success_teacher_fail": "0",
            "num_both_fail": "0",
            "comparable_pairs": "0",
            "ours_wins_by_pr_auc": "0",
            "teacher_wins_by_pr_auc": "0",
            "ours_wins": "0",
            "teacher_wins": "0",
            "ties": "0",
            "not_comparable_pairs": "1",
            "mean_delta_pr_auc": "",
            "median_delta_pr_auc": "",
            "mean_delta_f1": "",
            "median_delta_f1": "",
            "ours_winrate": "",
            "teacher_winrate": "",
            "tie_rate": "",
        },
    ]
    assert "- metrics_long_rows: 2" in summary
    assert "- metrics_wide_paired_rows: 1" in summary
    assert "- non_success_rows: 1" in summary
    assert "not comparable 1" in summary
    assert "xgboost_not_installed" in summary


def test_paired_summary_includes_preflight_metadata_and_required_sections(tmp_path: Path) -> None:
    from mind.wavelet_course.paired_reporting import write_paired_reports

    preflight = {
        "cache_audit": {
            "accepted": True,
            "num_entries": 10,
            "subsets": {"popular": 4, "random": 3, "adversarial": 3},
        },
        "population_summary": {
            "num_primary_population": 10,
            "num_hard_hallucination": 4,
            "num_correct": 6,
            "split_source": "v1_wavelet_population",
        },
        "split_validation": {
            "valid": True,
            "counts": {
                "train": {"pos": 3, "neg": 3},
                "validation": {"pos": 1, "neg": 1},
                "test": {"pos": 1, "neg": 1},
            },
        },
        "paired_grid_audit": {
            "paired_grid_path": str(tmp_path / "audit" / "paired_grid.json"),
            "num_pair_rows": 2,
            "num_pair_ids": 1,
            "requested_blocks": ["A"],
        },
        "sample_grid_audit": {
            "sample_grid_path": str(tmp_path / "audit" / "selected_sample_grid.csv"),
            "num_rows": 10,
            "row_order_hash": "samplehash123",
        },
    }
    config = {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subsets": ["popular", "random", "adversarial"],
        "population_grid": {"source": "v1_wavelet_population"},
        "paired_wavelet_v2": {"description": "Paired Teacher/Ours extension."},
    }

    paths = write_paired_reports(
        _paired_metric_rows(),
        tmp_path,
        config=config,
        preflight=preflight,
        metrics_ledger_path=tmp_path / "reports" / "metrics_ledger.csv",
    )

    text = paths["summary"].read_text(encoding="utf-8")
    required_headings = [
        "Experiment Overview",
        "Task and Population",
        "Why Paired Comparison",
        "Method Definitions",
        "Wavelet Selection Rationale",
        "Paired Results",
        "Interpretation",
        "Limitations",
        "Conclusion",
    ]
    for heading in required_headings:
        assert f"## {heading}" in text

    assert "v2 paired extension" in text
    assert "v1 not overwritten" in text
    assert "Teacher:" in text
    assert "Ours:" in text
    assert "wavelet rationale" in text
    assert "paired best" in text
    assert "winrate" in text
    assert "failure counts" in text
    assert "limitations" in text
    assert "- extension: paired Teacher/Ours wavelet-course v2" in text
    assert "- v1_preservation: v1_wavelet_population" in text
    assert "- cache_entries: 10" in text
    assert "- primary_population: 10" in text
    assert "- hard_hallucinations: 4" in text
    assert "- correct: 6" in text
    assert "- split_train: pos=3 neg=3" in text
    assert "- split_validation: pos=1 neg=1" in text
    assert "- split_test: pos=1 neg=1" in text
    assert f"- paired_grid_path: {tmp_path / 'audit' / 'paired_grid.json'}" in text
    assert "- paired_grid_rows: 2" in text
    assert "- paired_grid_pair_ids: 1" in text
    assert f"- sample_grid_path: {tmp_path / 'audit' / 'selected_sample_grid.csv'}" in text
    assert "- sample_grid_rows: 10" in text
    assert "- sample_grid_row_order_hash: samplehash123" in text
    assert f"- metrics_ledger.csv: {tmp_path / 'reports' / 'metrics_ledger.csv'}" in text
