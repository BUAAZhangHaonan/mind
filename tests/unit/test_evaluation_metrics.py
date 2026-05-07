from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mind.evaluation import (
    canonicalize_result_frame,
    compute_binary_metrics,
    evaluate_by_subset,
    write_metrics_report,
    write_results_table,
)


def test_compute_binary_metrics_returns_expected_keys() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["tpr_at_fpr_0.01"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_evaluate_by_subset_computes_metrics_per_subset() -> None:
    frame = pd.DataFrame(
        [
            {"subset": "popular", "label": 0, "prediction": 0, "score": 0.1},
            {"subset": "popular", "label": 1, "prediction": 1, "score": 0.9},
            {"subset": "adversarial", "label": 0, "prediction": 0, "score": 0.2},
            {"subset": "adversarial", "label": 1, "prediction": 1, "score": 0.8},
        ]
    )

    metrics = evaluate_by_subset(frame)

    assert sorted(metrics) == ["adversarial", "popular"]
    assert metrics["popular"]["accuracy"] == 1.0


def test_write_metrics_report_writes_json_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    payload = {"popular": {"accuracy": 1.0}}

    write_metrics_report(payload, output_path)

    restored = json.loads(output_path.read_text(encoding="utf-8"))
    assert restored == payload


def test_canonicalize_result_frame_keeps_only_result_columns_and_synthesizes_sample_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": 3,
                "object_name": "dog",
                "subset": "popular",
                "label": 1,
                "prediction": 1,
                "score": 0.9,
                "fold": 0,
                "extra_feature_0": 0.3,
                "hidden_0": 4.2,
            }
        ]
    )

    canonical = canonicalize_result_frame(frame)

    assert canonical.columns.tolist() == [
        "sample_id",
        "image_id",
        "object_name",
        "subset",
        "label",
        "prediction",
        "score",
        "fold",
    ]
    assert canonical.loc[0, "sample_id"] == "row-000000"


def test_write_results_table_exports_only_canonical_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "results.csv"
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-2",
                "image_id": 12,
                "object_name": "dog",
                "subset": "popular",
                "label": 1,
                "prediction": 1,
                "score": 0.8,
                "fold": 1,
                "extra_feature_0": 3.1,
                "hidden_0": 5.7,
            }
        ]
    )

    write_results_table(frame, output_path)

    restored = pd.read_csv(output_path)
    assert restored.columns.tolist() == [
        "sample_id",
        "image_id",
        "object_name",
        "subset",
        "label",
        "prediction",
        "score",
        "fold",
    ]


def test_write_results_table_preserves_requested_extra_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "results.csv"
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-2",
                "image_id": 12,
                "object_name": "dog",
                "subset": "popular",
                "label": 1,
                "prediction": 1,
                "score": 0.8,
                "fold": 1,
                "selected_probe": "vision_only",
                "extra_feature_0": 3.1,
            }
        ]
    )

    write_results_table(frame, output_path, extra_columns=("selected_probe",))

    restored = pd.read_csv(output_path)
    assert restored.columns.tolist() == [
        "sample_id",
        "image_id",
        "object_name",
        "subset",
        "label",
        "prediction",
        "score",
        "fold",
        "selected_probe",
    ]
    assert restored.loc[0, "selected_probe"] == "vision_only"
