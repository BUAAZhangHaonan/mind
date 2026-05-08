from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

from mind.trajectory.stage_a_metrics import bootstrap_binary_metrics, binary_diagnostic_metrics


def test_pr_auc_uses_precision_recall_curve_auc_not_average_precision() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    expected_pr_auc = auc(recall, precision)

    metrics = binary_diagnostic_metrics(y_true, scores)

    assert metrics["pr_auc"] == expected_pr_auc
    assert metrics["average_precision"] == average_precision_score(y_true, scores)
    assert metrics["pr_auc"] != metrics["average_precision"]


def test_bootstrap_samples_only_pr_auc_and_roc_auc() -> None:
    y_true = np.asarray([0, 1] * 10, dtype=np.int64)
    scores = np.asarray(
        [
            0.10,
            0.86,
            0.21,
            0.82,
            0.34,
            0.76,
            0.45,
            0.67,
            0.36,
            0.73,
            0.29,
            0.71,
            0.48,
            0.64,
            0.42,
            0.79,
            0.26,
            0.88,
            0.31,
            0.69,
        ],
        dtype=np.float32,
    )
    metric_fn_calls = 0

    def metric_fn(
        labels: np.ndarray,
        score_vector: np.ndarray,
        *,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        nonlocal metric_fn_calls
        metric_fn_calls += 1
        return binary_diagnostic_metrics(labels, score_vector, threshold=threshold)

    intervals = bootstrap_binary_metrics(
        y_true,
        scores,
        num_bootstrap=8,
        seed=20260506,
        metric_fn=metric_fn,
    )

    assert set(intervals) == set(binary_diagnostic_metrics(y_true, scores))
    assert metric_fn_calls == 1
    assert intervals["pr_auc"].num_bootstrap == 8
    assert intervals["roc_auc"].num_bootstrap == 8
    assert intervals["accuracy"].num_bootstrap == 0
    assert np.isfinite(intervals["pr_auc"].lower)
    assert np.isfinite(intervals["pr_auc"].upper)
    assert np.isfinite(intervals["roc_auc"].lower)
    assert np.isfinite(intervals["roc_auc"].upper)
