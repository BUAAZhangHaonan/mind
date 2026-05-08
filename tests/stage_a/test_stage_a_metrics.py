from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

from mind.trajectory.stage_a_metrics import binary_diagnostic_metrics


def test_pr_auc_uses_precision_recall_curve_auc_not_average_precision() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    expected_pr_auc = auc(recall, precision)

    metrics = binary_diagnostic_metrics(y_true, scores)

    assert metrics["pr_auc"] == expected_pr_auc
    assert metrics["average_precision"] == average_precision_score(y_true, scores)
    assert metrics["pr_auc"] != metrics["average_precision"]
