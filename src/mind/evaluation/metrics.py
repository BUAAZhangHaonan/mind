"""Evaluation metrics for hallucination detection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_object_hallucination_label(
    *,
    ground_truth_label: int,
    answer_label: int | None,
) -> int:
    return int(answer_label == 1 and int(ground_truth_label) == 0)


def compute_binary_metrics(
    *,
    y_true,
    y_pred,
    y_score,
) -> dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    y_score = list(y_score)

    negative_total = sum(1 for value in y_true if value == 0)
    false_positives = sum(
        1
        for truth, prediction in zip(y_true, y_pred)
        if truth == 0 and prediction == 1
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "false_positive_rate": 0.0
        if negative_total == 0
        else float(false_positives / negative_total),
    }


def evaluate_by_subset(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for subset, subset_frame in frame.groupby("subset"):
        metrics[str(subset)] = compute_binary_metrics(
            y_true=subset_frame["label"],
            y_pred=subset_frame["prediction"],
            y_score=subset_frame["score"],
        )
    return metrics


def write_metrics_report(payload: dict[str, dict[str, float]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_results_table(frame: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
