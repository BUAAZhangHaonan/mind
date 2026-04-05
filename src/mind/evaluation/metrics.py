"""Evaluation metrics for hallucination detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RESULT_COLUMNS = (
    "sample_id",
    "image_id",
    "object_name",
    "subset",
    "label",
    "prediction",
    "score",
    "fold",
)


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
    unique_labels = set(y_true)

    negative_total = sum(1 for value in y_true if value == 0)
    false_positives = sum(
        1
        for truth, prediction in zip(y_true, y_pred)
        if truth == 0 and prediction == 1
    )

    roc_auc = float(roc_auc_score(y_true, y_score)) if len(unique_labels) > 1 else float("nan")
    if len(unique_labels) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(recall[::-1], precision[::-1]))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        tpr_at_fpr = float(np.max(tpr[fpr <= 0.01])) if np.any(fpr <= 0.01) else 0.0
    else:
        pr_auc = float("nan")
        tpr_at_fpr = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tpr_at_fpr_0.01": tpr_at_fpr,
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


def canonicalize_result_frame(
    frame: pd.DataFrame,
    *,
    extra_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    results = frame.copy()
    if "sample_id" not in results.columns:
        results["sample_id"] = [f"row-{index:06d}" for index in range(len(results))]
    results["sample_id"] = results["sample_id"].astype(str)

    defaults = {
        "image_id": -1,
        "object_name": "",
        "subset": "",
        "label": 0,
        "prediction": 0,
        "score": 0.0,
        "fold": 0,
    }
    for column, default in defaults.items():
        if column not in results.columns:
            results[column] = default

    ordered_columns = list(RESULT_COLUMNS)
    for column in extra_columns:
        if column not in results.columns:
            results[column] = ""
        ordered_columns.append(column)

    results = results.loc[:, ordered_columns].sort_values("sample_id").reset_index(drop=True)
    return results


def write_results_table(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    extra_columns: tuple[str, ...] | list[str] = (),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonicalize_result_frame(frame, extra_columns=extra_columns).to_csv(path, index=False)
