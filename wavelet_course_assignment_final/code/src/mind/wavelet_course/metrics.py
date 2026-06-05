"""Binary metrics for wavelet-course classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_NAMES = (
    "pr_auc",
    "average_precision",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "fpr_at_95pct_tpr",
)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    f1: float
    reason: str | None = None


def evaluate_validation_test(
    y_validation: list[int] | np.ndarray,
    validation_scores: list[float] | np.ndarray,
    y_test: list[int] | np.ndarray,
    test_scores: list[float] | np.ndarray,
) -> dict[str, Any]:
    selection = select_best_f1_threshold(y_validation, validation_scores)
    validation = binary_metrics(y_validation, validation_scores, threshold=selection.threshold)
    if selection.reason is not None:
        test = _undefined_metrics(selection.reason, num_samples=len(y_test), threshold=selection.threshold)
    else:
        test = binary_metrics(y_test, test_scores, threshold=selection.threshold)
    return {
        "threshold": selection.threshold,
        "threshold_selection_reason": selection.reason,
        "validation": validation,
        "test": test,
    }


def select_best_f1_threshold(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> ThresholdSelection:
    labels = _as_label_vector(y_true)
    score_vector = _as_score_vector(scores, expected_size=labels.shape[0])
    reason = _single_class_reason(labels)
    if reason is not None:
        return ThresholdSelection(threshold=float("nan"), f1=float("nan"), reason=reason)
    thresholds = np.unique(score_vector)
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = (score_vector >= threshold).astype(np.int64)
        value = float(f1_score(labels, predictions, zero_division=0))
        if value > best_f1 or (value == best_f1 and float(threshold) < best_threshold):
            best_threshold = float(threshold)
            best_f1 = value
    return ThresholdSelection(threshold=best_threshold, f1=best_f1)


def binary_metrics(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = _as_label_vector(y_true)
    score_vector = _as_score_vector(scores, expected_size=labels.shape[0])
    if not np.isfinite(float(threshold)):
        return _undefined_metrics("threshold is undefined", num_samples=labels.shape[0], threshold=threshold)
    reason = _single_class_reason(labels)
    if reason is not None:
        return _undefined_metrics(reason, num_samples=labels.shape[0], threshold=threshold)
    predictions = (score_vector >= float(threshold)).astype(np.int64)
    precision, recall, _ = precision_recall_curve(labels, score_vector)
    fpr, tpr, _ = roc_curve(labels, score_vector)
    return {
        "pr_auc": float(auc(recall[::-1], precision[::-1])),
        "average_precision": float(average_precision_score(labels, score_vector)),
        "roc_auc": float(roc_auc_score(labels, score_vector)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "tpr_at_1pct_fpr": _tpr_at_fpr(fpr, tpr, target_fpr=0.01),
        "fpr_at_95pct_tpr": _fpr_at_tpr(fpr, tpr, target_tpr=0.95),
        "threshold": float(threshold),
        "num_samples": int(labels.shape[0]),
        "undefined_reason": None,
    }


def _as_label_vector(values: list[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("y_true must be a 1D vector")
    if array.size == 0:
        raise ValueError("y_true must not be empty")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return array


def _as_score_vector(values: list[float] | np.ndarray, *, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("scores must be a 1D vector")
    if array.shape[0] != expected_size:
        raise ValueError(
            f"scores length {array.shape[0]} does not match labels length {expected_size}"
        )
    if not np.isfinite(array).all():
        raise ValueError("scores must be finite")
    return array


def _single_class_reason(labels: np.ndarray) -> str | None:
    unique = np.unique(labels)
    if unique.shape[0] == 1:
        return f"single class present: {int(unique[0])}"
    return None


def _undefined_metrics(reason: str, *, num_samples: int, threshold: float) -> dict[str, Any]:
    metrics = {name: float("nan") for name in METRIC_NAMES}
    metrics["threshold"] = float(threshold)
    metrics["num_samples"] = int(num_samples)
    metrics["undefined_reason"] = reason
    return metrics


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, *, target_fpr: float) -> float:
    eligible = tpr[fpr <= target_fpr]
    if eligible.size == 0:
        return float("nan")
    return float(np.max(eligible))


def _fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray, *, target_tpr: float) -> float:
    eligible = fpr[tpr >= target_tpr]
    if eligible.size == 0:
        return float("nan")
    return float(np.min(eligible))
