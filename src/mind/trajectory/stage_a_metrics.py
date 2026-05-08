"""Stage A binary diagnostic metrics and bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_BOOTSTRAP_SEED = 20260506
BOOTSTRAP_CI_METRICS = ("pr_auc", "roc_auc")


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile bootstrap interval for one metric."""

    value: float
    mean: float
    lower: float
    upper: float
    num_bootstrap: int


def binary_diagnostic_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Stage A binary metrics from anomaly scores.

    Scores are interpreted as higher-is-more-hallucinatory. Ranking metrics are
    reported as NaN when the input has only one class.
    """

    labels = _as_label_vector(y_true)
    score_vector = _as_score_vector(scores, expected_size=labels.shape[0])
    predictions = (score_vector >= float(threshold)).astype(np.int64)
    has_two_classes = np.unique(labels).shape[0] > 1

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": _balanced_accuracy(labels, predictions)
        if has_two_classes
        else float("nan"),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, score_vector)) if has_two_classes else float("nan"),
        "pr_auc": _pr_curve_auc(labels, score_vector) if has_two_classes else float("nan"),
        "average_precision": float(average_precision_score(labels, score_vector))
        if has_two_classes
        else float("nan"),
        "tpr_at_1pct_fpr": _tpr_at_fpr(labels, score_vector, target_fpr=0.01)
        if has_two_classes
        else float("nan"),
        "fpr_at_95pct_tpr": _fpr_at_tpr(labels, score_vector, target_tpr=0.95)
        if has_two_classes
        else float("nan"),
        "threshold": float(threshold),
        "num_samples": float(labels.shape[0]),
    }


def bootstrap_binary_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    metric_fn: Callable[..., Mapping[str, float]] = binary_diagnostic_metrics,
) -> dict[str, BootstrapInterval]:
    """Compute deterministic percentile bootstrap intervals for Stage A CI metrics."""

    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    labels = _as_label_vector(y_true)
    score_vector = _as_score_vector(scores, expected_size=labels.shape[0])
    point_metrics = dict(metric_fn(labels, score_vector, threshold=threshold))

    rng = np.random.default_rng(seed)
    ci_metric_names = tuple(name for name in BOOTSTRAP_CI_METRICS if name in point_metrics)
    bootstrap_values: dict[str, list[float]] = {name: [] for name in ci_metric_names}
    for _ in range(num_bootstrap):
        indices = rng.integers(0, labels.shape[0], size=labels.shape[0])
        sampled = _bootstrap_ci_metrics(labels[indices], score_vector[indices])
        for name, value in sampled.items():
            if name in bootstrap_values:
                bootstrap_values[name].append(float(value))

    alpha = (1.0 - confidence) / 2.0
    lower_q = 100.0 * alpha
    upper_q = 100.0 * (1.0 - alpha)
    intervals: dict[str, BootstrapInterval] = {}
    for name, point_value in point_metrics.items():
        values = np.asarray(bootstrap_values.get(name, []), dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            intervals[name] = BootstrapInterval(
                value=float(point_value),
                mean=float("nan"),
                lower=float("nan"),
                upper=float("nan"),
                num_bootstrap=0,
            )
            continue
        intervals[name] = BootstrapInterval(
            value=float(point_value),
            mean=float(np.mean(finite_values)),
            lower=float(np.percentile(finite_values, lower_q)),
            upper=float(np.percentile(finite_values, upper_q)),
            num_bootstrap=int(finite_values.size),
        )
    return intervals


def bootstrap_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    **kwargs: Any,
) -> dict[str, BootstrapInterval]:
    """Alias used by Stage A callers that ask for bootstrap metrics."""

    return bootstrap_binary_metrics(y_true, scores, **kwargs)


def _as_label_vector(values: np.ndarray | list[int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("y_true must be a 1D vector")
    if array.size == 0:
        raise ValueError("y_true must not be empty")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return array


def _as_score_vector(values: np.ndarray | list[float], *, expected_size: int) -> np.ndarray:
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


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive_rate = np.mean(predictions[labels == 1] == 1)
    true_negative_rate = np.mean(predictions[labels == 0] == 0)
    return float((true_positive_rate + true_negative_rate) / 2.0)


def _pr_curve_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def _bootstrap_ci_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(labels).shape[0] < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {
        "pr_auc": _pr_curve_auc(labels, scores),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, *, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= target_fpr]
    if eligible.size == 0:
        return 0.0
    return float(np.max(eligible))


def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, *, target_tpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = fpr[tpr >= target_tpr]
    if eligible.size == 0:
        return 1.0
    return float(np.min(eligible))
