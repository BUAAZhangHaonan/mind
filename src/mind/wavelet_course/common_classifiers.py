"""Common static readouts for paired-wavelet v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import inspect
import itertools
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

STATIC_CLASSIFIERS = ("logreg", "linear_svm", "rf", "extra_trees", "xgboost")
XGBOOST_ALIASES = ("xgboost", "xgb")
XGBOOST_NOT_INSTALLED = "xgboost_not_installed"


@dataclass(frozen=True)
class SplitScores:
    train: np.ndarray
    validation: np.ndarray | None
    test: np.ndarray | None


@dataclass(frozen=True)
class StaticTrainingResult:
    classifier: str
    status: str
    model: Any | None
    scores: SplitScores | None
    training_curve: list[dict[str, float]] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_validation_pr_auc: float | None = None
    failure_reason: str = ""


class XGBoostNotInstalledError(ImportError):
    """Raised when xgboost is requested but unavailable."""


def train_static_classifier(
    classifier: str,
    train_x: Any,
    train_y: Any,
    *,
    validation_x: Any | None = None,
    validation_y: Any | None = None,
    test_x: Any | None = None,
    random_state: int = 0,
    max_iter: int = 20000,
    n_estimators: int = 1000,
    n_jobs: int = 1,
    allow_missing_xgboost: bool = False,
    model_params: Mapping[str, Any] | None = None,
) -> StaticTrainingResult:
    """Train one static binary readout and return continuous split scores."""

    name = _normalize_classifier_name(classifier)
    x_train = _as_2d_float_array(train_x, name="train_x")
    y_train = _as_labels(train_y, expected_size=x_train.shape[0], name="train_y")
    _require_two_classes(y_train, name="train_y")
    x_val, y_val = _optional_validation(validation_x, validation_y, feature_dim=x_train.shape[1])
    x_test = _optional_2d(test_x, expected_dim=x_train.shape[1], name="test_x")

    try:
        candidate_params = _candidate_model_params(
            name,
            y_train,
            random_state=int(random_state),
            max_iter=int(max_iter),
            n_estimators=int(n_estimators),
            n_jobs=int(n_jobs),
            model_params=dict(model_params or {}),
        )
    except XGBoostNotInstalledError:
        if not allow_missing_xgboost:
            raise
        return StaticTrainingResult(
            classifier=name,
            status="failure",
            model=None,
            scores=None,
            failure_reason=XGBOOST_NOT_INSTALLED,
        )

    if len(candidate_params) > 1 and (x_val is None or y_val is None):
        raise ValueError(f"{name} hyperparameter grid requires validation_x and validation_y")
    model: Any | None = None
    best_params: dict[str, Any] = {}
    best_validation_pr_auc: float | None = None
    training_curve: list[dict[str, float]] = []
    for index, params in enumerate(candidate_params):
        candidate = _instantiate_static_model(name, params)
        _fit_static_model(candidate, x_train, y_train, x_val=x_val, y_val=y_val)
        if x_val is None or y_val is None:
            validation_pr_auc = None
            is_better = model is None
        else:
            validation_scores = _predict_continuous_scores(candidate, x_val)
            validation_pr_auc = _validation_pr_auc(y_val, validation_scores)
            is_better = (
                best_validation_pr_auc is None
                or validation_pr_auc > best_validation_pr_auc
            )
        training_curve.append(
            {
                "candidate_index": float(index),
                "validation_pr_auc": float(validation_pr_auc) if validation_pr_auc is not None else float("nan"),
            }
        )
        if is_better:
            model = candidate
            best_params = dict(params)
            best_validation_pr_auc = validation_pr_auc
    if model is None:
        raise RuntimeError(f"{name} did not produce a trained candidate model")
    scores = SplitScores(
        train=_predict_continuous_scores(model, x_train),
        validation=_predict_continuous_scores(model, x_val) if x_val is not None else None,
        test=_predict_continuous_scores(model, x_test) if x_test is not None else None,
    )
    return StaticTrainingResult(
        classifier=name,
        status="success",
        model=model,
        scores=scores,
        training_curve=training_curve,
        best_params=_jsonable_params(best_params),
        best_validation_pr_auc=best_validation_pr_auc,
    )


def xgboost_missing_failure_rows(
    paired_configs: Sequence[Mapping[str, Any]],
    *,
    failure_reason: str = XGBOOST_NOT_INSTALLED,
) -> list[dict[str, Any]]:
    """Return one paired failure row per config when xgboost is unavailable."""

    rows: list[dict[str, Any]] = []
    for config in paired_configs:
        row = dict(config)
        row["status"] = "failure"
        row["failure_reason"] = str(failure_reason)
        rows.append(row)
    return rows


def _normalize_classifier_name(classifier: str) -> str:
    name = str(classifier).strip().lower()
    if name in XGBOOST_ALIASES:
        return "xgboost"
    if name not in STATIC_CLASSIFIERS:
        raise ValueError(f"classifier must be one of {STATIC_CLASSIFIERS} or 'xgb', got {classifier!r}")
    return name


def _candidate_model_params(
    classifier: str,
    y_train: np.ndarray,
    *,
    random_state: int,
    max_iter: int,
    n_estimators: int,
    n_jobs: int,
    model_params: dict[str, Any],
) -> Any:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")

    if classifier == "logreg":
        params: dict[str, Any] = {
            "class_weight": "balanced",
            "max_iter": max_iter,
            "random_state": random_state,
            "solver": "lbfgs",
        }
        params.update(model_params)
        return _expand_param_grid(params)
    if classifier == "linear_svm":
        params = {
            "class_weight": "balanced",
            "dual": "auto",
            "max_iter": max_iter,
            "random_state": random_state,
        }
        params.update(model_params)
        return _expand_param_grid(params)
    if classifier == "rf":
        params = {
            "class_weight": "balanced_subsample",
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        params.update(model_params)
        return _expand_param_grid(params)
    if classifier == "extra_trees":
        params = {
            "class_weight": "balanced",
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        params.update(model_params)
        return _expand_param_grid(params)
    if classifier == "xgboost":
        _load_xgboost_classifier()
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        params = {
            "colsample_bytree": 0.9,
            "eval_metric": "aucpr",
            "learning_rate": 0.05,
            "max_depth": 3,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "objective": "binary:logistic",
            "random_state": random_state,
            "scale_pos_weight": negatives / positives,
            "subsample": 0.9,
        }
        params.update(model_params)
        return _expand_param_grid(params)
    raise AssertionError(f"unhandled classifier: {classifier}")


def _instantiate_static_model(classifier: str, params: Mapping[str, Any]) -> Any:
    concrete_params = dict(params)
    if classifier == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**concrete_params)),
            ]
        )
    if classifier == "linear_svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(**concrete_params)),
            ]
        )
    if classifier == "rf":
        return RandomForestClassifier(**concrete_params)
    if classifier == "extra_trees":
        return ExtraTreesClassifier(**concrete_params)
    if classifier == "xgboost":
        return _load_xgboost_classifier()(**concrete_params)
    raise AssertionError(f"unhandled classifier: {classifier}")


def _expand_param_grid(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid_keys = [
        key
        for key, value in params.items()
        if isinstance(value, list)
    ]
    if not grid_keys:
        return [dict(params)]
    candidates: list[dict[str, Any]] = []
    for values in itertools.product(*(params[key] for key in grid_keys)):
        candidate = dict(params)
        for key, value in zip(grid_keys, values, strict=True):
            candidate[key] = value
        candidates.append(candidate)
    return candidates


def _load_xgboost_classifier() -> Any:
    sentinel = object()
    if sys.modules.get("xgboost", sentinel) is None:
        raise XGBoostNotInstalledError("xgboost is not installed")
    try:
        module = importlib.import_module("xgboost")
    except ImportError as exc:
        raise XGBoostNotInstalledError("xgboost is not installed") from exc
    try:
        return module.XGBClassifier
    except AttributeError as exc:
        raise XGBoostNotInstalledError("xgboost.XGBClassifier is unavailable") from exc


def _fit_static_model(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> None:
    kwargs: dict[str, Any] = {}
    try:
        fit_parameters = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        fit_parameters = {}
    if x_val is not None and y_val is not None and "eval_set" in fit_parameters:
        kwargs["eval_set"] = [(x_val, y_val)]
        if "verbose" in fit_parameters:
            kwargs["verbose"] = False
    model.fit(train_x, train_y, **kwargs)


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty sample and feature dimensions")
    _raise_if_non_finite(array, name=name)
    return array


def _optional_2d(values: Any | None, *, expected_dim: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = _as_2d_float_array(values, name=name)
    if array.shape[1] != expected_dim:
        raise ValueError(f"{name} feature dimension must match train_x")
    return array


def _optional_validation(
    validation_x: Any | None,
    validation_y: Any | None,
    *,
    feature_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if validation_x is None and validation_y is None:
        return None, None
    if validation_x is None or validation_y is None:
        raise ValueError("validation_x and validation_y must be provided together")
    x_val = _optional_2d(validation_x, expected_dim=feature_dim, name="validation_x")
    if x_val is None:
        raise ValueError("validation_x is required")
    y_val = _as_labels(validation_y, expected_size=x_val.shape[0], name="validation_y")
    return x_val, y_val


def _as_labels(values: Any, *, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if array.shape[0] != expected_size:
        raise ValueError(f"{name} length must match sample count")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 labels")
    return array


def _require_two_classes(labels: np.ndarray, *, name: str) -> None:
    if np.unique(labels).shape[0] < 2:
        raise ValueError(f"{name} must contain at least two classes")


def _predict_continuous_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=np.float32)
    else:
        raise TypeError("model must expose predict_proba or decision_function")
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if scores.shape[0] != x.shape[0]:
        raise ValueError("score count must match sample count")
    _raise_if_non_finite(scores, name="scores")
    return scores


def _validation_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).shape[0] < 2:
        raise ValueError("validation_y must contain at least two classes for PR-AUC selection")
    return float(average_precision_score(labels, scores))


def _jsonable_params(params: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            output[str(key)] = value.item()
        else:
            output[str(key)] = value
    return output


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


__all__ = [
    "STATIC_CLASSIFIERS",
    "XGBOOST_NOT_INSTALLED",
    "SplitScores",
    "StaticTrainingResult",
    "XGBoostNotInstalledError",
    "train_static_classifier",
    "xgboost_missing_failure_rows",
]
