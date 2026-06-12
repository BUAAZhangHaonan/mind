"""Stage C support estimators, candidate selection, and score helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.special import betainc

from .stage_a_metrics import binary_diagnostic_metrics
from .stage_b_knn import ALLOWED_STAGE_B_K_VALUES, generate_stage_b_k_candidates
from .stage_b4_vmf import (
    fit_mixture_vmf_support,
    fit_single_vmf_support,
    score_mixture_vmf_support,
    score_single_vmf_support,
)
from .stage_c_manifest import REQUIRED_STAGE_C_RATIO, STAGE_C_OBJECTIVE


STAGE_C_METHODS = ("single_vmf", "mixture_vmf", "radius_ball", "knn", "logistic")
STAGE_C_SUPPORT_METHODS = ("single_vmf", "mixture_vmf", "radius_ball", "knn")
STAGE_C_COMPARATOR_METHODS = ("logistic",)
STAGE_C_MIXTURE_K_VALUES = (2, 4, 8)
STAGE_C_KNN_K_VALUES = ALLOWED_STAGE_B_K_VALUES
STAGE_C_RADIUS_QUANTILES = (0.50, 0.65, 0.80, 0.90, 0.95)
STAGE_C_LOGISTIC_C_VALUES = (0.1, 1.0, 10.0)
STAGE_C_SELECTION_SPLIT = "repope/cal"


def stage_c_method_contract() -> dict[str, object]:
    """Return the frozen Stage C method and role contract."""

    return {
        "methods": list(STAGE_C_METHODS),
        "support_methods": list(STAGE_C_SUPPORT_METHODS),
        "comparator_methods": list(STAGE_C_COMPARATOR_METHODS),
        "method_roles": {
            "single_vmf": "support_estimator",
            "mixture_vmf": "support_estimator",
            "radius_ball": "support_estimator",
            "knn": "support_estimator",
            "logistic": "supervised_comparator",
        },
    }


def stage_c_hyperparameter_grids() -> dict[str, list[float] | list[int]]:
    """Return the frozen Stage C candidate grids."""

    return {
        "single_vmf": [],
        "mixture_vmf_K": list(STAGE_C_MIXTURE_K_VALUES),
        "knn_k": list(STAGE_C_KNN_K_VALUES),
        "radius_ball_quantiles": list(STAGE_C_RADIUS_QUANTILES),
        "logistic_C": list(STAGE_C_LOGISTIC_C_VALUES),
    }


def select_stage_c_candidate(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    parameter_name: str,
    allowed_values: Sequence[int | float],
) -> dict[str, object]:
    """Select a Stage C hyperparameter on RePOPE calibration rows only."""

    candidates: list[dict[str, object]] = []
    allowed = {float(value) for value in allowed_values}
    for row in metric_rows:
        if str(row.get("dataset_family", "")).lower() != "repope":
            continue
        if str(row.get("metric_split") or row.get("split") or row.get("eval_split")) != "cal":
            continue
        if str(row.get("metric_status", "passed")) != "passed":
            continue
        if str(row.get("method", row.get("support_family", ""))) != str(method):
            continue
        value = _parameter_value(row, parameter_name)
        if value is None:
            continue
        if float(value) not in allowed:
            raise ValueError(f"Stage C {method} candidate {value} is outside the allowed grid")
        candidate = dict(row)
        candidate["parameter_value"] = value
        candidate["pr_auc"] = float(candidate["pr_auc"])
        candidate["roc_auc"] = float(candidate["roc_auc"])
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"no RePOPE cal rows were available for Stage C {method} selection")
    candidates.sort(key=lambda row: (-float(row["pr_auc"]), -float(row["roc_auc"]), float(row["parameter_value"])))
    selected = candidates[0]
    value = selected["parameter_value"]
    output = {
        "row_type": "selected",
        "model_alias": str(selected.get("model_alias") or selected.get("model_name") or ""),
        "objective": STAGE_C_OBJECTIVE,
        "method": str(method),
        "selected_on": STAGE_C_SELECTION_SPLIT,
        "frozen_for_test": True,
        "selection_metric": "pr_auc",
        "selection_pr_auc": float(selected["pr_auc"]),
        "selection_roc_auc": float(selected["roc_auc"]),
        "negative_budget_ratio": float(selected.get("negative_budget_ratio", REQUIRED_STAGE_C_RATIO)),
        "negative_budget_seed": int(float(selected.get("negative_budget_seed", selected.get("seed", 0)))),
        "metric_status": "passed",
    }
    _set_selected_parameter(output, parameter_name, value)
    return output


def build_stage_c_vmf_grid(
    *,
    model_alias: str,
    dataset_family: str,
    labels: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    seed: int,
    method: str,
    metric_split: str,
    ratio: float = REQUIRED_STAGE_C_RATIO,
) -> list[dict[str, object]]:
    """Evaluate single-vMF or mixture-vMF candidates for one family/split pair."""

    label_array, split_array, embedding_array = _aligned_arrays(labels, splits, embeddings)
    bank_mask = (split_array == "bank") & (label_array == 0)
    eval_mask = split_array == str(metric_split)
    y = label_array[eval_mask]
    num_bank_correct = int(bank_mask.sum())
    candidates = (None,) if method == "single_vmf" else STAGE_C_MIXTURE_K_VALUES
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        base = _candidate_base(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method=method,
            metric_split=metric_split,
            seed=seed,
            ratio=ratio,
            num_bank_correct=num_bank_correct,
            num_eval=int(y.size),
        )
        if method == "mixture_vmf":
            base["K"] = int(candidate)
            base["parameter_value"] = int(candidate)
            cap = int(math.floor(num_bank_correct / 20.0))
            base["valid_bank_correct_cap"] = cap
            if int(candidate) > cap:
                rows.append({**base, "metric_status": "skipped", "failure_reason": "insufficient_bank_correct", **_undefined_metrics()})
                continue
        if num_bank_correct <= 0:
            rows.append({**base, "metric_status": "skipped", "failure_reason": "insufficient_bank_correct", **_undefined_metrics()})
            continue
        if method == "single_vmf":
            model = fit_single_vmf_support(embedding_array[bank_mask])
            scores = score_single_vmf_support(model, embedding_array[eval_mask])
        else:
            model = fit_mixture_vmf_support(embedding_array[bank_mask], k=int(candidate), seed=int(seed))
            scores = score_mixture_vmf_support(model, embedding_array[eval_mask])
        rows.append(_candidate_metric_row(base, y, scores))
    return rows


def build_stage_c_knn_grid(
    *,
    model_alias: str,
    dataset_family: str,
    labels: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    seed: int,
    metric_split: str,
    ratio: float = REQUIRED_STAGE_C_RATIO,
) -> list[dict[str, object]]:
    """Evaluate density-style geodesic kNN candidates for one family/split pair."""

    label_array, split_array, embedding_array = _aligned_arrays(labels, splits, embeddings)
    bank_mask = (split_array == "bank") & (label_array == 0)
    eval_mask = split_array == str(metric_split)
    y = label_array[eval_mask]
    num_bank_correct = int(bank_mask.sum())
    candidate_values = generate_stage_b_k_candidates(num_bank_correct=num_bank_correct)
    rows: list[dict[str, object]] = []
    for k_value in candidate_values:
        base = _candidate_base(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="knn",
            metric_split=metric_split,
            seed=seed,
            ratio=ratio,
            num_bank_correct=num_bank_correct,
            num_eval=int(y.size),
        )
        base["k"] = int(k_value)
        base["parameter_value"] = int(k_value)
        scores = score_knn_density_support(
            bank_embeddings=embedding_array[bank_mask],
            query_embeddings=embedding_array[eval_mask],
            k=int(k_value),
        )
        rows.append(_candidate_metric_row(base, y, scores))
    return rows


def build_stage_c_radius_grid(
    *,
    model_alias: str,
    dataset_family: str,
    labels: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    seed: int,
    metric_split: str,
    ratio: float = REQUIRED_STAGE_C_RATIO,
) -> list[dict[str, object]]:
    """Evaluate radius-ball candidates derived from RePOPE calibration support thickness."""

    label_array, split_array, embedding_array = _aligned_arrays(labels, splits, embeddings)
    bank_mask = (split_array == "bank") & (label_array == 0)
    cal_correct_mask = (split_array == "cal") & (label_array == 0)
    eval_mask = split_array == str(metric_split)
    y = label_array[eval_mask]
    radius_rows = build_stage_c_radius_candidates(
        bank_embeddings=embedding_array[bank_mask],
        cal_embeddings=embedding_array[cal_correct_mask],
        cal_labels=np.zeros(int(cal_correct_mask.sum()), dtype=np.int64),
    )
    rows: list[dict[str, object]] = []
    for radius_row in radius_rows:
        base = _candidate_base(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="radius_ball",
            metric_split=metric_split,
            seed=seed,
            ratio=ratio,
            num_bank_correct=int(bank_mask.sum()),
            num_eval=int(y.size),
        )
        base.update(radius_row)
        base["parameter_value"] = float(radius_row["rho"])
        scores = score_radius_ball_support(
            bank_embeddings=embedding_array[bank_mask],
            query_embeddings=embedding_array[eval_mask],
            rho=float(radius_row["rho"]),
        )
        rows.append(_candidate_metric_row(base, y, scores))
    return rows


def build_stage_c_radius_candidates(
    *,
    bank_embeddings: np.ndarray | Sequence[Sequence[float]],
    cal_embeddings: np.ndarray | Sequence[Sequence[float]],
    cal_labels: Sequence[int] | np.ndarray,
) -> list[dict[str, object]]:
    """Derive radius-ball candidates from calibration correct support radii."""

    bank = _row_normalize(_as_2d_array(bank_embeddings, name="bank_embeddings"))
    cal = _row_normalize(_as_2d_array(cal_embeddings, name="cal_embeddings"))
    labels = np.asarray(cal_labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != cal.shape[0]:
        raise ValueError("cal_labels and cal_embeddings must have the same row count")
    correct = cal[labels == 0]
    if correct.shape[0] == 0:
        raise ValueError("radius candidates require calibration correct samples")
    k0 = min(4, int(math.floor(math.sqrt(bank.shape[0]))))
    if k0 < 1:
        raise ValueError("radius candidates require a non-empty correct bank")
    distances = _geodesic_distance_matrix(correct, bank)
    kth = np.sort(distances, axis=1)[:, min(k0 - 1, distances.shape[1] - 1)]
    rows: list[dict[str, object]] = []
    for quantile in STAGE_C_RADIUS_QUANTILES:
        rho = float(np.quantile(kth, quantile))
        rows.append(
            {
                "rho": max(rho, 1e-6),
                "quantile": float(quantile),
                "selection_split": STAGE_C_SELECTION_SPLIT,
                "source": "calibration_correct_support_radii",
                "k0": int(k0),
            }
        )
    return rows


def score_knn_density_support(
    *,
    bank_embeddings: np.ndarray | Sequence[Sequence[float]],
    query_embeddings: np.ndarray | Sequence[Sequence[float]],
    k: int,
) -> np.ndarray:
    """Return anomaly scores from a spherical kNN density estimate."""

    bank = _row_normalize(_as_2d_array(bank_embeddings, name="bank_embeddings"))
    query = _row_normalize(_as_2d_array(query_embeddings, name="query_embeddings"))
    k_value = min(max(int(k), 1), bank.shape[0])
    distances = _geodesic_distance_matrix(query, bank)
    radius = np.sort(distances, axis=1)[:, k_value - 1]
    return _log_spherical_cap_proxy(radius, bank.shape[1]).astype(np.float32)


def score_radius_ball_support(
    *,
    bank_embeddings: np.ndarray | Sequence[Sequence[float]],
    query_embeddings: np.ndarray | Sequence[Sequence[float]],
    rho: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return anomaly scores from a fixed-radius spherical support estimate."""

    bank = _row_normalize(_as_2d_array(bank_embeddings, name="bank_embeddings"))
    query = _row_normalize(_as_2d_array(query_embeddings, name="query_embeddings"))
    radius = max(float(rho), 1e-6)
    distances = _geodesic_distance_matrix(query, bank)
    counts = np.sum(distances <= radius, axis=1).astype(np.float64)
    constant = _log_spherical_cap_proxy(np.asarray([radius], dtype=np.float64), bank.shape[1])[0]
    return (constant - np.log(counts + float(eps))).astype(np.float32)


def _candidate_metric_row(base: Mapping[str, object], y: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    undefined_reason = ""
    if y.size == 0:
        undefined_reason = f"no samples in {base['metric_split']} split"
    elif np.unique(y).size < 2:
        undefined_reason = f"one class present in {base['metric_split']} split"
    elif not np.isfinite(scores).all():
        undefined_reason = "non-finite scores"
    metrics = _undefined_metrics() if undefined_reason else binary_diagnostic_metrics(y, scores)
    return {
        **base,
        "metric_status": "undefined" if undefined_reason else "passed",
        "failure_reason": undefined_reason,
        **metrics,
    }


def _candidate_base(
    *,
    model_alias: str,
    dataset_family: str,
    method: str,
    metric_split: str,
    seed: int,
    ratio: float,
    num_bank_correct: int,
    num_eval: int,
) -> dict[str, object]:
    return {
        "row_type": "support_candidate",
        "model_alias": str(model_alias),
        "model_name": str(model_alias),
        "objective": STAGE_C_OBJECTIVE,
        "dataset_family": str(dataset_family),
        "method": str(method),
        "support_family": str(method),
        "readout": f"StageC-{method}",
        "split": str(metric_split),
        "eval_split": str(metric_split),
        "metric_split": str(metric_split),
        "eval_scope": "pooled",
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
        "num_bank_correct": int(num_bank_correct),
        "bank_size": int(num_bank_correct),
        "num_eval": int(num_eval),
    }


def _parameter_value(row: Mapping[str, object], parameter_name: str) -> int | float | None:
    if row.get(parameter_name) not in (None, ""):
        value = row[parameter_name]
    elif row.get(f"selected_{parameter_name}") not in (None, ""):
        value = row[f"selected_{parameter_name}"]
    elif row.get("parameter_value") not in (None, ""):
        value = row["parameter_value"]
    else:
        return None
    value_float = float(value)
    if parameter_name in {"k", "K"}:
        return int(value_float)
    return value_float


def _set_selected_parameter(row: dict[str, object], parameter_name: str, value: int | float) -> None:
    row["parameter_name"] = parameter_name
    row["parameter_value"] = value
    if parameter_name == "K":
        row["selected_K"] = int(value)
        row["K"] = int(value)
    elif parameter_name == "k":
        row["selected_k"] = int(value)
        row["k"] = int(value)
    elif parameter_name == "rho":
        row["selected_rho"] = float(value)
        row["rho"] = float(value)
    elif parameter_name == "C":
        row["selected_C"] = float(value)
        row["C"] = float(value)


def _aligned_arrays(
    labels: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    split_array = np.asarray(splits).reshape(-1)
    embedding_array = _as_2d_array(embeddings, name="embeddings").astype(np.float32, copy=False)
    if label_array.shape[0] != embedding_array.shape[0] or split_array.shape[0] != embedding_array.shape[0]:
        raise ValueError("labels, splits, and embeddings must have the same row count")
    return label_array, split_array, embedding_array


def _geodesic_distance_matrix(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    cosine = np.clip(query @ bank.T, -1.0, 1.0)
    return np.arccos(cosine)


def _log_spherical_cap_proxy(radius: np.ndarray, dim: int) -> np.ndarray:
    clipped = np.clip(radius, 1e-6, math.pi - 1e-6)
    sphere_dim = max(int(dim) - 1, 1)
    a = float(sphere_dim) / 2.0
    x = np.sin(clipped) ** 2
    regularized = betainc(a, 0.5, x)
    fraction = np.where(clipped <= (math.pi / 2.0), 0.5 * regularized, 1.0 - 0.5 * regularized)
    return np.log(np.clip(fraction, 1e-300, 1.0)).astype(np.float64)


def _as_2d_array(value: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _row_normalize(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("embedding rows must have positive finite norm")
    return value / norms


def _undefined_metrics() -> dict[str, float]:
    return {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
        "fpr_at_95pct_tpr": float("nan"),
    }


__all__ = [
    "STAGE_C_COMPARATOR_METHODS",
    "STAGE_C_KNN_K_VALUES",
    "STAGE_C_LOGISTIC_C_VALUES",
    "STAGE_C_METHODS",
    "STAGE_C_MIXTURE_K_VALUES",
    "STAGE_C_RADIUS_QUANTILES",
    "STAGE_C_SUPPORT_METHODS",
    "build_stage_c_knn_grid",
    "build_stage_c_radius_candidates",
    "build_stage_c_radius_grid",
    "build_stage_c_vmf_grid",
    "score_knn_density_support",
    "score_radius_ball_support",
    "select_stage_c_candidate",
    "stage_c_hyperparameter_grids",
    "stage_c_method_contract",
]
