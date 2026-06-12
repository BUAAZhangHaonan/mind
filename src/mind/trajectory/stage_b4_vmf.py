"""Stage B4 vMF support-family fitting, scoring, selection, and stability."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .stage_a_metrics import binary_diagnostic_metrics
from .stage_b4_manifest import (
    REQUIRED_STAGE_B4_RATIO,
    REQUIRED_STAGE_B4_SEEDS,
    STAGE_B4_OBJECTIVE,
)


ALLOWED_STAGE_B4_VMF_K_VALUES = (1, 2, 4, 8)
STAGE_B4_VMF_STABILITY_DROP_TOLERANCE = 0.02


def fit_single_vmf_support(embeddings: np.ndarray | Sequence[Sequence[float]]) -> dict[str, object]:
    """Fit a single spherical vMF support prototype from correct bank rows."""

    values = _row_normalize(_as_2d_array(embeddings, name="embeddings"))
    resultant = values.sum(axis=0)
    resultant_norm = float(np.linalg.norm(resultant))
    if resultant_norm <= 0.0 or not np.isfinite(resultant_norm):
        mean_direction = np.zeros(values.shape[1], dtype=np.float64)
        mean_direction[0] = 1.0
        r_bar = 0.0
    else:
        mean_direction = resultant / resultant_norm
        r_bar = resultant_norm / float(values.shape[0])
    concentration = _concentration_proxy(r_bar, values.shape[1])
    log_likelihood = float(np.mean(values @ mean_direction * concentration))
    return {
        "support_family": "single_vmf",
        "k": 1,
        "num_components": 1,
        "mean_direction": mean_direction.astype(float).tolist(),
        "mean_directions": [mean_direction.astype(float).tolist()],
        "weights": [1.0],
        "concentrations": [float(concentration)],
        "resultant_length": float(resultant_norm),
        "mean_resultant_length": float(r_bar),
        "concentration_proxy": float(concentration),
        "mean_support_log_likelihood": log_likelihood,
        "num_bank_correct": int(values.shape[0]),
        "embedding_dim": int(values.shape[1]),
        "num_iterations": 1,
        "converged": True,
    }


def score_single_vmf_support(
    prototype: Mapping[str, object],
    query_embeddings: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Score queries as negative support under a single vMF direction."""

    queries = _row_normalize(_as_2d_array(query_embeddings, name="query_embeddings"))
    direction = np.asarray(prototype["mean_direction"], dtype=np.float64)
    if direction.ndim != 1 or direction.shape[0] != queries.shape[1]:
        raise ValueError("prototype mean_direction does not match query dimensionality")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("prototype mean_direction must have positive finite norm")
    direction = direction / norm
    concentration = float(prototype.get("concentration_proxy", 1.0))
    return (-(queries @ direction) * concentration).astype(np.float32)


def fit_mixture_vmf_support(
    embeddings: np.ndarray | Sequence[Sequence[float]],
    *,
    k: int,
    seed: int = 0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, object]:
    """Fit a mixture of vMF supports with spherical initialization and EM updates."""

    component_count = int(k)
    if component_count not in ALLOWED_STAGE_B4_VMF_K_VALUES:
        raise ValueError(
            "Stage B4 vMF k must be one of: "
            + ", ".join(str(value) for value in ALLOWED_STAGE_B4_VMF_K_VALUES)
        )
    values = _row_normalize(_as_2d_array(embeddings, name="embeddings"))
    if component_count > values.shape[0]:
        raise ValueError("vMF component count must not exceed the bank size")
    if component_count == 1:
        single = fit_single_vmf_support(values)
        return {
            **single,
            "support_family": "mixture_vmf",
            "k": 1,
            "num_components": 1,
        }

    directions = _directional_initialization(values, component_count, seed=seed)
    weights = np.full(component_count, 1.0 / float(component_count), dtype=np.float64)
    concentrations = np.ones(component_count, dtype=np.float64)
    previous_ll = -np.inf
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        logits = _component_logits(values, directions, concentrations, weights)
        responsibilities = _softmax(logits, axis=1)
        Nk = responsibilities.sum(axis=0)
        for component in range(component_count):
            if Nk[component] <= 1e-8:
                directions[component] = _farthest_direction(values, directions)
                weights[component] = 1.0 / float(values.shape[0])
                concentrations[component] = 1.0
                continue
            weighted = responsibilities[:, component][:, None] * values
            resultant = weighted.sum(axis=0)
            resultant_norm = float(np.linalg.norm(resultant))
            if resultant_norm <= 0.0 or not np.isfinite(resultant_norm):
                directions[component] = _farthest_direction(values, directions)
                concentrations[component] = 0.0
            else:
                directions[component] = resultant / resultant_norm
                r_bar = resultant_norm / float(Nk[component])
                concentrations[component] = _concentration_proxy(r_bar, values.shape[1])
            weights[component] = max(float(Nk[component]) / float(values.shape[0]), 1e-12)
        weights = weights / weights.sum()
        current_ll = float(np.mean(_logsumexp(_component_logits(values, directions, concentrations, weights), axis=1)))
        iterations = iteration
        if np.isfinite(previous_ll) and abs(current_ll - previous_ll) <= float(tol):
            converged = True
            break
        previous_ll = current_ll

    order = np.argsort(-weights)
    directions = directions[order]
    weights = weights[order]
    concentrations = concentrations[order]
    final_ll = float(np.mean(_logsumexp(_component_logits(values, directions, concentrations, weights), axis=1)))
    return {
        "support_family": "mixture_vmf",
        "k": component_count,
        "num_components": component_count,
        "mean_directions": directions.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "concentrations": concentrations.astype(float).tolist(),
        "mean_support_log_likelihood": final_ll,
        "num_bank_correct": int(values.shape[0]),
        "embedding_dim": int(values.shape[1]),
        "num_iterations": int(iterations),
        "converged": bool(converged),
    }


def score_mixture_vmf_support(
    model: Mapping[str, object],
    query_embeddings: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Score queries as negative log support under a vMF mixture."""

    queries = _row_normalize(_as_2d_array(query_embeddings, name="query_embeddings"))
    directions = np.asarray(model["mean_directions"], dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != queries.shape[1]:
        raise ValueError("model mean_directions do not match query dimensionality")
    directions = _row_normalize(directions)
    weights = np.asarray(model["weights"], dtype=np.float64).reshape(-1)
    concentrations = np.asarray(model["concentrations"], dtype=np.float64).reshape(-1)
    if weights.shape[0] != directions.shape[0] or concentrations.shape[0] != directions.shape[0]:
        raise ValueError("mixture weights, concentrations, and directions must have the same length")
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()
    logits = _component_logits(queries, directions, concentrations, weights)
    return (-_logsumexp(logits, axis=1)).astype(np.float32)


def build_stage_b4_vmf_support_grid(
    *,
    model_alias: str,
    dataset_family: str,
    labels: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    seed: int,
    ratio: float = REQUIRED_STAGE_B4_RATIO,
    candidates: Sequence[int] = ALLOWED_STAGE_B4_VMF_K_VALUES,
    metric_split: str = "cal",
) -> list[dict[str, object]]:
    """Evaluate the exact Stage B4 vMF candidate grid for one family/split pair."""

    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    split_array = np.asarray(splits).reshape(-1)
    embedding_array = _as_2d_array(embeddings, name="embeddings").astype(np.float32, copy=False)
    if label_array.shape[0] != embedding_array.shape[0] or split_array.shape[0] != embedding_array.shape[0]:
        raise ValueError("labels, splits, and embeddings must have the same row count")

    candidate_values = tuple(int(value) for value in candidates)
    if candidate_values != ALLOWED_STAGE_B4_VMF_K_VALUES:
        raise ValueError("Stage B4 vMF candidates must be exactly 1, 2, 4, 8")

    bank_mask = (split_array == "bank") & (label_array == 0)
    eval_mask = split_array == str(metric_split)
    num_bank_correct = int(bank_mask.sum())
    valid_cap = int(math.floor(num_bank_correct / 20.0))
    y = label_array[eval_mask]
    rows: list[dict[str, object]] = []
    for k_value in candidate_values:
        support_family = "single_vmf" if int(k_value) == 1 else "mixture_vmf"
        base = {
            "row_type": "support_candidate",
            "model_alias": str(model_alias),
            "model_name": str(model_alias),
            "objective": STAGE_B4_OBJECTIVE,
            "dataset_family": str(dataset_family),
            "readout": "Diag-vMF-support-grid",
            "support_family": support_family,
            "split": str(metric_split),
            "eval_split": str(metric_split),
            "metric_split": str(metric_split),
            "eval_scope": "pooled",
            "negative_budget_ratio": float(ratio),
            "negative_budget_seed": int(seed),
            "k": int(k_value),
            "K": int(k_value),
            "num_bank_correct": num_bank_correct,
            "bank_size": num_bank_correct,
            "valid_bank_correct_cap": valid_cap,
            "num_eval": int(y.size),
            "all_k_evaluated": True,
        }
        if num_bank_correct <= 0:
            rows.append(
                {
                    **base,
                    "metric_status": "skipped",
                    "failure_reason": "insufficient_bank_correct",
                    "skipped_reason": "insufficient_bank_correct",
                    **_undefined_metrics(),
                }
            )
            continue
        if int(k_value) != 1 and int(k_value) > valid_cap:
            rows.append(
                {
                    **base,
                    "metric_status": "skipped",
                    "failure_reason": "insufficient_bank_correct",
                    "skipped_reason": "insufficient_bank_correct",
                    **_undefined_metrics(),
                }
            )
            continue
        undefined_reason = ""
        if y.size == 0:
            undefined_reason = f"no samples in {metric_split} split"
            scores = np.asarray([], dtype=np.float32)
        else:
            if int(k_value) == 1:
                model = fit_single_vmf_support(embedding_array[bank_mask])
                scores = score_single_vmf_support(model, embedding_array[eval_mask])
            else:
                model = fit_mixture_vmf_support(embedding_array[bank_mask], k=int(k_value), seed=int(seed))
                scores = score_mixture_vmf_support(model, embedding_array[eval_mask])
            if np.unique(y).size < 2:
                undefined_reason = f"one class present in {metric_split} split"
            elif not np.isfinite(scores).all():
                undefined_reason = "non-finite scores"
        metrics = _undefined_metrics() if undefined_reason else binary_diagnostic_metrics(y, scores)
        rows.append(
            {
                **base,
                "metric_status": "undefined" if undefined_reason else "passed",
                "failure_reason": undefined_reason,
                "skipped_reason": "",
                **metrics,
            }
        )
    return rows


def select_stage_b4_vmf_k(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select vMF K on RePOPE calibration rows by PR-AUC, ROC-AUC, then smaller K."""

    candidates: list[dict[str, object]] = []
    for row in metric_rows:
        if not _is_repope_cal_vmf_row(row):
            continue
        k_value = int(row.get("k", -1))
        if k_value not in ALLOWED_STAGE_B4_VMF_K_VALUES:
            raise ValueError(f"Stage B4 vMF candidate {k_value} is outside the allowed grid")
        candidate = dict(row)
        candidate["k"] = k_value
        candidate["pr_auc"] = float(candidate["pr_auc"])
        candidate["roc_auc"] = float(candidate["roc_auc"])
        candidates.append(candidate)
    if not candidates:
        raise ValueError("no RePOPE cal vMF metric rows were available for Stage B4 K selection")

    candidates.sort(key=lambda row: (-float(row["pr_auc"]), -float(row["roc_auc"]), int(row["k"])))
    selected = candidates[0]
    return {
        "row_type": "selected",
        "model_alias": str(selected.get("model_alias") or selected.get("model_name") or ""),
        "objective": STAGE_B4_OBJECTIVE,
        "selected_k": int(selected["k"]),
        "selected_K": int(selected["k"]),
        "k": int(selected["k"]),
        "K": int(selected["k"]),
        "selected_support_family": str(selected.get("support_family", "")),
        "selected_on": "repope/cal",
        "frozen_for_test": True,
        "selection_metric": "pr_auc",
        "selection_pr_auc": float(selected["pr_auc"]),
        "selection_roc_auc": float(selected["roc_auc"]),
        "num_bank_correct": int(selected.get("num_bank_correct", 0)),
        "negative_budget_ratio": float(selected.get("negative_budget_ratio", REQUIRED_STAGE_B4_RATIO)),
        "negative_budget_seed": int(float(selected.get("negative_budget_seed", selected.get("seed", 0)))),
        "metric_status": "passed",
    }


def build_stage_b4_vmf_stability_band(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    selected_k_rows: Sequence[Mapping[str, object]],
    drop_tolerance: float = STAGE_B4_VMF_STABILITY_DROP_TOLERANCE,
    required_seed_count: int = len(REQUIRED_STAGE_B4_SEEDS),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build selected-K-centered vMF support stability bands on RePOPE test rows."""

    test_scores: dict[tuple[str, int], dict[int, float]] = {}
    for row in metric_rows:
        if str(row.get("dataset_family", "")).lower() != "repope":
            continue
        if str(row.get("metric_split") or row.get("split") or row.get("eval_split")) != "test":
            continue
        if str(row.get("metric_status", "passed")) != "passed":
            continue
        if row.get("k") is None:
            continue
        value = _finite_float(row.get("pr_auc"))
        if value is None:
            continue
        model = str(row.get("model_alias") or row.get("model_name") or "")
        if not model:
            continue
        seed = int(float(row.get("negative_budget_seed", row.get("seed", 0))))
        test_scores.setdefault((model, seed), {})[int(row["k"])] = value

    selected_by_model: dict[str, list[tuple[int, int]]] = {}
    for row in selected_k_rows:
        if str(row.get("metric_status", "passed")) != "passed":
            continue
        if row.get("selected_k") is None:
            continue
        model = str(row.get("model_alias") or row.get("model_name") or "")
        if not model:
            continue
        seed = int(float(row.get("negative_budget_seed", row.get("seed", 0))))
        selected_by_model.setdefault(model, []).append((seed, int(row["selected_k"])))

    band_rows: list[dict[str, object]] = []
    for model in sorted(selected_by_model):
        for seed, selected_k in sorted(selected_by_model[model]):
            per_k = test_scores.get((model, seed), {})
            if selected_k not in per_k:
                continue
            sorted_k = [k_value for k_value in ALLOWED_STAGE_B4_VMF_K_VALUES if k_value in per_k]
            selected_index = sorted_k.index(selected_k)
            reference = float(per_k[selected_k])
            threshold = reference - float(drop_tolerance)
            stable_flags = {
                k_value: float(per_k[k_value]) >= threshold - 1e-12
                for k_value in sorted_k
            }
            left = selected_index
            while left > 0 and stable_flags[sorted_k[left - 1]]:
                left -= 1
            right = selected_index
            while right < len(sorted_k) - 1 and stable_flags[sorted_k[right + 1]]:
                right += 1
            band_values = sorted_k[left : right + 1]
            best_k, best_test_pr_auc = sorted(
                per_k.items(),
                key=lambda item: (-float(item[1]), int(item[0])),
            )[0]
            band_rows.append(
                {
                    "model_alias": model,
                    "seed": int(seed),
                    "negative_budget_seed": int(seed),
                    "selected_k": int(selected_k),
                    "selected_K": int(selected_k),
                    "best_k": int(best_k),
                    "best_K": int(best_k),
                    "reference_test_pr_auc": reference,
                    "best_test_pr_auc": float(best_test_pr_auc),
                    "stability_drop_tolerance": float(drop_tolerance),
                    "stability_band_min_k": int(band_values[0]),
                    "stability_band_max_k": int(band_values[-1]),
                    "stability_band_min_K": int(band_values[0]),
                    "stability_band_max_K": int(band_values[-1]),
                    "stability_band_size": len(band_values),
                    "stability_band_values": ";".join(str(value) for value in band_values),
                    "metric_status": "passed",
                    "failure_reason": "",
                }
            )

    band_rows_by_model: dict[str, list[dict[str, object]]] = {}
    for row in band_rows:
        band_rows_by_model.setdefault(str(row["model_alias"]), []).append(row)

    per_model_rows: list[dict[str, object]] = []
    for model in sorted(selected_by_model):
        rows = sorted(band_rows_by_model.get(model, []), key=lambda row: int(row["seed"]))
        if len(rows) < int(required_seed_count):
            per_model_rows.append(
                {
                    "model_alias": model,
                    "verdict": "insufficient_coverage",
                    "status": "incomplete",
                    "median_band_size": "",
                    "min_band_size": "",
                    "max_band_size": "",
                    "num_valid_runs": len(rows),
                    "required_seed_count": int(required_seed_count),
                    "seed_band_sizes": _seed_band_sizes(rows),
                    "selected_k_values": _seed_selected_k_values(rows),
                    "stability_drop_tolerance": float(drop_tolerance),
                    "reason": "fewer valid selected-k test runs than required",
                }
            )
            continue
        sizes = [int(row["stability_band_size"]) for row in rows]
        per_model_rows.append(
            {
                "model_alias": model,
                "verdict": "vmf_support_stable" if float(np.median(sizes)) >= 2.0 else "vmf_support_sensitive",
                "status": "evaluated",
                "median_band_size": float(np.median(np.asarray(sizes, dtype=np.float64))),
                "min_band_size": int(min(sizes)),
                "max_band_size": int(max(sizes)),
                "num_valid_runs": len(rows),
                "required_seed_count": int(required_seed_count),
                "seed_band_sizes": _seed_band_sizes(rows),
                "selected_k_values": _seed_selected_k_values(rows),
                "stability_drop_tolerance": float(drop_tolerance),
                "reason": "",
            }
        )
    return band_rows, per_model_rows


def _is_repope_cal_vmf_row(row: Mapping[str, object]) -> bool:
    split = str(row.get("metric_split") or row.get("split") or row.get("eval_split") or "")
    return (
        str(row.get("dataset_family", "")).lower() == "repope"
        and split == "cal"
        and str(row.get("metric_status", "passed")) == "passed"
        and row.get("pr_auc") is not None
        and row.get("roc_auc") is not None
        and row.get("k") is not None
    )


def _component_logits(
    values: np.ndarray,
    directions: np.ndarray,
    concentrations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    safe_weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, None)
    safe_weights = safe_weights / safe_weights.sum()
    return values @ directions.T * concentrations.reshape(1, -1) + np.log(safe_weights).reshape(1, -1)


def _directional_initialization(values: np.ndarray, k: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    resultant = values.sum(axis=0)
    norm = float(np.linalg.norm(resultant))
    if norm <= 0.0 or not np.isfinite(norm):
        first = int(rng.integers(values.shape[0]))
    else:
        mean_direction = resultant / norm
        first = int(np.argmax(values @ mean_direction))
    selected = [first]
    while len(selected) < int(k):
        chosen = values[np.asarray(selected, dtype=np.int64)]
        nearest_cosine = np.max(values @ chosen.T, axis=1)
        nearest_cosine[np.asarray(selected, dtype=np.int64)] = np.inf
        jitter = rng.uniform(0.0, 1e-9, size=nearest_cosine.shape)
        selected.append(int(np.argmin(nearest_cosine + jitter)))
    return values[np.asarray(selected, dtype=np.int64)].copy()


def _farthest_direction(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    nearest_cosine = np.max(values @ directions.T, axis=1)
    return values[int(np.argmin(nearest_cosine))].copy()


def _concentration_proxy(r_bar: float, dim: int) -> float:
    clipped = min(max(float(r_bar), 0.0), 0.999999)
    if clipped <= 0.0:
        return 0.0
    value = clipped * (float(dim) - clipped * clipped) / max(1.0 - clipped * clipped, 1e-12)
    if not np.isfinite(value):
        return 0.0
    return max(float(value), 0.0)


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


def _softmax(value: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _logsumexp(value: np.ndarray, *, axis: int) -> np.ndarray:
    max_value = np.max(value, axis=axis, keepdims=True)
    result = max_value + np.log(np.sum(np.exp(value - max_value), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _seed_band_sizes(rows: Sequence[Mapping[str, object]]) -> str:
    return ";".join(
        f"{int(row['seed'])}:{int(row['stability_band_size'])}"
        for row in rows
    )


def _seed_selected_k_values(rows: Sequence[Mapping[str, object]]) -> str:
    return ";".join(
        f"{int(row['seed'])}:{int(row['selected_k'])}"
        for row in rows
    )


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _undefined_metrics() -> dict[str, float]:
    return {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
        "fpr_at_95pct_tpr": float("nan"),
    }


__all__ = [
    "ALLOWED_STAGE_B4_VMF_K_VALUES",
    "STAGE_B4_VMF_STABILITY_DROP_TOLERANCE",
    "build_stage_b4_vmf_stability_band",
    "build_stage_b4_vmf_support_grid",
    "fit_mixture_vmf_support",
    "fit_single_vmf_support",
    "score_mixture_vmf_support",
    "score_single_vmf_support",
    "select_stage_b4_vmf_k",
]
