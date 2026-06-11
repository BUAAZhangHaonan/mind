"""Stage B spherical kNN helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


ALLOWED_STAGE_B_K_VALUES = (1, 2, 4, 8, 16, 32, 64)


def generate_stage_b_k_candidates(
    *,
    num_bank_correct: int,
    bank_size: int | None = None,
    allowed_values: Sequence[int] = ALLOWED_STAGE_B_K_VALUES,
) -> tuple[int, ...]:
    """Generate auto-tuned Stage B k candidates clipped by bank evidence."""

    correct = int(num_bank_correct)
    if correct <= 0:
        return ()
    size_cap = correct if bank_size is None else min(correct, int(bank_size))
    sqrt_cap = int(math.floor(math.sqrt(correct)))
    cap = min(size_cap, sqrt_cap)
    return tuple(int(k) for k in allowed_values if int(k) <= cap)


def clipped_stage_b_k_values(
    *,
    num_bank_correct: int,
    bank_size: int | None = None,
) -> tuple[int, ...]:
    """Public alias used by Stage B k auto-tuning tests."""

    return generate_stage_b_k_candidates(
        num_bank_correct=num_bank_correct,
        bank_size=bank_size,
    )


def stage_b_k_candidates(
    num_bank_correct: int,
    bank_size: int | None = None,
) -> tuple[int, ...]:
    """Backward-compatible alias for candidate generation."""

    return generate_stage_b_k_candidates(
        num_bank_correct=num_bank_correct,
        bank_size=bank_size,
    )


def geodesic_distance(left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]) -> np.ndarray | float:
    """Compute angular distance on the unit sphere."""

    left_array = _l2_normalize(np.asarray(left, dtype=np.float64))
    right_array = _l2_normalize(np.asarray(right, dtype=np.float64))
    left_was_vector = left_array.ndim == 1
    right_was_vector = right_array.ndim == 1
    if left_was_vector:
        left_array = left_array.reshape(1, -1)
    if right_was_vector:
        right_array = right_array.reshape(1, -1)
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("left and right embeddings must have the same dimensionality")

    cosine = np.clip(left_array @ right_array.T, -1.0, 1.0)
    distances = np.arccos(cosine)
    if left_was_vector and right_was_vector:
        return float(distances[0, 0])
    if left_was_vector or right_was_vector:
        return distances.reshape(-1)
    return distances


def compute_stage_b_knn_scores(
    *,
    bank_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """Score queries by mean geodesic distance to their k nearest bank rows."""

    bank = _as_2d_array(bank_embeddings, name="bank_embeddings")
    queries = _as_2d_array(query_embeddings, name="query_embeddings")
    if bank.shape[1] != queries.shape[1]:
        raise ValueError("bank and query embeddings must have the same dimensionality")
    if int(k) <= 0 or int(k) > bank.shape[0]:
        raise ValueError("k must be between 1 and the bank size")
    distances = geodesic_distance(queries, bank)
    assert isinstance(distances, np.ndarray)
    nearest = np.partition(distances, kth=int(k) - 1, axis=1)[:, : int(k)]
    return nearest.mean(axis=1).astype(np.float32)


def select_stage_b_knn_k(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select k on RePOPE cal rows by PR-AUC, ROC-AUC, then smaller k."""

    candidates: list[dict[str, object]] = []
    for row in metric_rows:
        if not _is_repope_cal_row(row):
            continue
        k = int(row.get("k", -1))
        allowed = _allowed_k_values_for_row(row)
        if k not in allowed:
            raise ValueError(
                f"Stage B k candidate {k} is outside the allowed candidate set {allowed}"
            )
        candidate = dict(row)
        candidate["k"] = k
        candidate["pr_auc"] = float(candidate["pr_auc"])
        candidate["roc_auc"] = float(candidate["roc_auc"])
        candidates.append(candidate)

    if not candidates:
        raise ValueError("no RePOPE cal kNN metric rows were available for Stage B k selection")

    candidates.sort(key=lambda row: (-float(row["pr_auc"]), -float(row["roc_auc"]), int(row["k"])))
    return candidates[0]


def _is_repope_cal_row(row: Mapping[str, object]) -> bool:
    split = str(row.get("metric_split") or row.get("split") or row.get("eval_split") or "")
    return (
        str(row.get("dataset_family", "")).lower() == "repope"
        and split == "cal"
        and str(row.get("metric_status", "passed")) == "passed"
        and row.get("pr_auc") is not None
        and row.get("roc_auc") is not None
        and row.get("k") is not None
    )


def _allowed_k_values_for_row(row: Mapping[str, object]) -> tuple[int, ...]:
    if row.get("num_bank_correct") is None:
        return ALLOWED_STAGE_B_K_VALUES
    return generate_stage_b_k_candidates(
        num_bank_correct=int(row["num_bank_correct"]),
        bank_size=int(row.get("bank_size", row["num_bank_correct"])),
    )


def _as_2d_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("embedding vector must have positive finite norm")
        return array / norm
    if array.ndim != 2:
        raise ValueError("embeddings must be a vector or 2D matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("embedding rows must have positive finite norm")
    return array / norms
