"""Stage B3 kNN scale-grid and stability-band helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .stage_a_metrics import binary_diagnostic_metrics
from .stage_b_knn import (
    ALLOWED_STAGE_B_K_VALUES,
    compute_stage_b_knn_scores,
    generate_stage_b_k_candidates,
    select_stage_b_knn_k,
)
from .stage_b3_manifest import (
    REQUIRED_STAGE_B3_RATIO,
    REQUIRED_STAGE_B3_SEEDS,
    STAGE_B3_OBJECTIVE,
)


ALLOWED_STAGE_B3_K_VALUES = ALLOWED_STAGE_B_K_VALUES
STAGE_B3_STABILITY_DROP_TOLERANCE = 0.02


def generate_stage_b3_k_candidates(
    *,
    num_bank_correct: int,
    bank_size: int | None = None,
) -> tuple[int, ...]:
    """Generate the Stage B3 k candidate grid using the Stage B bank evidence rule."""

    return generate_stage_b_k_candidates(
        num_bank_correct=int(num_bank_correct),
        bank_size=bank_size,
    )


def build_stage_b3_knn_scale_grid(
    *,
    model_alias: str,
    dataset_family: str,
    labels: Sequence[int] | np.ndarray,
    splits: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    seed: int,
    ratio: float = REQUIRED_STAGE_B3_RATIO,
    candidates: Sequence[int] | None = None,
    metric_split: str = "cal",
) -> list[dict[str, object]]:
    """Evaluate every k candidate for one family/split pair."""

    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    split_array = np.asarray(splits).reshape(-1)
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    if embedding_array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if label_array.shape[0] != embedding_array.shape[0] or split_array.shape[0] != embedding_array.shape[0]:
        raise ValueError("labels, splits, and embeddings must have the same row count")

    bank_mask = (split_array == "bank") & (label_array == 0)
    eval_mask = split_array == str(metric_split)
    num_bank_correct = int(bank_mask.sum())
    candidate_values = tuple(
        int(value)
        for value in (
            candidates
            if candidates is not None
            else generate_stage_b3_k_candidates(num_bank_correct=num_bank_correct)
        )
    )
    if not candidate_values:
        raise ValueError("Stage B3 kNN scale grid has no k candidates")

    y = label_array[eval_mask]
    rows: list[dict[str, object]] = []
    for k_value in candidate_values:
        if int(k_value) not in ALLOWED_STAGE_B3_K_VALUES:
            raise ValueError(f"Stage B3 k candidate {k_value} is outside the allowed grid")
        undefined_reason = ""
        if y.size == 0:
            undefined_reason = f"no samples in {metric_split} split"
            scores = np.asarray([], dtype=np.float32)
        else:
            scores = compute_stage_b_knn_scores(
                bank_embeddings=embedding_array[bank_mask],
                query_embeddings=embedding_array[eval_mask],
                k=int(k_value),
            )
            if np.unique(y).size < 2:
                undefined_reason = f"one class present in {metric_split} split"
            elif not np.isfinite(scores).all():
                undefined_reason = "non-finite scores"
        metrics = _undefined_metrics() if undefined_reason else binary_diagnostic_metrics(y, scores)
        rows.append(
            {
                "row_type": "scale_candidate",
                "model_alias": str(model_alias),
                "model_name": str(model_alias),
                "objective": STAGE_B3_OBJECTIVE,
                "dataset_family": str(dataset_family),
                "readout": "Diag-kNN-scale-grid",
                "split": str(metric_split),
                "eval_split": str(metric_split),
                "metric_split": str(metric_split),
                "eval_scope": "pooled",
                "negative_budget_ratio": float(ratio),
                "negative_budget_seed": int(seed),
                "k": int(k_value),
                "scale_id": f"k={int(k_value)}",
                "num_bank_correct": num_bank_correct,
                "bank_size": num_bank_correct,
                "num_eval": int(y.size),
                "all_k_evaluated": True,
                "metric_status": "undefined" if undefined_reason else "passed",
                "failure_reason": undefined_reason,
                **metrics,
            }
        )
    return rows


def select_stage_b3_knn_k(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select k on RePOPE calibration rows and freeze it for Stage B3 test rows."""

    selected = dict(select_stage_b_knn_k(metric_rows))
    selected["selected_on"] = "repope/cal"
    selected["frozen_for_test"] = True
    selected["objective"] = STAGE_B3_OBJECTIVE
    return selected


def build_stage_b3_stability_band(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    selected_k_rows: Sequence[Mapping[str, object]],
    drop_tolerance: float = STAGE_B3_STABILITY_DROP_TOLERANCE,
    required_seed_count: int = len(REQUIRED_STAGE_B3_SEEDS),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build selected-k-centered per-seed bands and per-model stability rows."""

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
            sorted_k = sorted(per_k)
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
                    "best_k": int(best_k),
                    "reference_test_pr_auc": reference,
                    "best_test_pr_auc": float(best_test_pr_auc),
                    "stability_drop_tolerance": float(drop_tolerance),
                    "stability_band_min_k": int(band_values[0]),
                    "stability_band_max_k": int(band_values[-1]),
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
        median_size = float(np.median(np.asarray(sizes, dtype=np.float64)))
        per_model_rows.append(
            {
                "model_alias": model,
                "verdict": "scale_stable" if median_size >= 3.0 else "scale_sensitive",
                "status": "evaluated",
                "median_band_size": median_size,
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


def _undefined_metrics() -> dict[str, float]:
    return {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
        "fpr_at_95pct_tpr": float("nan"),
    }


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


__all__ = [
    "ALLOWED_STAGE_B3_K_VALUES",
    "STAGE_B3_STABILITY_DROP_TOLERANCE",
    "build_stage_b3_knn_scale_grid",
    "build_stage_b3_stability_band",
    "compute_stage_b_knn_scores",
    "generate_stage_b3_k_candidates",
    "select_stage_b3_knn_k",
]
