from __future__ import annotations

import numpy as np
import pytest

from .conftest import stage_c_attr


def test_stage_c_hyperparameter_grids_are_frozen() -> None:
    support = stage_c_attr("stage_c_support", "stage_c_hyperparameter_grids")()

    assert support["single_vmf"] == []
    assert support["mixture_vmf_K"] == [2, 4, 8]
    assert support["knn_k"] == [1, 2, 4, 8, 16, 32, 64]
    assert support["radius_ball_quantiles"] == [0.50, 0.65, 0.80, 0.90, 0.95]
    assert support["logistic_C"] == [0.1, 1.0, 10.0]


def test_stage_c_selects_candidates_on_repope_cal_with_tie_breaks() -> None:
    select_candidate = stage_c_attr("stage_c_support", "select_stage_c_candidate")

    selected = select_candidate(
        [
            _candidate_row(method="mixture_vmf", value=2, pr_auc=0.71, roc_auc=0.80),
            _candidate_row(method="mixture_vmf", value=4, pr_auc=0.72, roc_auc=0.75),
            _candidate_row(method="mixture_vmf", value=8, pr_auc=0.72, roc_auc=0.75),
            _candidate_row(method="mixture_vmf", value=2, pr_auc=0.99, roc_auc=0.99, dataset="pope"),
        ],
        method="mixture_vmf",
        parameter_name="K",
        allowed_values=(2, 4, 8),
    )

    assert selected["selected_K"] == 4
    assert selected["selected_on"] == "repope/cal"
    assert selected["frozen_for_test"] is True
    assert selected["selection_pr_auc"] == pytest.approx(0.72)
    assert selected["selection_roc_auc"] == pytest.approx(0.75)


def test_stage_c_radius_ball_candidates_use_calibration_support_quantiles_only() -> None:
    build_radius_candidates = stage_c_attr("stage_c_support", "build_stage_c_radius_candidates")

    bank = _unit_rows([[1, 0], [0.98, 0.02], [0, 1], [0.02, 0.98], [-1, 0], [-0.98, 0.02]])
    cal = _unit_rows([[1, 0], [0, 1], [-1, 0], [0.7, 0.7]])
    cal_labels = np.asarray([0, 0, 0, 1], dtype=np.int64)

    rows = build_radius_candidates(bank_embeddings=bank, cal_embeddings=cal, cal_labels=cal_labels)

    assert [row["quantile"] for row in rows] == [0.50, 0.65, 0.80, 0.90, 0.95]
    assert all(row["selection_split"] == "repope/cal" for row in rows)
    assert all(row["source"] == "calibration_correct_support_radii" for row in rows)
    assert all(float(row["rho"]) >= 0.0 for row in rows)


def _candidate_row(
    *,
    method: str,
    value: int | float,
    pr_auc: float,
    roc_auc: float,
    dataset: str = "repope",
    split: str = "cal",
) -> dict[str, object]:
    return {
        "model_alias": "model-a",
        "method": method,
        "dataset_family": dataset,
        "metric_split": split,
        "metric_status": "passed",
        "parameter_value": value,
        "K": value if method == "mixture_vmf" else "",
        "k": value if method == "knn" else "",
        "rho": value if method == "radius_ball" else "",
        "C": value if method == "logistic" else "",
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "negative_budget_seed": 20260506,
    }


def _unit_rows(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)
