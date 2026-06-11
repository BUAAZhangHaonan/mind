from __future__ import annotations

import pytest

from .conftest import EXPECTED_STAGE_B_K_VALUES, repope_cal_metric_row, stage_b_attr


def test_knn_autotune_uses_allowed_candidates_and_repope_cal_tie_breaks() -> None:
    allowed_k_values = stage_b_attr(
        "stage_b_knn",
        "ALLOWED_STAGE_B_K_VALUES",
    )
    select_stage_b_knn_k = stage_b_attr(
        "stage_b_knn",
        "select_stage_b_knn_k",
    )
    rows = [
        repope_cal_metric_row(8, pr_auc=0.99, roc_auc=0.99, dataset_family="pope"),
        repope_cal_metric_row(16, pr_auc=0.99, roc_auc=0.99, split="test"),
        repope_cal_metric_row(1, pr_auc=0.72, roc_auc=0.80),
        repope_cal_metric_row(2, pr_auc=0.72, roc_auc=0.85),
        repope_cal_metric_row(4, pr_auc=0.72, roc_auc=0.85),
        repope_cal_metric_row(8, pr_auc=0.72, roc_auc=0.79),
    ]

    selected = select_stage_b_knn_k(rows)

    assert tuple(allowed_k_values) == EXPECTED_STAGE_B_K_VALUES
    assert selected["k"] == 2
    assert selected["dataset_family"] == "repope"
    assert selected["split"] == "cal"
    assert selected["pr_auc"] == pytest.approx(0.72)
    assert selected["roc_auc"] == pytest.approx(0.85)


def test_knn_autotune_rejects_k_outside_allowed_candidate_set() -> None:
    select_stage_b_knn_k = stage_b_attr(
        "stage_b_knn",
        "select_stage_b_knn_k",
    )

    with pytest.raises(ValueError, match="candidate|allowed"):
        select_stage_b_knn_k([repope_cal_metric_row(3, pr_auc=0.90, roc_auc=0.90)])


def test_knn_candidate_grid_clips_to_available_calibration_bank() -> None:
    clipped_stage_b_k_values = stage_b_attr(
        "stage_b_knn",
        "clipped_stage_b_k_values",
    )

    assert tuple(clipped_stage_b_k_values(num_bank_correct=5)) == (1, 2)
    assert tuple(clipped_stage_b_k_values(num_bank_correct=70)) == (1, 2, 4, 8)
