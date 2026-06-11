from __future__ import annotations

import pytest

from tests.stage_b.conftest import repope_cal_metric_row

from .conftest import EXPECTED_STAGE_B_K_VALUES, stage_b2_attr


def test_stage_b2_knn_selects_from_repope_cal_and_freezes_selected_k() -> None:
    select_stage_b2_knn_k = stage_b2_attr("stage_b2_knn", "select_stage_b2_knn_k")
    allowed_values = stage_b2_attr("stage_b2_knn", "ALLOWED_STAGE_B2_K_VALUES")
    rows = [
        repope_cal_metric_row(1, pr_auc=0.50, roc_auc=0.80),
        repope_cal_metric_row(2, pr_auc=0.60, roc_auc=0.70),
        repope_cal_metric_row(4, pr_auc=0.60, roc_auc=0.75),
        repope_cal_metric_row(8, pr_auc=0.99, roc_auc=0.99, split="test"),
    ]

    selected = select_stage_b2_knn_k(rows)

    assert tuple(allowed_values) == EXPECTED_STAGE_B_K_VALUES
    assert selected["k"] == 4
    assert selected["selected_on"] == "repope/cal"
    assert selected["frozen_for_test"] is True


def test_stage_b2_knn_tie_breaks_to_smaller_k_after_pr_and_roc() -> None:
    select_stage_b2_knn_k = stage_b2_attr("stage_b2_knn", "select_stage_b2_knn_k")

    selected = select_stage_b2_knn_k(
        [
            repope_cal_metric_row(2, pr_auc=0.60, roc_auc=0.75),
            repope_cal_metric_row(4, pr_auc=0.60, roc_auc=0.75),
        ]
    )

    assert selected["k"] == 2


def test_stage_b2_knn_rejects_invalid_candidate() -> None:
    select_stage_b2_knn_k = stage_b2_attr("stage_b2_knn", "select_stage_b2_knn_k")

    with pytest.raises(ValueError, match="candidate|allowed"):
        select_stage_b2_knn_k([repope_cal_metric_row(3, pr_auc=0.9, roc_auc=0.9)])
