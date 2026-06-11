from __future__ import annotations

import numpy as np
import pytest

from .conftest import EXPECTED_STAGE_B_K_VALUES, repope_cal_metric_row, stage_b3_attr


def test_stage_b3_knn_candidates_reuse_stage_b_allowed_grid_and_bank_clip() -> None:
    allowed_values = stage_b3_attr("stage_b3_knn", "ALLOWED_STAGE_B3_K_VALUES")
    generate_stage_b3_k_candidates = stage_b3_attr(
        "stage_b3_knn",
        "generate_stage_b3_k_candidates",
    )

    assert tuple(allowed_values) == EXPECTED_STAGE_B_K_VALUES
    assert generate_stage_b3_k_candidates(num_bank_correct=5) == (1, 2)
    assert generate_stage_b3_k_candidates(num_bank_correct=70) == (1, 2, 4, 8)


def test_stage_b3_knn_scale_grid_evaluates_every_candidate_on_repope_cal() -> None:
    build_stage_b3_knn_scale_grid = stage_b3_attr(
        "stage_b3_knn",
        "build_stage_b3_knn_scale_grid",
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.85, 0.15],
            [0.15, 0.85],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 0, 0, 1, 0, 1], dtype=np.int64)
    splits = np.asarray(["bank", "bank", "cal", "cal", "bank", "cal", "bank", "cal"])

    rows = build_stage_b3_knn_scale_grid(
        model_alias="model-a",
        dataset_family="repope",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=20260506,
        ratio=0.5,
        candidates=(1, 2),
    )

    assert [row["k"] for row in rows] == [1, 2]
    assert {row["row_type"] for row in rows} == {"scale_candidate"}
    assert {row["metric_split"] for row in rows} == {"cal"}
    assert {row["all_k_evaluated"] for row in rows} == {True}
    assert {row["objective"] for row in rows} == {"proxy_anchor"}
    assert all(row["metric_status"] == "passed" for row in rows)


def test_stage_b3_knn_scale_grid_clips_supplied_candidates_by_current_bank() -> None:
    build_stage_b3_knn_scale_grid = stage_b3_attr(
        "stage_b3_knn",
        "build_stage_b3_knn_scale_grid",
    )
    embeddings = np.eye(8, dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    splits = np.asarray(["bank", "bank", "test", "test", "test", "test", "test", "test"])

    rows = build_stage_b3_knn_scale_grid(
        model_alias="model-a",
        dataset_family="dash-b",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=20260506,
        ratio=0.5,
        candidates=(1, 2, 4, 8, 16, 32),
        metric_split="test",
    )

    assert [row["k"] for row in rows] == [1]
    assert {row["num_bank_correct"] for row in rows} == {2}


def test_stage_b3_knn_selects_from_repope_cal_and_rejects_invalid_candidate() -> None:
    select_stage_b3_knn_k = stage_b3_attr("stage_b3_knn", "select_stage_b3_knn_k")

    selected = select_stage_b3_knn_k(
        [
            repope_cal_metric_row(1, pr_auc=0.50, roc_auc=0.80),
            repope_cal_metric_row(2, pr_auc=0.60, roc_auc=0.70),
            repope_cal_metric_row(4, pr_auc=0.60, roc_auc=0.75),
            repope_cal_metric_row(8, pr_auc=0.99, roc_auc=0.99, split="test"),
        ]
    )

    assert selected["k"] == 4
    assert selected["selected_on"] == "repope/cal"
    assert selected["frozen_for_test"] is True

    with pytest.raises(ValueError, match="candidate|allowed"):
        select_stage_b3_knn_k([repope_cal_metric_row(3, pr_auc=0.9, roc_auc=0.9)])
