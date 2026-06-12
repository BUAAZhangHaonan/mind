from __future__ import annotations

import numpy as np
import pytest

from .conftest import stage_b4_attr


def _unit_bank(num_per_cluster: int = 24) -> np.ndarray:
    rows: list[list[float]] = []
    for _ in range(num_per_cluster):
        rows.append([1.0, 0.04, 0.0])
        rows.append([0.04, 1.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


def test_stage_b4_single_and_mixture_k1_have_same_scores_within_tolerance() -> None:
    fit_single = stage_b4_attr("stage_b4_vmf", "fit_single_vmf_support")
    score_single = stage_b4_attr("stage_b4_vmf", "score_single_vmf_support")
    fit_mixture = stage_b4_attr("stage_b4_vmf", "fit_mixture_vmf_support")
    score_mixture = stage_b4_attr("stage_b4_vmf", "score_mixture_vmf_support")

    bank = _unit_bank()
    queries = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    single = fit_single(bank)
    mixture = fit_mixture(bank, k=1, seed=20260506)

    assert single["support_family"] == "single_vmf"
    assert mixture["support_family"] == "mixture_vmf"
    assert mixture["k"] == 1
    assert mixture["num_components"] == 1
    assert mixture["num_bank_correct"] == bank.shape[0]
    np.testing.assert_allclose(
        score_mixture(mixture, queries),
        score_single(single, queries),
        rtol=1e-6,
        atol=1e-6,
    )


def test_stage_b4_vmf_grid_uses_exact_candidates_and_records_skipped_k() -> None:
    build_grid = stage_b4_attr("stage_b4_vmf", "build_stage_b4_vmf_support_grid")

    bank = _unit_bank(num_per_cluster=24)
    cal = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = np.concatenate([bank, cal], axis=0)
    labels = np.asarray([0] * bank.shape[0] + [0, 0, 1, 1], dtype=np.int64)
    splits = np.asarray(["bank"] * bank.shape[0] + ["cal"] * cal.shape[0])

    rows = build_grid(
        model_alias="model-a",
        dataset_family="repope",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=20260506,
        metric_split="cal",
    )

    assert [row["k"] for row in rows] == [1, 2, 4, 8]
    assert [row["metric_status"] for row in rows] == ["passed", "passed", "skipped", "skipped"]
    assert {row["support_family"] for row in rows} == {"single_vmf", "mixture_vmf"}
    skipped = [row for row in rows if row["metric_status"] == "skipped"]
    assert all(row["skipped_reason"] == "insufficient_bank_correct" for row in skipped)
    assert all(row["valid_bank_correct_cap"] == 2 for row in rows)


def test_stage_b4_vmf_grid_keeps_single_vmf_when_bank_smaller_than_component_rule() -> None:
    build_grid = stage_b4_attr("stage_b4_vmf", "build_stage_b4_vmf_support_grid")

    bank = _unit_bank(num_per_cluster=4)
    cal = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = np.concatenate([bank, cal], axis=0)
    labels = np.asarray([0] * bank.shape[0] + [0, 0, 1, 1], dtype=np.int64)
    splits = np.asarray(["bank"] * bank.shape[0] + ["cal"] * cal.shape[0])

    rows = build_grid(
        model_alias="model-small-bank",
        dataset_family="repope",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=20260506,
        metric_split="cal",
    )

    by_k = {int(row["k"]): row for row in rows}
    assert by_k[1]["metric_status"] == "passed"
    assert by_k[1]["support_family"] == "single_vmf"
    assert [by_k[k]["metric_status"] for k in (2, 4, 8)] == ["skipped", "skipped", "skipped"]
    assert all(by_k[k]["skipped_reason"] == "insufficient_bank_correct" for k in (2, 4, 8))


def test_stage_b4_vmf_selection_uses_repope_cal_pr_auc_then_roc_auc_then_smaller_k() -> None:
    select_k = stage_b4_attr("stage_b4_vmf", "select_stage_b4_vmf_k")

    selected = select_k(
        [
            _metric_row(k=1, pr_auc=0.82, roc_auc=0.70),
            _metric_row(k=2, pr_auc=0.84, roc_auc=0.60),
            _metric_row(k=4, pr_auc=0.84, roc_auc=0.62),
            _metric_row(k=8, pr_auc=0.84, roc_auc=0.62, metric_status="skipped"),
            _metric_row(k=1, pr_auc=0.99, roc_auc=0.99, dataset_family="pope"),
        ]
    )

    assert selected["selected_k"] == 4
    assert selected["selected_K"] == 4
    assert selected["selection_pr_auc"] == pytest.approx(0.84)
    assert selected["selection_roc_auc"] == pytest.approx(0.62)
    assert selected["selected_on"] == "repope/cal"
    assert selected["frozen_for_test"] is True


def test_stage_b4_vmf_stability_band_is_contiguous_around_selected_k() -> None:
    build_band = stage_b4_attr("stage_b4_vmf", "build_stage_b4_vmf_stability_band")

    rows = [
        _metric_row(k=1, pr_auc=0.91, split="test"),
        _metric_row(k=2, pr_auc=0.86, split="test"),
        _metric_row(k=4, pr_auc=0.90, split="test"),
        _metric_row(k=8, pr_auc=0.881, split="test"),
    ]
    selected_rows = [
        {
            "model_alias": "model-a",
            "negative_budget_seed": 20260506,
            "selected_k": 4,
            "metric_status": "passed",
        }
    ]

    band_rows, per_model_rows = build_band(
        rows,
        selected_k_rows=selected_rows,
        drop_tolerance=0.02,
        required_seed_count=1,
    )

    assert band_rows[0]["stability_band_values"] == "4;8"
    assert band_rows[0]["best_k"] == 1
    assert per_model_rows[0]["model_alias"] == "model-a"
    assert per_model_rows[0]["num_valid_runs"] == 1


def _metric_row(
    *,
    k: int,
    pr_auc: float,
    roc_auc: float = 0.75,
    model_alias: str = "model-a",
    dataset_family: str = "repope",
    split: str = "cal",
    metric_status: str = "passed",
) -> dict[str, object]:
    return {
        "model_alias": model_alias,
        "dataset_family": dataset_family,
        "readout": "Diag-vMF-support-grid",
        "support_family": "single_vmf" if k == 1 else "mixture_vmf",
        "metric_split": split,
        "eval_split": split,
        "negative_budget_ratio": 0.5,
        "negative_budget_seed": 20260506,
        "metric_status": metric_status,
        "k": k,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "num_bank_correct": 160,
        "bank_size": 160,
    }
