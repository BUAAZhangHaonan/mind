from __future__ import annotations

import pytest

from .conftest import stage_b3_attr


def _row(model: str, k: int, seed: int, pr_auc: float) -> dict[str, object]:
    return {
        "model_alias": model,
        "dataset_family": "repope",
        "readout": "Diag-kNN-scale-grid",
        "metric_split": "test",
        "metric_status": "passed",
        "negative_budget_ratio": 0.5,
        "negative_budget_seed": seed,
        "k": k,
        "pr_auc": pr_auc,
        "roc_auc": pr_auc,
    }


def _selected(model: str, seed: int, selected_k: int) -> dict[str, object]:
    return {
        "model_alias": model,
        "negative_budget_seed": seed,
        "selected_k": selected_k,
        "selected_on": "repope/cal",
        "metric_status": "passed",
    }


def test_stage_b3_stability_band_uses_seed_selected_k_reference_and_contiguous_band() -> None:
    build_stage_b3_stability_band = stage_b3_attr(
        "stage_b3_knn",
        "build_stage_b3_stability_band",
    )
    rows = [
        _row("model-a", 1, 20260506, 0.91),
        _row("model-a", 2, 20260506, 0.86),
        _row("model-a", 4, 20260506, 0.90),
        _row("model-a", 8, 20260506, 0.881),
        _row("model-a", 16, 20260506, 0.879),
    ]

    band_rows, per_model_rows = build_stage_b3_stability_band(
        rows,
        selected_k_rows=[_selected("model-a", 20260506, 4)],
        drop_tolerance=0.02,
        required_seed_count=1,
    )

    assert band_rows == [
        {
            "model_alias": "model-a",
            "seed": 20260506,
            "negative_budget_seed": 20260506,
            "selected_k": 4,
            "best_k": 1,
            "reference_test_pr_auc": pytest.approx(0.90),
            "best_test_pr_auc": pytest.approx(0.91),
            "stability_drop_tolerance": 0.02,
            "stability_band_min_k": 4,
            "stability_band_max_k": 8,
            "stability_band_size": 2,
            "stability_band_values": "4;8",
            "metric_status": "passed",
            "failure_reason": "",
        }
    ]
    assert per_model_rows == [
        {
            "model_alias": "model-a",
            "verdict": "scale_sensitive",
            "status": "evaluated",
            "median_band_size": 2.0,
            "min_band_size": 2,
            "max_band_size": 2,
            "num_valid_runs": 1,
            "required_seed_count": 1,
            "seed_band_sizes": "20260506:2",
            "selected_k_values": "20260506:4",
            "stability_drop_tolerance": 0.02,
            "reason": "",
        }
    ]


def test_stage_b3_per_model_scale_stability_uses_median_band_size_across_seeds() -> None:
    build_stage_b3_stability_band = stage_b3_attr(
        "stage_b3_knn",
        "build_stage_b3_stability_band",
    )
    rows = []
    selected = []
    seed_to_values = {
        20260506: {1: 0.90, 2: 0.895, 4: 0.89, 8: 0.50},
        20260507: {1: 0.50, 2: 0.89, 4: 0.90, 8: 0.885},
        20260508: {1: 0.50, 2: 0.50, 4: 0.90, 8: 0.879},
    }
    for seed, values in seed_to_values.items():
        selected.append(_selected("model-a", seed, 4))
        for k, pr_auc in values.items():
            rows.append(_row("model-a", k, seed, pr_auc))

    band_rows, per_model_rows = build_stage_b3_stability_band(
        rows,
        selected_k_rows=selected,
        drop_tolerance=0.02,
    )

    assert [row["stability_band_values"] for row in band_rows] == ["1;2;4", "2;4;8", "4"]
    assert per_model_rows[0]["verdict"] == "scale_stable"
    assert per_model_rows[0]["median_band_size"] == 3.0
    assert per_model_rows[0]["seed_band_sizes"] == "20260506:3;20260507:3;20260508:1"


def test_stage_b3_stability_band_marks_missing_selected_or_test_rows_insufficient() -> None:
    build_stage_b3_stability_band = stage_b3_attr(
        "stage_b3_knn",
        "build_stage_b3_stability_band",
    )

    band_rows, per_model_rows = build_stage_b3_stability_band(
        [_row("model-a", 4, 20260506, 0.90)],
        selected_k_rows=[_selected("model-a", 20260506, 4), _selected("model-a", 20260507, 4)],
    )

    assert len(band_rows) == 1
    assert per_model_rows[0]["verdict"] == "insufficient_coverage"
    assert per_model_rows[0]["num_valid_runs"] == 1
    assert per_model_rows[0]["required_seed_count"] == 3
