from __future__ import annotations

import numpy as np
import pytest

from .conftest import stage_b2_attr


def test_stage_b2_negative_budget_plan_is_frozen() -> None:
    validate_stage_b2_budget_plan = stage_b2_attr(
        "stage_b2_budget",
        "validate_stage_b2_budget_plan",
    )

    plan = validate_stage_b2_budget_plan(
        ratios=[1.0, 0.5, 0.25, 0.10],
        seeds=[20260506, 20260507, 20260508],
        objective="proxy_anchor",
        encoder_family="Sphere-Traj-LSTM",
    )

    assert plan["ratios"] == [1.0, 0.5, 0.25, 0.1]
    assert plan["seeds"] == [20260506, 20260507, 20260508]
    assert plan["objective"] == "proxy_anchor"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "ratios": [1.0, 0.33],
                "seeds": [20260506, 20260507, 20260508],
                "objective": "proxy_anchor",
                "encoder_family": "Sphere-Traj-LSTM",
            },
            "ratio",
        ),
        (
            {
                "ratios": [1.0],
                "seeds": [1],
                "objective": "proxy_anchor",
                "encoder_family": "Sphere-Traj-LSTM",
            },
            "seed",
        ),
        (
            {
                "ratios": [1.0],
                "seeds": [20260506, 20260507, 20260508],
                "objective": "bce",
                "encoder_family": "Sphere-Traj-LSTM",
            },
            "Proxy Anchor|proxy_anchor",
        ),
    ],
)
def test_stage_b2_rejects_unfrozen_budget_plan(kwargs: dict[str, object], match: str) -> None:
    validate_stage_b2_budget_plan = stage_b2_attr(
        "stage_b2_budget",
        "validate_stage_b2_budget_plan",
    )

    with pytest.raises(ValueError, match=match):
        validate_stage_b2_budget_plan(**kwargs)


def test_subsample_hard_negatives_without_touching_correct_rows() -> None:
    subsample_stage_b2_training_indices = stage_b2_attr(
        "stage_b2_budget",
        "subsample_stage_b2_training_indices",
    )
    labels = np.asarray([0] * 5 + [1] * 8, dtype=np.int64)

    selected = subsample_stage_b2_training_indices(labels, ratio=0.5, seed=20260506)

    assert set(range(5)).issubset(set(selected.tolist()))
    assert len([idx for idx in selected if labels[idx] == 1]) == 4
    assert len(set(selected.tolist())) == len(selected)


def test_too_small_negative_budget_is_skipped_with_reason() -> None:
    subsample_stage_b2_training_indices = stage_b2_attr(
        "stage_b2_budget",
        "subsample_stage_b2_training_indices",
    )
    labels = np.asarray([0] * 100 + [1] * 100, dtype=np.int64)

    with pytest.raises(ValueError, match="fewer than 20 hard negatives"):
        subsample_stage_b2_training_indices(labels, ratio=0.10, seed=20260506)
