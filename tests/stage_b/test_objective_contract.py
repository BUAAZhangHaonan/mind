from __future__ import annotations

import pytest

from .conftest import ALLOWED_STAGE_B_OBJECTIVES, stage_b_attr


def test_stage_b_objective_surface_is_frozen_to_sphere_traj_lstm() -> None:
    allowed_objectives = stage_b_attr(
        "stage_b_objectives",
        "ALLOWED_STAGE_B_OBJECTIVES",
    )
    encoder_family = stage_b_attr(
        "stage_b_objectives",
        "STAGE_B_ENCODER_FAMILY",
    )
    validate_stage_b_objective_plan = stage_b_attr(
        "stage_b_objectives",
        "validate_stage_b_objective_plan",
    )

    assert tuple(allowed_objectives) == ALLOWED_STAGE_B_OBJECTIVES
    assert encoder_family == "Sphere-Traj-LSTM"

    plan = validate_stage_b_objective_plan(
        objectives=list(ALLOWED_STAGE_B_OBJECTIVES),
        encoder_family="Sphere-Traj-LSTM",
    )

    assert tuple(plan["objectives"]) == ALLOWED_STAGE_B_OBJECTIVES
    assert plan["encoder_family"] == "Sphere-Traj-LSTM"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "objectives": ["bce", "unsupported_objective"],
                "encoder_family": "Sphere-Traj-LSTM",
            },
            "objective|allowed",
        ),
        (
            {
                "objectives": ["bce"],
                "encoder_family": "Other-Encoder",
            },
            "Sphere-Traj-LSTM|encoder",
        ),
    ],
)
def test_stage_b_rejects_values_outside_frozen_objective_surface(
    kwargs: dict[str, object],
    match: str,
) -> None:
    validate_stage_b_objective_plan = stage_b_attr(
        "stage_b_objectives",
        "validate_stage_b_objective_plan",
    )

    with pytest.raises(ValueError, match=match):
        validate_stage_b_objective_plan(**kwargs)
