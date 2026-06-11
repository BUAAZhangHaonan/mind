from __future__ import annotations

import pytest

from .conftest import GLM_MODEL_ALIAS, PANEL_MODELS, stage_b2_attr


def test_stage_b2_summary_accounts_for_models_and_does_not_start_stage_c() -> None:
    validate_stage_b2_summary = stage_b2_attr(
        "stage_b2_status",
        "validate_stage_b2_summary",
    )

    summary = validate_stage_b2_summary(
        {
            "stage": "stage_b2",
            "stage_c_started": False,
            "detector_selected": False,
            "panel_models": list(PANEL_MODELS),
            "evaluated_models": [model for model in PANEL_MODELS if model != GLM_MODEL_ALIAS],
            "excluded_models": {GLM_MODEL_ALIAS: "GLM answer_text is not parseable"},
            "verdict": {
                "verdict": "negative_budget_stable_to_25pct",
                "material_degradation_below_ratio": 0.25,
            },
        }
    )

    assert summary["all_panel_models_accounted_for"] is True
    assert summary["stage_c_started"] is False
    assert summary["verdict"]["verdict"].startswith("negative_budget_")


def test_stage_b2_summary_rejects_detector_language_and_missing_models() -> None:
    validate_stage_b2_summary = stage_b2_attr(
        "stage_b2_status",
        "validate_stage_b2_summary",
    )

    with pytest.raises(ValueError, match="Stage C"):
        validate_stage_b2_summary(
            {
                "stage": "stage_b2",
                "stage_c_started": True,
                "panel_models": list(PANEL_MODELS),
                "evaluated_models": list(PANEL_MODELS),
                "excluded_models": {},
                "verdict": {"verdict": "detector_winner"},
            }
        )

    with pytest.raises(ValueError, match="missing"):
        validate_stage_b2_summary(
            {
                "stage": "stage_b2",
                "stage_c_started": False,
                "panel_models": list(PANEL_MODELS),
                "evaluated_models": list(PANEL_MODELS[:-1]),
                "excluded_models": {},
                "verdict": {"verdict": "negative_budget_inconclusive"},
            }
        )
