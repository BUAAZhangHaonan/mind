from __future__ import annotations

import pytest

from .conftest import GLM_MODEL_ALIAS, PANEL_MODELS, stage_b2_attr
from .conftest import stage_b2_script_attr


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


def test_negative_budget_verdict_requires_complete_model_seed_coverage() -> None:
    negative_budget_verdict = stage_b2_script_attr(
        "stage_b2_run",
        "_negative_budget_verdict",
    )
    rows = []
    for ratio in (1.0, 0.5, 0.1):
        for seed in (20260506, 20260507, 20260508):
            models = ("model-a", "model-b")
            if ratio == 0.1:
                models = ("model-a",)
            for model in models:
                rows.append(
                    {
                        "model_alias": model,
                        "dataset_family": "repope",
                        "readout": "Diag-kNN-tuned",
                        "metric_status": "passed",
                        "negative_budget_ratio": ratio,
                        "negative_budget_seed": seed,
                        "pr_auc": 0.9 if ratio == 0.1 else 0.8,
                    }
                )

    verdict = negative_budget_verdict(rows)

    assert verdict["verdict"] == "negative_budget_stable_to_50pct"
    assert verdict["per_ratio_complete_model_count"]["0.1"] == 1
    assert verdict["baseline_complete_model_count"] == 2


def test_per_model_summary_keeps_excluded_models_not_in_selected_subset() -> None:
    per_model_summary = stage_b2_script_attr(
        "stage_b2_run",
        "_per_model_negative_budget_summary",
    )

    rows = per_model_summary(
        [],
        panel_models=["model-a"],
        excluded_models={GLM_MODEL_ALIAS: "not parseable"},
        model_failures={},
    )

    assert {"model_alias": GLM_MODEL_ALIAS, "status": "excluded", "reason": "not parseable"} in rows
