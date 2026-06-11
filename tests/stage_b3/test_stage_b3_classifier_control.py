from __future__ import annotations

import pytest

from .conftest import GLM_MODEL_ALIAS, PANEL_MODELS, STAGE_B3_GLM_EXCLUSION_REASON, stage_b3_attr


def test_stage_b3_classifier_control_is_secondary_logistic_at_fixed_budget() -> None:
    classifier_control_config = stage_b3_attr(
        "stage_b3_status",
        "classifier_control_config",
    )

    config = classifier_control_config()

    assert config["readout"] == "Diag-Classifier"
    assert config["model"] == "logistic_regression"
    assert config["role"] == "secondary_control"
    assert config["objective"] == "proxy_anchor"
    assert config["negative_budget_ratio"] == 0.5
    assert config["negative_budget_seeds"] == [20260506, 20260507, 20260508]
    assert config["primary_decision_signal"] is False
    assert config["uses_large_mlp"] is False


def test_stage_b3_classifier_control_status_accounts_for_glm_exclusion() -> None:
    summarize_classifier_control_status = stage_b3_attr(
        "stage_b3_status",
        "summarize_classifier_control_status",
    )
    metric_rows = [
        {
            "model_alias": model,
            "readout": "Diag-Classifier",
            "metric_status": "passed",
            "negative_budget_ratio": 0.5,
        }
        for model in PANEL_MODELS
        if model != GLM_MODEL_ALIAS
    ]

    rows = summarize_classifier_control_status(
        metric_rows,
        panel_models=PANEL_MODELS,
        excluded_models={GLM_MODEL_ALIAS: STAGE_B3_GLM_EXCLUSION_REASON},
    )

    status_by_model = {row["model_alias"]: row for row in rows}
    assert status_by_model[GLM_MODEL_ALIAS]["status"] == "excluded"
    assert status_by_model[GLM_MODEL_ALIAS]["reason"] == STAGE_B3_GLM_EXCLUSION_REASON
    assert all(
        status_by_model[model]["status"] == "evaluated"
        for model in PANEL_MODELS
        if model != GLM_MODEL_ALIAS
    )

    with pytest.raises(ValueError, match="missing"):
        summarize_classifier_control_status(
            metric_rows[:-1],
            panel_models=PANEL_MODELS,
            excluded_models={GLM_MODEL_ALIAS: STAGE_B3_GLM_EXCLUSION_REASON},
        )
