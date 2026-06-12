from __future__ import annotations

import pytest

from .conftest import (
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    STAGE_B4_GLM_EXCLUSION_REASON,
    stage_b4_attr,
)


def test_stage_b4_classifier_control_is_secondary_logistic_at_fixed_budget() -> None:
    classifier_control_config = stage_b4_attr(
        "stage_b4_status",
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


def test_stage_b4_classifier_control_status_accounts_for_glm_exclusion() -> None:
    summarize_classifier_control_status = stage_b4_attr(
        "stage_b4_status",
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
        excluded_models={GLM_MODEL_ALIAS: STAGE_B4_GLM_EXCLUSION_REASON},
    )

    status_by_model = {row["model_alias"]: row for row in rows}
    assert status_by_model[GLM_MODEL_ALIAS]["status"] == "excluded"
    assert status_by_model[GLM_MODEL_ALIAS]["reason"] == STAGE_B4_GLM_EXCLUSION_REASON
    assert all(
        status_by_model[model]["status"] == "evaluated"
        for model in PANEL_MODELS
        if model != GLM_MODEL_ALIAS
    )

    with pytest.raises(ValueError, match="missing"):
        summarize_classifier_control_status(
            metric_rows[:-1],
            panel_models=PANEL_MODELS,
            excluded_models={GLM_MODEL_ALIAS: STAGE_B4_GLM_EXCLUSION_REASON},
        )


def test_stage_b4_support_family_verdict_labels_are_exact_prompt_labels() -> None:
    build_summary = stage_b4_attr("stage_b4_status", "build_stage_b4_support_family_summary")
    panel_verdict = stage_b4_attr("stage_b4_status", "support_family_panel_verdict")

    rows = []
    for seed in [20260506, 20260507, 20260508]:
        rows.extend(
            [
                _support_row("param-model", seed, "nonparametric_knn", 0.70),
                _support_row("param-model", seed, "mixture_vmf", 0.75),
                _support_row("nonparam-model", seed, "nonparametric_knn", 0.80),
                _support_row("nonparam-model", seed, "single_vmf", 0.73),
                _support_row("mixed-model", seed, "nonparametric_knn", 0.80),
                _support_row("mixed-model", seed, "mixture_vmf", 0.795),
            ]
        )

    summary_rows = build_summary(
        rows,
        panel_models=["param-model", "nonparam-model", "mixed-model", "missing-model"],
        excluded_models={},
    )
    by_model = {row["model_alias"]: row for row in summary_rows}

    assert by_model["param-model"]["verdict"] == "parametric_preferred"
    assert by_model["param-model"]["selected_parametric_support_families"] == "mixture_vmf:3"
    assert by_model["nonparam-model"]["verdict"] == "nonparametric_preferred"
    assert by_model["nonparam-model"]["selected_parametric_support_families"] == "single_vmf:3"
    assert by_model["mixed-model"]["verdict"] == "mixed_support"
    assert by_model["missing-model"]["verdict"] == "insufficient_coverage"
    assert panel_verdict([by_model["param-model"]])["verdict"] == "parametric_support_preferred"
    assert panel_verdict([by_model["nonparam-model"]])["verdict"] == "nonparametric_support_preferred"
    assert panel_verdict([by_model["param-model"], by_model["nonparam-model"]])["verdict"] == "mixed_support_panel"


def test_stage_b4_summary_keeps_stage_c_and_detector_closed() -> None:
    validate_stage_b4_summary = stage_b4_attr("stage_b4_status", "validate_stage_b4_summary")
    base = {
        "stage": "stage_b4",
        "stage_c_started": False,
        "detector_selected": False,
        "objective": "proxy_anchor",
        "encoder_family": "Sphere-Traj-LSTM",
        "negative_budget_ratio": 0.5,
        "negative_budget_seeds": [20260506, 20260507, 20260508],
        "panel_models": list(PANEL_MODELS),
        "evaluated_models": [model for model in PANEL_MODELS if model != GLM_MODEL_ALIAS],
        "excluded_models": {GLM_MODEL_ALIAS: STAGE_B4_GLM_EXCLUSION_REASON},
        "verdict": {"verdict": "mixed_support_panel"},
    }

    summary = validate_stage_b4_summary(base)

    assert summary["all_panel_models_accounted_for"] is True
    assert summary["missing_models"] == []
    assert summary["stage_c_started"] is False
    assert summary["detector_selected"] is False

    with pytest.raises(ValueError, match="Stage C"):
        validate_stage_b4_summary({**base, "stage_c_started": True})
    with pytest.raises(ValueError, match="detector"):
        validate_stage_b4_summary({**base, "detector_selected": True})
    with pytest.raises(ValueError, match="panel verdict"):
        validate_stage_b4_summary({**base, "verdict": {"verdict": "scale_stable_panel"}})


def _support_row(model_alias: str, seed: int, support_family: str, pr_auc: float) -> dict[str, object]:
    return {
        "model_alias": model_alias,
        "dataset_family": "repope",
        "metric_split": "test",
        "metric_status": "passed",
        "negative_budget_seed": seed,
        "support_family": support_family,
        "readout": support_family,
        "pr_auc": pr_auc,
        "roc_auc": pr_auc,
    }
