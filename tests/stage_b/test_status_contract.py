from __future__ import annotations

import pytest

from .conftest import (
    ALLOWED_STAGE_B_OBJECTIVES,
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    stage_b_attr,
    stage_b_metric_row,
)


def test_status_lists_every_panel_model_or_explicit_excluded_reason() -> None:
    summarize_stage_b_status = stage_b_attr(
        "stage_b_status",
        "summarize_stage_b_status",
    )
    metric_rows = [
        stage_b_metric_row(alias, "bce", pr_auc=0.60)
        for alias in PANEL_MODELS
        if alias != GLM_MODEL_ALIAS
    ]

    summary = summarize_stage_b_status(
        panel_models=PANEL_MODELS,
        metric_rows=metric_rows,
        excluded_models={
            GLM_MODEL_ALIAS: "excluded because GLM answer_text is not parseable",
        },
    )

    assert set(summary["model_status"]) == set(PANEL_MODELS)
    assert summary["model_status"][GLM_MODEL_ALIAS]["status"] == "excluded"
    assert summary["model_status"][GLM_MODEL_ALIAS]["reason"]
    assert summary["stage_c_started"] is False


def test_status_rejects_missing_panel_model_without_excluded_reason() -> None:
    summarize_stage_b_status = stage_b_attr(
        "stage_b_status",
        "summarize_stage_b_status",
    )
    metric_rows = [
        stage_b_metric_row(alias, "bce", pr_auc=0.60)
        for alias in PANEL_MODELS[:-1]
    ]

    with pytest.raises(ValueError, match="excluded reason|panel model"):
        summarize_stage_b_status(
            panel_models=PANEL_MODELS,
            metric_rows=metric_rows,
            excluded_models={},
        )


def test_stage_b_verdict_labels_are_winner_tie_or_inconclusive_only() -> None:
    decide_stage_b_verdict = stage_b_attr(
        "stage_b_status",
        "decide_stage_b_verdict",
    )
    winner_rows = [
        stage_b_metric_row(PANEL_MODELS[0], "bce", pr_auc=0.70, roc_auc=0.80),
        stage_b_metric_row(PANEL_MODELS[0], "supcon", pr_auc=0.74, roc_auc=0.78),
        stage_b_metric_row(PANEL_MODELS[0], "proxy_anchor", pr_auc=0.72, roc_auc=0.82),
    ]
    tie_rows = [
        stage_b_metric_row(PANEL_MODELS[0], "bce", pr_auc=0.74, roc_auc=0.80),
        stage_b_metric_row(PANEL_MODELS[0], "supcon", pr_auc=0.74, roc_auc=0.79),
        stage_b_metric_row(PANEL_MODELS[0], "proxy_anchor", pr_auc=0.70, roc_auc=0.82),
    ]

    winner = decide_stage_b_verdict(winner_rows, objectives=ALLOWED_STAGE_B_OBJECTIVES)
    tie = decide_stage_b_verdict(tie_rows, objectives=ALLOWED_STAGE_B_OBJECTIVES)
    inconclusive = decide_stage_b_verdict([], objectives=ALLOWED_STAGE_B_OBJECTIVES)

    assert winner["verdict"] == "winner"
    assert winner["winner"] == "supcon"
    assert tie["verdict"] == "tie"
    assert set(tie["winners"]) == {"bce", "supcon"}
    assert inconclusive["verdict"] == "inconclusive"
    assert {winner["verdict"], tie["verdict"], inconclusive["verdict"]} <= {
        "winner",
        "tie",
        "inconclusive",
    }
