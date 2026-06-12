from __future__ import annotations

import pytest

from .conftest import stage_c_attr


def test_stage_c_verdict_labels_are_frozen() -> None:
    status = stage_c_attr("stage_c_status", "stage_c_allowed_verdicts")()

    assert status["support_winners"] == ["single_vmf", "mixture_vmf", "radius_ball", "knn"]
    assert status["comparator_status"] == ["beats_supervised", "matches_supervised", "trails_supervised"]
    assert status["panel_verdicts"] == ["parametric_winner", "nonparametric_winner", "mixed_detector_panel"]


def test_stage_c_selects_support_winner_then_comparator_status() -> None:
    summarize = stage_c_attr("stage_c_status", "summarize_stage_c_detector_panel")

    summary = summarize(
        [
            _metric("m1", "single_vmf", 0.42, 0.81),
            _metric("m1", "mixture_vmf", 0.45, 0.82),
            _metric("m1", "radius_ball", 0.30, 0.70),
            _metric("m1", "knn", 0.31, 0.71),
            _metric("m1", "logistic", 0.44, 0.80),
            _metric("m2", "single_vmf", 0.40, 0.80),
            _metric("m2", "mixture_vmf", 0.41, 0.80),
            _metric("m2", "radius_ball", 0.28, 0.70),
            _metric("m2", "knn", 0.32, 0.72),
            _metric("m2", "logistic", 0.41, 0.79),
        ]
    )

    assert summary["support_winner"] == "mixture_vmf"
    assert summary["comparator_status"] == "matches_supervised"
    assert summary["panel_verdict"] == "parametric_winner"


def test_stage_c_summary_rejects_stage_d_started() -> None:
    validate = stage_c_attr("stage_c_status", "validate_stage_c_summary")

    with pytest.raises(ValueError, match="Stage D"):
        validate(
            {
                "stage": "stage_c",
                "stage_d_started": True,
                "panel_models": ["m1"],
                "evaluated_models": ["m1"],
                "excluded_models": {},
                "support_winner": "single_vmf",
                "comparator_status": "matches_supervised",
                "panel_verdict": "parametric_winner",
            }
        )


def test_stage_c_summary_rejects_non_glm_exclusions() -> None:
    validate = stage_c_attr("stage_c_status", "validate_stage_c_summary")

    with pytest.raises(ValueError, match="Only GLM"):
        validate(
            {
                "stage": "stage_c",
                "stage_d_started": False,
                "objective": "proxy_anchor",
                "negative_budget_ratio": 0.5,
                "negative_budget_seeds": [20260506, 20260507, 20260508],
                "panel_models": ["glm-4.6v-flash", "model-a"],
                "evaluated_models": [],
                "excluded_models": {
                    "glm-4.6v-flash": "answer format incompatible with frozen yes/no population rule",
                    "model-a": "RuntimeError: failed",
                },
                "failed_models": {},
                "skipped_models": {},
                "support_winner": "single_vmf",
                "comparator_status": "matches_supervised",
                "panel_verdict": "parametric_winner",
            }
        )


def test_stage_c_summary_allows_failed_and_skipped_panel_models_without_dropping_them() -> None:
    validate = stage_c_attr("stage_c_status", "validate_stage_c_summary")

    payload = validate(
        {
            "stage": "stage_c",
            "stage_d_started": False,
            "objective": "proxy_anchor",
            "negative_budget_ratio": 0.5,
            "negative_budget_seeds": [20260506, 20260507, 20260508],
            "panel_models": ["glm-4.6v-flash", "model-a", "model-b", "model-c"],
            "evaluated_models": ["model-a"],
            "excluded_models": {
                "glm-4.6v-flash": "answer format incompatible with frozen yes/no population rule",
            },
            "failed_models": {"model-b": "RuntimeError: failed"},
            "skipped_models": {"model-c": "not requested in this run"},
            "support_winner": "single_vmf",
            "comparator_status": "matches_supervised",
            "panel_verdict": "parametric_winner",
        }
    )

    assert payload["failed_models"] == {"model-b": "RuntimeError: failed"}
    assert payload["skipped_models"] == {"model-c": "not requested in this run"}


def _metric(model: str, method: str, pr_auc: float, roc_auc: float) -> dict[str, object]:
    return {
        "model_alias": model,
        "dataset_family": "repope",
        "method": method,
        "metric_split": "test",
        "metric_status": "passed",
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
