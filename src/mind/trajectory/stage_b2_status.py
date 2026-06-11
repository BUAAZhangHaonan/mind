"""Stage B2 status helpers."""

from __future__ import annotations

from typing import Mapping


ALLOWED_STAGE_B2_VERDICT_PREFIX = "negative_budget_"
FORBIDDEN_STAGE_B2_VERDICTS = {
    "detector_winner",
    "detector_selected",
    "radius_ball_winner",
    "stage_c_winner",
}


def classifier_control_config() -> dict[str, object]:
    """Return the frozen Stage B2 classifier-control readout config."""

    return {
        "readout": "Diag-Classifier",
        "model": "logistic_regression",
        "role": "secondary_control",
        "uses_large_mlp": False,
        "primary_decision_signal": False,
    }


def validate_stage_b2_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate that a Stage B2 summary stays inside the negative-budget scope."""

    payload = dict(summary)
    if bool(payload.get("stage_c_started", False)):
        raise ValueError("Stage C must not be started by Stage B2")
    if bool(payload.get("detector_selected", False)):
        raise ValueError("Stage B2 must not select a detector")

    panel_models = [str(model) for model in payload.get("panel_models", [])]
    evaluated_models = [str(model) for model in payload.get("evaluated_models", [])]
    excluded = {
        str(model): str(reason)
        for model, reason in dict(payload.get("excluded_models", {}) or {}).items()
    }
    missing = sorted(set(panel_models) - set(evaluated_models) - set(excluded))
    if missing:
        raise ValueError(
            "missing Stage B2 panel model(s) without evaluation or exclusion: "
            + ", ".join(missing)
        )

    verdict = payload.get("verdict", {})
    if not isinstance(verdict, Mapping):
        raise ValueError("Stage B2 verdict must be an object")
    label = str(verdict.get("verdict", ""))
    if label in FORBIDDEN_STAGE_B2_VERDICTS or "detector" in label or "radius_ball" in label:
        raise ValueError("Stage B2 verdict must not use detector or Stage C language")
    if not label.startswith(ALLOWED_STAGE_B2_VERDICT_PREFIX):
        raise ValueError("Stage B2 verdict must use negative_budget_* language")

    payload["missing_models"] = missing
    payload["all_panel_models_accounted_for"] = not missing
    payload["excluded_models"] = excluded
    return payload
