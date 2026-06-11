"""Stage B3 status validation helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

from .stage_b3_manifest import (
    REQUIRED_STAGE_B3_RATIO,
    REQUIRED_STAGE_B3_SEEDS,
    STAGE_B3_OBJECTIVE,
)


FORBIDDEN_STAGE_B3_VERDICT_TERMS = (
    "detector",
    "winner",
    "objective",
    "radius_ball",
    "stage_c",
)
ALLOWED_STAGE_B3_PANEL_VERDICTS = (
    "scale_stable_panel",
    "scale_mixed_panel",
    "scale_sensitive_panel",
)


def classifier_control_config() -> dict[str, object]:
    """Return the frozen Stage B3 classifier-control readout config."""

    return {
        "readout": "Diag-Classifier",
        "model": "logistic_regression",
        "role": "secondary_control",
        "objective": STAGE_B3_OBJECTIVE,
        "negative_budget_ratio": REQUIRED_STAGE_B3_RATIO,
        "negative_budget_seeds": list(REQUIRED_STAGE_B3_SEEDS),
        "uses_large_mlp": False,
        "primary_decision_signal": False,
    }


def summarize_classifier_control_status(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    panel_models: Sequence[str],
    excluded_models: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build classifier-control status rows and reject hidden missing panel models."""

    panel = [str(model) for model in panel_models]
    excluded = {str(model): str(reason) for model, reason in dict(excluded_models or {}).items()}
    evaluated = sorted(
        {
            str(row.get("model_alias") or row.get("model_name"))
            for row in metric_rows
            if str(row.get("readout", "")) == "Diag-Classifier"
            and round(float(row.get("negative_budget_ratio", REQUIRED_STAGE_B3_RATIO)), 6)
            == REQUIRED_STAGE_B3_RATIO
            and (row.get("model_alias") or row.get("model_name"))
        }
    )
    missing = sorted(set(panel) - set(evaluated) - set(excluded))
    if missing:
        raise ValueError(
            "missing Stage B3 classifier-control panel model(s): "
            + ", ".join(missing)
        )

    rows: list[dict[str, object]] = []
    for model in panel:
        if model in excluded:
            status = "excluded"
            reason = excluded[model]
        elif model in evaluated:
            status = "evaluated"
            reason = ""
        else:
            status = "missing"
            reason = "no classifier-control metric row"
        row = classifier_control_config()
        row.update(
            {
                "model_alias": model,
                "status": status,
                "reason": reason,
            }
        )
        rows.append(row)
    return rows


def validate_stage_b3_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate that Stage B3 stays inside the scale-stability scope."""

    payload = dict(summary)
    if bool(payload.get("stage_c_started", False)):
        raise ValueError("Stage C must not be started by Stage B3")
    if bool(payload.get("detector_selected", False)):
        raise ValueError("Stage B3 must not select a detector")
    if "objective_comparison" in payload:
        raise ValueError("Stage B3 must not include objective comparison output")
    if str(payload.get("objective", "")) != STAGE_B3_OBJECTIVE:
        raise ValueError("Stage B3 objective must be fixed to proxy_anchor")
    if round(float(payload.get("negative_budget_ratio", -1.0)), 6) != REQUIRED_STAGE_B3_RATIO:
        raise ValueError("Stage B3 negative-budget ratio must be fixed to 0.5")
    seeds = [int(seed) for seed in payload.get("negative_budget_seeds", [])]
    if tuple(seeds) != REQUIRED_STAGE_B3_SEEDS:
        raise ValueError("Stage B3 seeds are not the frozen required seeds")

    panel_models = [str(model) for model in payload.get("panel_models", [])]
    evaluated_models = [str(model) for model in payload.get("evaluated_models", [])]
    excluded = {
        str(model): str(reason)
        for model, reason in dict(payload.get("excluded_models", {}) or {}).items()
    }
    missing = sorted(set(panel_models) - set(evaluated_models) - set(excluded))
    if missing:
        raise ValueError(
            "missing Stage B3 panel model(s) without evaluation or exclusion: "
            + ", ".join(missing)
        )

    verdict = payload.get("verdict", {})
    if not isinstance(verdict, Mapping):
        raise ValueError("Stage B3 verdict must be an object")
    label = str(verdict.get("verdict", ""))
    if any(term in label for term in FORBIDDEN_STAGE_B3_VERDICT_TERMS):
        raise ValueError("Stage B3 verdict must not use detector, winner, or objective language")
    if label not in ALLOWED_STAGE_B3_PANEL_VERDICTS:
        raise ValueError(
            "Stage B3 panel verdict must be one of: "
            + ", ".join(ALLOWED_STAGE_B3_PANEL_VERDICTS)
        )

    payload["negative_budget_seeds"] = seeds
    payload["missing_models"] = missing
    payload["all_panel_models_accounted_for"] = not missing
    payload["excluded_models"] = excluded
    return payload


def scale_stability_verdict(per_model_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize per-model kNN scale-stability rows without choosing a detector."""

    rows = [dict(row) for row in per_model_rows]
    if not rows:
        return {
            "verdict": "scale_mixed_panel",
            "reason": "no per-model stability rows were available",
            "stability_drop_tolerance": 0.02,
        }
    labels = [str(row.get("verdict", "")) for row in rows]
    stable_count = sum(label == "scale_stable" for label in labels)
    sensitive_count = sum(label == "scale_sensitive" for label in labels)
    insufficient_count = sum(label == "insufficient_coverage" for label in labels)
    evaluable_total = stable_count + sensitive_count
    if evaluable_total > 0 and stable_count > evaluable_total / 2.0:
        panel_label = "scale_stable_panel"
    elif evaluable_total > 0 and sensitive_count > evaluable_total / 2.0:
        panel_label = "scale_sensitive_panel"
    else:
        panel_label = "scale_mixed_panel"
    return {
        "verdict": panel_label,
        "models_scale_stable": stable_count,
        "models_scale_sensitive": sensitive_count,
        "models_insufficient_coverage": insufficient_count,
        "models_total": len(rows),
        "stability_drop_tolerance": 0.02,
    }


__all__ = [
    "ALLOWED_STAGE_B3_PANEL_VERDICTS",
    "classifier_control_config",
    "scale_stability_verdict",
    "summarize_classifier_control_status",
    "validate_stage_b3_summary",
]
