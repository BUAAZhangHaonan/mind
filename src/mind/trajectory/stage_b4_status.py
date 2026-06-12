"""Stage B4 status, support-family verdict, and classifier-control helpers."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .stage_b_objectives import STAGE_B_ENCODER_FAMILY
from .stage_b4_manifest import (
    REQUIRED_STAGE_B4_RATIO,
    REQUIRED_STAGE_B4_SEEDS,
    STAGE_B4_OBJECTIVE,
)


PARAMETRIC_SUPPORT_FAMILIES = frozenset({"single_vmf", "mixture_vmf", "parametric_vmf"})
NONPARAMETRIC_SUPPORT_FAMILIES = frozenset({"nonparametric_knn", "knn"})
ALLOWED_STAGE_B4_PER_MODEL_VERDICTS = (
    "parametric_preferred",
    "nonparametric_preferred",
    "mixed_support",
    "insufficient_coverage",
)
ALLOWED_STAGE_B4_PANEL_VERDICTS = (
    "parametric_support_preferred",
    "mixed_support_panel",
    "nonparametric_support_preferred",
)
FORBIDDEN_STAGE_B4_VERDICT_TERMS = (
    "detector",
    "winner",
    "radius_ball",
    "stage_c",
)


def classifier_control_config() -> dict[str, object]:
    """Return the frozen Stage B4 classifier-control readout config."""

    return {
        "readout": "Diag-Classifier",
        "model": "logistic_regression",
        "role": "secondary_control",
        "objective": STAGE_B4_OBJECTIVE,
        "negative_budget_ratio": REQUIRED_STAGE_B4_RATIO,
        "negative_budget_seeds": list(REQUIRED_STAGE_B4_SEEDS),
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
            and round(float(row.get("negative_budget_ratio", REQUIRED_STAGE_B4_RATIO)), 6)
            == REQUIRED_STAGE_B4_RATIO
            and (row.get("model_alias") or row.get("model_name"))
        }
    )
    missing = sorted(set(panel) - set(evaluated) - set(excluded))
    if missing:
        raise ValueError(
            "missing Stage B4 classifier-control panel model(s): "
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


def build_stage_b4_support_family_summary(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    panel_models: Sequence[str],
    excluded_models: Mapping[str, object] | None = None,
    required_seed_count: int = len(REQUIRED_STAGE_B4_SEEDS),
    pr_auc_margin: float = 0.02,
) -> list[dict[str, object]]:
    """Compare parametric vMF support against nonparametric kNN support per model."""

    rows = [dict(row) for row in metric_rows]
    panel = [str(model) for model in panel_models]
    excluded = {str(model): str(reason) for model, reason in dict(excluded_models or {}).items()}
    output: list[dict[str, object]] = []
    for model in panel:
        if model in excluded:
            output.append(
                _insufficient_support_row(
                    model,
                    status="excluded",
                    reason=excluded[model],
                    required_seed_count=required_seed_count,
                    pr_auc_margin=pr_auc_margin,
                )
            )
            continue
        model_rows = [
            row
            for row in rows
            if str(row.get("model_alias") or row.get("model_name")) == model
            and str(row.get("dataset_family", "")).lower() == "repope"
            and str(row.get("metric_split") or row.get("split") or row.get("eval_split")) == "test"
            and str(row.get("metric_status", "passed")) == "passed"
        ]
        per_seed: dict[int, dict[str, tuple[float, str]]] = {}
        for row in model_rows:
            score = _finite_float(row.get("pr_auc"))
            if score is None:
                continue
            seed = int(float(row.get("negative_budget_seed", row.get("seed", 0))))
            family = _support_family(row)
            if family in PARAMETRIC_SUPPORT_FAMILIES:
                _keep_best(per_seed.setdefault(seed, {}), "parametric", score, family)
            elif family in NONPARAMETRIC_SUPPORT_FAMILIES:
                _keep_best(per_seed.setdefault(seed, {}), "nonparametric", score, family)

        complete = {
            seed: values
            for seed, values in per_seed.items()
            if "parametric" in values and "nonparametric" in values
        }
        if len(complete) < int(required_seed_count):
            output.append(
                _insufficient_support_row(
                    model,
                    status="incomplete",
                    reason="fewer valid paired support-family runs than required",
                    required_seed_count=required_seed_count,
                    pr_auc_margin=pr_auc_margin,
                    num_valid_runs=len(complete),
                    seed_values=sorted(complete),
                )
            )
            continue

        parametric_scores = [values["parametric"][0] for _, values in sorted(complete.items())]
        nonparametric_scores = [values["nonparametric"][0] for _, values in sorted(complete.items())]
        parametric_median = float(np.median(np.asarray(parametric_scores, dtype=np.float64)))
        nonparametric_median = float(np.median(np.asarray(nonparametric_scores, dtype=np.float64)))
        delta = parametric_median - nonparametric_median
        if delta >= float(pr_auc_margin):
            verdict = "parametric_preferred"
        elif -delta >= float(pr_auc_margin):
            verdict = "nonparametric_preferred"
        else:
            verdict = "mixed_support"

        selected_parametric_families = Counter(values["parametric"][1] for values in complete.values())
        selected_parametric_family_summary = ";".join(
            f"{family}:{count}" for family, count in sorted(selected_parametric_families.items())
        )
        output.append(
            {
                "model_alias": model,
                "verdict": verdict,
                "status": "evaluated",
                "median_parametric_pr_auc": parametric_median,
                "median_nonparametric_pr_auc": nonparametric_median,
                "median_pr_auc_delta_parametric_minus_nonparametric": delta,
                "num_valid_runs": len(complete),
                "required_seed_count": int(required_seed_count),
                "pr_auc_margin": float(pr_auc_margin),
                "valid_seeds": ";".join(str(seed) for seed in sorted(complete)),
                "selected_parametric_support_families": selected_parametric_family_summary,
                "best_parametric_support_families": selected_parametric_family_summary,
                "reason": "",
            }
        )
    return output


def support_family_panel_verdict(per_model_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize per-model support-family rows without choosing a detector."""

    labels = [str(row.get("verdict", "")) for row in per_model_rows]
    parametric_count = sum(label == "parametric_preferred" for label in labels)
    nonparametric_count = sum(label == "nonparametric_preferred" for label in labels)
    mixed_count = sum(label == "mixed_support" for label in labels)
    insufficient_count = sum(label == "insufficient_coverage" for label in labels)
    evaluable_total = parametric_count + nonparametric_count + mixed_count
    if evaluable_total > 0 and parametric_count > evaluable_total / 2.0:
        label = "parametric_support_preferred"
    elif evaluable_total > 0 and nonparametric_count > evaluable_total / 2.0:
        label = "nonparametric_support_preferred"
    else:
        label = "mixed_support_panel"
    return {
        "verdict": label,
        "models_parametric_preferred": parametric_count,
        "models_nonparametric_preferred": nonparametric_count,
        "models_mixed_support": mixed_count,
        "models_insufficient_coverage": insufficient_count,
        "models_total": len(labels),
    }


def validate_stage_b4_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate that Stage B4 stays inside support-family analysis scope."""

    payload = dict(summary)
    if bool(payload.get("stage_c_started", False)):
        raise ValueError("Stage C must not be started by Stage B4")
    if bool(payload.get("detector_selected", False)):
        raise ValueError("Stage B4 must not select a detector")
    if str(payload.get("objective", "")) != STAGE_B4_OBJECTIVE:
        raise ValueError("Stage B4 objective must be fixed to proxy_anchor")
    if str(payload.get("encoder_family", "")) != STAGE_B_ENCODER_FAMILY:
        raise ValueError(f"Stage B4 encoder must be {STAGE_B_ENCODER_FAMILY}")
    if round(float(payload.get("negative_budget_ratio", -1.0)), 6) != REQUIRED_STAGE_B4_RATIO:
        raise ValueError("Stage B4 negative-budget ratio must be fixed to 0.5")
    seeds = [int(seed) for seed in payload.get("negative_budget_seeds", [])]
    if tuple(seeds) != REQUIRED_STAGE_B4_SEEDS:
        raise ValueError("Stage B4 seeds are not the frozen required seeds")

    panel_models = [str(model) for model in payload.get("panel_models", [])]
    evaluated_models = [str(model) for model in payload.get("evaluated_models", [])]
    excluded = {
        str(model): str(reason)
        for model, reason in dict(payload.get("excluded_models", {}) or {}).items()
    }
    missing = sorted(set(panel_models) - set(evaluated_models) - set(excluded))
    if missing:
        raise ValueError(
            "missing Stage B4 panel model(s) without evaluation or exclusion: "
            + ", ".join(missing)
        )

    verdict = payload.get("verdict", {})
    if not isinstance(verdict, Mapping):
        raise ValueError("Stage B4 verdict must be an object")
    label = str(verdict.get("verdict", ""))
    if any(term in label for term in FORBIDDEN_STAGE_B4_VERDICT_TERMS):
        raise ValueError("Stage B4 verdict must not use detector, winner, or Stage C language")
    if label not in ALLOWED_STAGE_B4_PANEL_VERDICTS:
        raise ValueError(
            "Stage B4 panel verdict must be one of: "
            + ", ".join(ALLOWED_STAGE_B4_PANEL_VERDICTS)
        )

    payload["negative_budget_seeds"] = seeds
    payload["missing_models"] = missing
    payload["all_panel_models_accounted_for"] = not missing
    payload["excluded_models"] = excluded
    return payload


def _support_family(row: Mapping[str, object]) -> str:
    value = str(row.get("support_family", "")).strip()
    if value:
        return value
    readout = str(row.get("readout", "")).lower()
    if "knn" in readout:
        return "nonparametric_knn"
    if "vmf" in readout:
        return "parametric_vmf"
    return ""


def _keep_best(target: dict[str, tuple[float, str]], key: str, score: float, family: str) -> None:
    existing = target.get(key)
    if existing is None or float(score) > float(existing[0]):
        target[key] = (float(score), str(family))


def _insufficient_support_row(
    model: str,
    *,
    status: str,
    reason: str,
    required_seed_count: int,
    pr_auc_margin: float,
    num_valid_runs: int = 0,
    seed_values: Sequence[int] = (),
) -> dict[str, object]:
    return {
        "model_alias": model,
        "verdict": "insufficient_coverage",
        "status": status,
        "median_parametric_pr_auc": "",
        "median_nonparametric_pr_auc": "",
        "median_pr_auc_delta_parametric_minus_nonparametric": "",
        "num_valid_runs": int(num_valid_runs),
        "required_seed_count": int(required_seed_count),
        "pr_auc_margin": float(pr_auc_margin),
        "valid_seeds": ";".join(str(seed) for seed in seed_values),
        "selected_parametric_support_families": "",
        "best_parametric_support_families": "",
        "reason": reason,
    }


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


__all__ = [
    "ALLOWED_STAGE_B4_PANEL_VERDICTS",
    "ALLOWED_STAGE_B4_PER_MODEL_VERDICTS",
    "classifier_control_config",
    "build_stage_b4_support_family_summary",
    "summarize_classifier_control_status",
    "support_family_panel_verdict",
    "validate_stage_b4_summary",
]
