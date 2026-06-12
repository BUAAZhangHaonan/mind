"""Stage C detector-family summary and verdict helpers."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .stage_c_manifest import (
    REQUIRED_STAGE_C_RATIO,
    REQUIRED_STAGE_C_SEEDS,
    STAGE_C_GLM_EXCLUSION_REASON,
    STAGE_C_OBJECTIVE,
)
from .stage_c_support import STAGE_C_METHODS, STAGE_C_SUPPORT_METHODS


STAGE_C_SUPPORT_WINNERS = ("single_vmf", "mixture_vmf", "radius_ball", "knn")
STAGE_C_COMPARATOR_STATUSES = ("beats_supervised", "matches_supervised", "trails_supervised")
STAGE_C_PANEL_VERDICTS = ("parametric_winner", "nonparametric_winner", "mixed_detector_panel")
PARAMETRIC_STAGE_C_METHODS = frozenset({"single_vmf", "mixture_vmf"})
NONPARAMETRIC_STAGE_C_METHODS = frozenset({"radius_ball", "knn"})


def stage_c_allowed_verdicts() -> dict[str, list[str]]:
    """Return the frozen Stage C verdict vocabularies."""

    return {
        "support_winners": list(STAGE_C_SUPPORT_WINNERS),
        "comparator_status": list(STAGE_C_COMPARATOR_STATUSES),
        "panel_verdicts": list(STAGE_C_PANEL_VERDICTS),
    }


def summarize_stage_c_detector_panel(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select the Stage C support winner and supervised-comparator status."""

    rows = [
        dict(row)
        for row in metric_rows
        if str(row.get("dataset_family", "")).lower() == "repope"
        and str(row.get("metric_split") or row.get("split") or row.get("eval_split")) == "test"
        and str(row.get("metric_status", "passed")) == "passed"
        and str(row.get("method", "")) in STAGE_C_METHODS
    ]
    support_stats = _method_stats(rows, methods=STAGE_C_SUPPORT_METHODS)
    if not support_stats:
        raise ValueError("Stage C support winner requires RePOPE test rows")
    support_winner = sorted(
        support_stats,
        key=lambda method: (
            -float(support_stats[method]["mean_pr_auc"]),
            -float(support_stats[method]["mean_roc_auc"]),
            -int(support_stats[method]["win_count"]),
            _simplicity_rank(method),
        ),
    )[0]
    logistic_stats = _method_stats(rows, methods=("logistic",))
    support_pr = float(support_stats[support_winner]["mean_pr_auc"])
    logistic_pr = float(logistic_stats.get("logistic", {}).get("mean_pr_auc", float("nan")))
    if np.isfinite(logistic_pr):
        delta = support_pr - logistic_pr
        if delta >= 0.01:
            comparator_status = "beats_supervised"
        elif delta <= -0.01:
            comparator_status = "trails_supervised"
        else:
            comparator_status = "matches_supervised"
    else:
        delta = float("nan")
        comparator_status = "trails_supervised"
    panel_verdict = (
        "parametric_winner"
        if support_winner in PARAMETRIC_STAGE_C_METHODS
        else "nonparametric_winner"
        if support_winner in NONPARAMETRIC_STAGE_C_METHODS
        else "mixed_detector_panel"
    )
    return {
        "support_winner": support_winner,
        "support_winner_mean_pr_auc": support_pr,
        "support_winner_mean_roc_auc": float(support_stats[support_winner]["mean_roc_auc"]),
        "logistic_mean_pr_auc": logistic_pr,
        "support_minus_logistic_pr_auc": delta,
        "comparator_status": comparator_status,
        "panel_verdict": panel_verdict,
        "method_stats": support_stats | logistic_stats,
    }


def build_stage_c_per_model_summary(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    panel_models: Sequence[str],
    excluded_models: Mapping[str, object] | None = None,
    failed_models: Mapping[str, object] | None = None,
    skipped_models: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build one Stage C detector summary row per panel model."""

    excluded = {str(model): str(reason) for model, reason in dict(excluded_models or {}).items()}
    failed = {str(model): str(reason) for model, reason in dict(failed_models or {}).items()}
    skipped = {str(model): str(reason) for model, reason in dict(skipped_models or {}).items()}
    output: list[dict[str, object]] = []
    rows = [dict(row) for row in metric_rows]
    for model in [str(value) for value in panel_models]:
        if model in excluded:
            output.append(
                {
                    "model_alias": model,
                    "status": "excluded",
                    "support_winner": "insufficient_coverage",
                    "comparator_status": "insufficient_coverage",
                    "panel_verdict": "insufficient_coverage",
                    "reason": excluded[model],
                }
            )
            continue
        if model in failed:
            output.append(
                {
                    "model_alias": model,
                    "status": "failed",
                    "support_winner": "insufficient_coverage",
                    "comparator_status": "insufficient_coverage",
                    "panel_verdict": "insufficient_coverage",
                    "reason": failed[model],
                }
            )
            continue
        if model in skipped:
            output.append(
                {
                    "model_alias": model,
                    "status": "skipped",
                    "support_winner": "insufficient_coverage",
                    "comparator_status": "insufficient_coverage",
                    "panel_verdict": "insufficient_coverage",
                    "reason": skipped[model],
                }
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
        if not model_rows:
            output.append(
                {
                    "model_alias": model,
                    "status": "incomplete",
                    "support_winner": "insufficient_coverage",
                    "comparator_status": "insufficient_coverage",
                    "panel_verdict": "insufficient_coverage",
                    "reason": "no valid RePOPE test metric rows",
                }
            )
            continue
        summary = summarize_stage_c_detector_panel(model_rows)
        output.append(
            {
                "model_alias": model,
                "status": "evaluated",
                "support_winner": summary["support_winner"],
                "comparator_status": summary["comparator_status"],
                "panel_verdict": summary["panel_verdict"],
                "support_winner_mean_pr_auc": summary["support_winner_mean_pr_auc"],
                "logistic_mean_pr_auc": summary["logistic_mean_pr_auc"],
                "reason": "",
            }
        )
    return output


def validate_stage_c_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate Stage C summary scope and panel coverage."""

    payload = dict(summary)
    if bool(payload.get("stage_d_started", False)):
        raise ValueError("Stage D must not be started by Stage C")
    if str(payload.get("objective", "")) != STAGE_C_OBJECTIVE:
        raise ValueError("Stage C objective must be proxy_anchor")
    if round(float(payload.get("negative_budget_ratio", -1.0)), 6) != REQUIRED_STAGE_C_RATIO:
        raise ValueError("Stage C negative-budget ratio must be fixed to 0.5")
    seeds = [int(seed) for seed in payload.get("negative_budget_seeds", [])]
    if seeds and tuple(seeds) != REQUIRED_STAGE_C_SEEDS:
        raise ValueError("Stage C seeds are not the frozen required seeds")
    support_winner = str(payload.get("support_winner", ""))
    if support_winner not in STAGE_C_SUPPORT_WINNERS:
        raise ValueError("invalid Stage C support_winner")
    comparator_status = str(payload.get("comparator_status", ""))
    if comparator_status not in STAGE_C_COMPARATOR_STATUSES:
        raise ValueError("invalid Stage C comparator_status")
    panel_verdict = str(payload.get("panel_verdict", ""))
    if panel_verdict not in STAGE_C_PANEL_VERDICTS:
        raise ValueError("invalid Stage C panel_verdict")
    panel_models = [str(model) for model in payload.get("panel_models", [])]
    evaluated_models = [str(model) for model in payload.get("evaluated_models", [])]
    excluded = {str(model): str(reason) for model, reason in dict(payload.get("excluded_models", {}) or {}).items()}
    invalid_exclusions = sorted(model for model in excluded if model != "glm-4.6v-flash")
    if invalid_exclusions:
        raise ValueError("Only GLM may be excluded from Stage C metrics: " + ", ".join(invalid_exclusions))
    if "glm-4.6v-flash" in excluded and excluded["glm-4.6v-flash"] != STAGE_C_GLM_EXCLUSION_REASON:
        raise ValueError("GLM Stage C exclusion reason does not match the frozen answer-format reason")
    failed = {str(model): str(reason) for model, reason in dict(payload.get("failed_models", {}) or {}).items()}
    skipped = {str(model): str(reason) for model, reason in dict(payload.get("skipped_models", {}) or {}).items()}
    missing = sorted(set(panel_models) - set(evaluated_models) - set(excluded) - set(failed) - set(skipped))
    if missing:
        raise ValueError("missing Stage C panel model(s): " + ", ".join(missing))
    invalid_status_models = sorted((set(excluded) | set(failed) | set(skipped)) - set(panel_models))
    if invalid_status_models:
        raise ValueError("Stage C status model(s) are not in the panel: " + ", ".join(invalid_status_models))
    return payload


def _method_stats(rows: Sequence[Mapping[str, object]], *, methods: Sequence[str]) -> dict[str, dict[str, object]]:
    method_set = {str(method) for method in methods}
    by_method: dict[str, list[dict[str, object]]] = {method: [] for method in method_set}
    for row in rows:
        method = str(row.get("method", ""))
        if method not in method_set:
            continue
        score = _finite_float(row.get("pr_auc"))
        roc = _finite_float(row.get("roc_auc"))
        if score is None or roc is None:
            continue
        by_method.setdefault(method, []).append(dict(row))
    output: dict[str, dict[str, object]] = {}
    for method, values in by_method.items():
        if not values:
            continue
        output[method] = {
            "mean_pr_auc": float(np.mean([float(row["pr_auc"]) for row in values])),
            "mean_roc_auc": float(np.mean([float(row["roc_auc"]) for row in values])),
            "num_rows": len(values),
            "win_count": _win_count(method, rows),
        }
    return output


def _win_count(method: str, rows: Sequence[Mapping[str, object]]) -> int:
    by_model_seed: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("method", "")) not in STAGE_C_SUPPORT_METHODS:
            continue
        key = (
            str(row.get("model_alias") or row.get("model_name")),
            int(float(row.get("negative_budget_seed", row.get("seed", 0)))),
        )
        by_model_seed.setdefault(key, []).append(dict(row))
    wins = 0
    for values in by_model_seed.values():
        values.sort(key=lambda row: (-float(row["pr_auc"]), -float(row["roc_auc"]), _simplicity_rank(str(row["method"]))))
        if values and str(values[0]["method"]) == method:
            wins += 1
    return wins


def _simplicity_rank(method: str) -> int:
    return {
        "single_vmf": 0,
        "mixture_vmf": 1,
        "radius_ball": 2,
        "knn": 3,
        "logistic": 4,
    }.get(str(method), 99)


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


__all__ = [
    "PARAMETRIC_STAGE_C_METHODS",
    "STAGE_C_COMPARATOR_STATUSES",
    "STAGE_C_PANEL_VERDICTS",
    "STAGE_C_SUPPORT_WINNERS",
    "build_stage_c_per_model_summary",
    "stage_c_allowed_verdicts",
    "summarize_stage_c_detector_panel",
    "validate_stage_c_summary",
]
