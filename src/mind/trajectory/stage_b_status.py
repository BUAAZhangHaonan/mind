"""Stage B status and verdict helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .stage_b_objectives import ALLOWED_STAGE_B_OBJECTIVES


STAGE_B_VERDICT_LABELS = ("winner", "tie", "inconclusive")


def summarize_stage_b_status(
    *,
    panel_models: Sequence[str],
    metric_rows: Sequence[Mapping[str, object]],
    excluded_models: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Summarize Stage B without hiding missing panel models."""

    panel = [str(model) for model in panel_models]
    excluded = {
        str(model): _reason_text(reason)
        for model, reason in dict(excluded_models or {}).items()
        if bool(reason)
    }
    evaluated = sorted(
        {
            str(row.get("model_alias") or row.get("model_name"))
            for row in metric_rows
            if row.get("model_alias") or row.get("model_name")
        }
    )
    missing = sorted(set(panel) - set(evaluated) - set(excluded))
    if missing:
        raise ValueError(
            "missing Stage B panel model metric rows without an excluded reason: "
            + ", ".join(missing)
        )

    model_status: dict[str, dict[str, object]] = {}
    for model in panel:
        if model in excluded:
            model_status[model] = {"status": "excluded", "reason": excluded[model]}
        elif model in evaluated:
            model_status[model] = {"status": "evaluated", "reason": ""}
        else:
            model_status[model] = {"status": "missing", "reason": "no metric row"}

    return {
        "stage": "stage_b",
        "stage_b_started": True,
        "stage_c_started": False,
        "panel_models": panel,
        "model_status": model_status,
        "evaluated_models": evaluated,
        "excluded_models": excluded,
        "missing_models": missing,
        "all_panel_models_accounted_for": not missing,
    }


def decide_stage_b_verdict(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    objectives: Sequence[str] = ALLOWED_STAGE_B_OBJECTIVES,
) -> dict[str, object]:
    """Choose the best Stage B objective by mean RePOPE PR-AUC."""

    allowed = tuple(str(objective) for objective in objectives)
    scores: dict[str, list[float]] = {objective: [] for objective in allowed}
    for row in metric_rows:
        objective = str(row.get("objective", ""))
        if objective not in scores:
            continue
        if str(row.get("metric_status", "passed")) != "passed":
            continue
        value = _finite_float(row.get("pr_auc"))
        if value is not None:
            scores[objective].append(value)

    means = {
        objective: sum(values) / len(values)
        for objective, values in scores.items()
        if values
    }
    if not means:
        return {
            "verdict": "inconclusive",
            "winner": None,
            "winners": [],
            "objective_scores": {},
            "reason": "no finite Stage B PR-AUC rows were available",
        }

    best = max(means.values())
    winners = [objective for objective, value in means.items() if math.isclose(value, best, abs_tol=1e-12)]
    if len(winners) == 1:
        return {
            "verdict": "winner",
            "winner": winners[0],
            "winners": winners,
            "objective_scores": means,
        }
    return {
        "verdict": "tie",
        "winner": None,
        "winners": winners,
        "objective_scores": means,
    }


def validate_stage_b_status(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate a Stage B status payload shape."""

    verdict = summary.get("verdict")
    if isinstance(verdict, Mapping):
        label = str(verdict.get("verdict", ""))
        if label not in STAGE_B_VERDICT_LABELS:
            raise ValueError("Stage B verdict must be winner, tie, or inconclusive")
    if bool(summary.get("stage_c_started", False)):
        raise ValueError("Stage C must not be started by Stage B")
    return dict(summary)


def render_stage_b_summary_markdown(summary: Mapping[str, object]) -> str:
    """Render a concise Stage B status report."""

    verdict = summary.get("verdict", {})
    if not isinstance(verdict, Mapping):
        verdict = {}
    status = summary.get("status", {})
    if not isinstance(status, Mapping):
        status = {}
    model_status = status.get("model_status", {})
    if not isinstance(model_status, Mapping):
        model_status = {}
    lines = [
        "# Stage B Summary",
        "",
        "Stage B1 identifies objective families only. It does not validate the final MIND detector.",
        "",
        f"- stage_b_started: {str(summary.get('stage_b_started', True)).lower()}",
        f"- stage_c_started: {str(summary.get('stage_c_started', False)).lower()}",
        f"- verdict: {verdict.get('verdict', 'inconclusive')}",
        "",
        "## Model Status",
        "",
        "| model | status | reason |",
        "| --- | --- | --- |",
    ]
    for model, row in model_status.items():
        if isinstance(row, Mapping):
            reason = str(row.get("reason", "")).replace("|", "\\|")
            lines.append(f"| {model} | {row.get('status', '')} | {reason} |")
    return "\n".join(lines) + "\n"


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _reason_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is True:
        return "excluded"
    return str(value)
