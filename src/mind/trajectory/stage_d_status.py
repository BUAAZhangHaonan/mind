"""Stage D family aggregation and summary validation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np

from .stage_d_manifest import STAGE_D_GLM_EXCLUSION_REASON


STAGE_D_FAMILIES = ("qwen", "internvl", "llava", "gemma", "phi", "minicpm", "glm", "molmo")
STAGE_D_DOMAIN_EXPANSION_VERDICTS = (
    "beats_constraint_baselines",
    "matches_constraint_baselines",
    "trails_constraint_baselines",
)


def stage_d_allowed_verdicts() -> dict[str, object]:
    """Return frozen Stage D verdict labels."""

    return {
        "domain_expansion": list(STAGE_D_DOMAIN_EXPANSION_VERDICTS),
        "stage_e_started_allowed": False,
        "method_redesign_allowed": False,
    }


def stage_d_family_contract() -> dict[str, object]:
    """Return the Stage D family vocabulary and mapper."""

    return {
        "families": list(STAGE_D_FAMILIES),
        "model_to_family": model_family,
    }


def model_family(model_alias: str) -> str:
    """Map one model alias to the frozen Stage D family vocabulary."""

    name = str(model_alias).lower()
    if "qwen" in name:
        return "qwen"
    if "internvl" in name:
        return "internvl"
    if "llava" in name:
        return "llava"
    if "gemma" in name:
        return "gemma"
    if "phi" in name:
        return "phi"
    if "minicpm" in name:
        return "minicpm"
    if "glm" in name:
        return "glm"
    if "molmo" in name:
        return "molmo"
    return "unknown"


def build_stage_d_family_summary(
    *,
    panel_models: Sequence[str],
    metric_rows: Sequence[Mapping[str, object]],
    excluded_models: Mapping[str, object] | None = None,
    separate_env_models: set[str] | frozenset[str] | None = None,
) -> list[dict[str, object]]:
    """Aggregate Stage D metrics into compact family rows."""

    excluded = {str(model): str(reason) for model, reason in dict(excluded_models or {}).items()}
    separate = {str(model) for model in (separate_env_models or set())}
    rows_by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if str(row.get("metric_status", "passed")) != "passed":
            continue
        rows_by_family[model_family(str(row.get("model_alias", "")))].append(row)

    panel_by_family: dict[str, list[str]] = defaultdict(list)
    for model in panel_models:
        panel_by_family[model_family(str(model))].append(str(model))

    output: list[dict[str, object]] = []
    for family in STAGE_D_FAMILIES:
        family_models = panel_by_family.get(family, [])
        family_rows = rows_by_family.get(family, [])
        evaluated_models = sorted({str(row.get("model_alias", "")) for row in family_rows})
        method_means = _method_mean_summary(family_rows)
        win_counts = _method_win_counts(family_rows)
        excluded_notes = [
            f"{model}: {excluded[model]}"
            for model in family_models
            if model in excluded
        ]
        separate_note = (
            "contains verified_separate_env model"
            if any(model in separate for model in family_models)
            else ""
        )
        output.append(
            {
                "family": family,
                "num_panel_models": len(family_models),
                "num_evaluable_models": len(evaluated_models),
                "panel_models": ";".join(family_models),
                "evaluable_models": ";".join(evaluated_models),
                "family_mean_pr_auc_by_method": method_means,
                "family_win_counts": win_counts,
                "family_specific_notes": "; ".join(excluded_notes),
                "main_env_vs_separate_env_note": separate_note,
            }
        )
    return output


def domain_expansion_verdict(metric_rows: Sequence[Mapping[str, object]]) -> str:
    """Compare MIND-main against Tier A constraint baselines."""

    rows = [
        row
        for row in metric_rows
        if str(row.get("metric_status", "passed")) == "passed"
        and str(row.get("calibration_scope", "")) == "source_calibration"
        and str(row.get("method", "")) in {
            "MIND-main",
            "MIND-param",
            "logistic(z)",
            "final-hidden linear probe",
            "output-confidence",
            "HALP-lite",
        }
    ]
    if not rows:
        return "trails_constraint_baselines"
    method_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = _finite_float(row.get("pr_auc"))
        if score is not None:
            method_scores[str(row["method"])].append(score)
    main = np.mean(method_scores.get("MIND-main", [float("nan")]))
    competitors = [
        np.mean(values)
        for method, values in method_scores.items()
        if method != "MIND-main" and values
    ]
    if not np.isfinite(main) or not competitors:
        return "trails_constraint_baselines"
    best_competitor = float(max(competitors))
    delta = float(main - best_competitor)
    if delta >= 0.01:
        return "beats_constraint_baselines"
    if delta <= -0.01:
        return "trails_constraint_baselines"
    return "matches_constraint_baselines"


def validate_stage_d_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Validate Stage D scope and panel coverage."""

    payload = dict(summary)
    if bool(payload.get("stage_e_started", False)):
        raise ValueError("Stage E must not be started by Stage D")
    if bool(payload.get("method_redesigned", False)):
        raise ValueError("Stage D must not redesign the frozen method")
    verdict = str(payload.get("domain_expansion_verdict", ""))
    if verdict not in STAGE_D_DOMAIN_EXPANSION_VERDICTS:
        raise ValueError("invalid Stage D domain_expansion_verdict")
    panel = {str(model) for model in payload.get("panel_models", [])}
    evaluated = {str(model) for model in payload.get("evaluated_models", [])}
    excluded = {str(model): str(reason) for model, reason in dict(payload.get("excluded_models", {}) or {}).items()}
    failed = {str(model): str(reason) for model, reason in dict(payload.get("failed_models", {}) or {}).items()}
    skipped = {str(model): str(reason) for model, reason in dict(payload.get("skipped_models", {}) or {}).items()}
    invalid_exclusions = sorted(model for model in excluded if model != "glm-4.6v-flash")
    if invalid_exclusions:
        raise ValueError("Only GLM may be excluded from Stage D metrics: " + ", ".join(invalid_exclusions))
    if "glm-4.6v-flash" in excluded and excluded["glm-4.6v-flash"] != STAGE_D_GLM_EXCLUSION_REASON:
        raise ValueError("GLM Stage D exclusion reason does not match the frozen answer-format reason")
    missing = sorted(panel - evaluated - set(excluded) - set(failed) - set(skipped))
    if missing:
        raise ValueError("missing Stage D panel model(s): " + ", ".join(missing))
    invalid_status = sorted((set(excluded) | set(failed) | set(skipped)) - panel)
    if invalid_status:
        raise ValueError("Stage D status model(s) are not in the panel: " + ", ".join(invalid_status))
    return payload


def _method_mean_summary(rows: Sequence[Mapping[str, object]]) -> str:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = _finite_float(row.get("pr_auc"))
        if score is not None:
            scores[str(row.get("method", ""))].append(score)
    return ";".join(
        f"{method}:{float(np.mean(values)):.6f}"
        for method, values in sorted(scores.items())
        if values
    )


def _method_win_counts(rows: Sequence[Mapping[str, object]]) -> str:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model_alias", "")),
            str(row.get("protocol", "")),
            str(row.get("negative_budget_seed", "")),
        )
        grouped[key].append(row)
    wins: Counter[str] = Counter()
    for values in grouped.values():
        passed = [
            row
            for row in values
            if str(row.get("metric_status", "passed")) == "passed"
            and _finite_float(row.get("pr_auc")) is not None
        ]
        if not passed:
            continue
        passed.sort(key=lambda row: (-float(row["pr_auc"]), str(row.get("method", ""))))
        wins[str(passed[0].get("method", ""))] += 1
    return ";".join(f"{method}:{count}" for method, count in sorted(wins.items()))


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


__all__ = [
    "STAGE_D_DOMAIN_EXPANSION_VERDICTS",
    "STAGE_D_FAMILIES",
    "build_stage_d_family_summary",
    "domain_expansion_verdict",
    "model_family",
    "stage_d_allowed_verdicts",
    "stage_d_family_contract",
    "validate_stage_d_summary",
]
