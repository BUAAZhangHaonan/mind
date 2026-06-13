from __future__ import annotations

import pytest

from .conftest import stage_d_attr, stage_d_script_attr


def test_stage_d_verdict_labels_are_frozen() -> None:
    labels = stage_d_attr("stage_d_status", "stage_d_allowed_verdicts")()

    assert labels["domain_expansion"] == [
        "beats_constraint_baselines",
        "matches_constraint_baselines",
        "trails_constraint_baselines",
    ]
    assert labels["stage_e_started_allowed"] is False
    assert labels["method_redesign_allowed"] is False


def test_stage_d_summary_rejects_stage_e_or_method_redesign() -> None:
    validate = stage_d_attr("stage_d_status", "validate_stage_d_summary")
    payload = {
        "stage": "stage_d",
        "stage_e_started": False,
        "method_redesigned": False,
        "domain_expansion_verdict": "matches_constraint_baselines",
        "panel_models": ["glm-4.6v-flash", "model-a"],
        "evaluated_models": ["model-a"],
        "excluded_models": {
            "glm-4.6v-flash": "answer format incompatible with frozen yes/no population rule",
        },
        "failed_models": {},
        "skipped_models": {},
    }

    assert validate(payload)["domain_expansion_verdict"] == "matches_constraint_baselines"

    with pytest.raises(ValueError, match="Stage E"):
        validate({**payload, "stage_e_started": True})

    with pytest.raises(ValueError, match="redesign"):
        validate({**payload, "method_redesigned": True})


def test_stage_d_runner_default_paths() -> None:
    build_parser = stage_d_script_attr("stage_d_run", "build_parser")
    output_paths = stage_d_script_attr("stage_d_run", "_stage_d_output_paths")

    args = build_parser().parse_args([])
    paths = output_paths(args.output_root)

    assert args.output_root.as_posix() == "outputs/stageD"
    assert args.ratio == 0.5
    assert args.objective == "proxy_anchor"
    assert args.protocols == ["repope_to_repope", "repope_to_pope", "repope_to_dashb", "pope_to_dashb"]
    assert paths["preflight_json"].as_posix() == "outputs/stageD/preflight/stageD_preflight.json"
    assert paths["cross_domain_metrics_long"].as_posix() == "outputs/stageD/reports/cross_domain_metrics_long.csv"
    assert paths["domain_expansion_tierA"].as_posix() == "outputs/stageD/reports/domain_expansion_tierA.csv"
    assert paths["related_method_feasibility_md"].as_posix() == "outputs/stageD/reports/related_method_feasibility.md"
    assert paths["model_family_summary"].as_posix() == "outputs/stageD/reports/model_family_summary.csv"
    assert paths["summary_json"].as_posix() == "outputs/stageD/reports/STAGE_D_SUMMARY.json"
