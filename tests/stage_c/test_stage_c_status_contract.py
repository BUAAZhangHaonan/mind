from __future__ import annotations

from pathlib import Path

from .conftest import stage_c_script_attr


def test_stage_c_runner_defaults_and_paths_match_required_outputs() -> None:
    build_parser = stage_c_script_attr("stage_c_run", "build_parser")
    output_paths = stage_c_script_attr("stage_c_run", "_stage_c_output_paths")

    args = build_parser().parse_args([])
    paths = output_paths(Path("outputs/stageC"))

    assert args.output_root.as_posix() == "outputs/stageC"
    assert args.ratio == 0.5
    assert args.seeds == [20260506, 20260507, 20260508]
    assert args.objective == "proxy_anchor"
    assert paths["preflight_json"].as_posix() == "outputs/stageC/preflight/stageC_preflight.json"
    assert paths["metrics_long"].as_posix() == "outputs/stageC/reports/stageC_metrics_long.csv"
    assert paths["repope_main_table"].as_posix() == "outputs/stageC/reports/repope_main_table.csv"
    assert paths["pope_secondary_table"].as_posix() == "outputs/stageC/reports/pope_secondary_table.csv"
    assert paths["dash_b_secondary_table"].as_posix() == "outputs/stageC/reports/dash_b_secondary_table.csv"
    assert paths["knn_selected_k"].as_posix() == "outputs/stageC/reports/knn_selected_k.csv"
    assert paths["radius_ball_selected_rho"].as_posix() == "outputs/stageC/reports/radius_ball_selected_rho.csv"
    assert paths["vmf_selected_k"].as_posix() == "outputs/stageC/reports/vmf_selected_k.csv"
    assert paths["logistic_selected_c"].as_posix() == "outputs/stageC/reports/logistic_selected_c.csv"
    assert paths["per_model_detector_summary"].as_posix() == "outputs/stageC/reports/per_model_detector_summary.csv"
    assert paths["summary_json"].as_posix() == "outputs/stageC/reports/STAGE_C_SUMMARY.json"


def test_stage_c_summary_panel_uses_full_manifest_not_requested_subset() -> None:
    build_summary_panel = stage_c_script_attr("stage_c_run", "_build_stage_c_summary_panel")

    panel = ["glm-4.6v-flash", "model-a", "model-b"]
    requested = ["model-a"]
    result = build_summary_panel(panel_models=panel, requested_models=requested)

    assert result["panel_models"] == panel
    assert result["skipped_models"] == {
        "model-b": "not requested in this run",
    }
    assert "glm-4.6v-flash" not in result["skipped_models"]
