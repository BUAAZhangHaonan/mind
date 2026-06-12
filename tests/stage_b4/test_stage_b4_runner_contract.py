from __future__ import annotations

from pathlib import Path

import numpy as np

from .conftest import stage_b4_script_attr


def test_stage_b4_runner_defaults_are_fixed_to_proxy_anchor_ratio_and_output_root() -> None:
    build_parser = stage_b4_script_attr("stage_b4_run", "build_parser")

    args = build_parser().parse_args([])

    assert args.output_root.as_posix() == "outputs/stageB4"
    assert args.ratio == 0.5
    assert args.seeds == [20260506, 20260507, 20260508]
    assert args.objective == "proxy_anchor"


def test_stage_b4_runner_paths_match_required_output_names() -> None:
    output_paths = stage_b4_script_attr("stage_b4_run", "_stage_b4_output_paths")

    paths = output_paths(Path("outputs/stageB4"))

    assert paths["preflight_dir"].as_posix() == "outputs/stageB4/preflight"
    assert paths["manifest_dir"].as_posix() == "outputs/stageB4/manifests"
    assert paths["report_dir"].as_posix() == "outputs/stageB4/reports"
    assert paths["preflight_json"].as_posix() == "outputs/stageB4/preflight/stageB4_preflight.json"
    assert paths["repope_split_manifest"].as_posix() == "outputs/stageB4/manifests/repope_family_split_manifest.json"
    assert paths["pope_split_manifest"].as_posix() == "outputs/stageB4/manifests/pope_family_split_manifest.json"
    assert paths["dash_b_split_manifest"].as_posix() == "outputs/stageB4/manifests/dash_b_split_manifest.json"
    assert paths["metrics_long"].as_posix() == "outputs/stageB4/reports/stageB4_metrics_long.csv"
    assert paths["repope_support_family_knn"].as_posix() == "outputs/stageB4/reports/repope_support_family_knn.csv"
    assert paths["repope_support_family_single_vmf"].as_posix() == "outputs/stageB4/reports/repope_support_family_single_vmf.csv"
    assert paths["repope_support_family_mixture_vmf"].as_posix() == "outputs/stageB4/reports/repope_support_family_mixture_vmf.csv"
    assert paths["pope_secondary_table"].as_posix() == "outputs/stageB4/reports/pope_secondary_table.csv"
    assert paths["dash_b_secondary_table"].as_posix() == "outputs/stageB4/reports/dash_b_secondary_table.csv"
    assert paths["knn_scale_grid"].as_posix() == "outputs/stageB4/reports/knn_scale_grid.csv"
    assert paths["knn_selected_k"].as_posix() == "outputs/stageB4/reports/knn_selected_k.csv"
    assert paths["knn_stability_band"].as_posix() == "outputs/stageB4/reports/knn_stability_band.csv"
    assert "per_model_knn_scale_stability" not in paths
    assert paths["vmf_selected_k"].as_posix() == "outputs/stageB4/reports/vmf_selected_k.csv"
    assert paths["vmf_stability_band"].as_posix() == "outputs/stageB4/reports/vmf_stability_band.csv"
    assert paths["per_model_support_family_summary"].as_posix() == "outputs/stageB4/reports/per_model_support_family_summary.csv"
    assert paths["classifier_control"].as_posix() == "outputs/stageB4/reports/classifier_control.csv"
    assert paths["summary_json"].as_posix() == "outputs/stageB4/reports/STAGE_B4_SUMMARY.json"


def test_stage_b4_metric_row_uses_repope_training_budget_counts() -> None:
    metric_row = stage_b4_script_attr("stage_b4_run", "_metric_row")

    row = metric_row(
        model_alias="model-a",
        dataset_family="pope",
        readout="Diag-Classifier",
        support_family="classifier_control",
        labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
        splits=np.asarray(["encoder_train", "encoder_train", "test", "test"]),
        scores=np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        entries=[
            {"stage_b4_split": "encoder_train", "parsed_answer": 0, "label": 0},
            {"stage_b4_split": "encoder_train", "parsed_answer": 1, "label": 0},
            {"stage_b4_split": "test", "parsed_answer": 0, "label": 0},
            {"stage_b4_split": "test", "parsed_answer": 1, "label": 0},
        ],
        all_entries=[],
        bootstrap=2,
        seed=20260506,
        ratio=0.5,
        selected_k="",
        num_bank_correct=7,
        train_correct_available=100,
        train_hard_available=40,
        used_hard=20,
    )

    assert row["objective"] == "proxy_anchor"
    assert row["training_dataset_family"] == "repope"
    assert row["support_family"] == "classifier_control"
    assert row["selected_K"] == ""
    assert row["num_encoder_train_correct"] == 100
    assert row["num_encoder_train_hard_hallucination_available"] == 40
    assert row["num_encoder_train_hard_hallucination_used"] == 20
    assert row["num_encoder_train"] == 120
