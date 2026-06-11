from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    STAGE_B3_GLM_EXCLUSION_REASON,
    stage_b3_attr,
    stage_b3_script_attr,
)


def test_stage_b3_summary_accounts_for_models_and_keeps_stage_c_closed() -> None:
    validate_stage_b3_summary = stage_b3_attr(
        "stage_b3_status",
        "validate_stage_b3_summary",
    )

    summary = validate_stage_b3_summary(
        {
            "stage": "stage_b3",
            "stage_c_started": False,
            "detector_selected": False,
            "objective": "proxy_anchor",
            "negative_budget_ratio": 0.5,
            "negative_budget_seeds": [20260506, 20260507, 20260508],
            "panel_models": list(PANEL_MODELS),
            "evaluated_models": [model for model in PANEL_MODELS if model != GLM_MODEL_ALIAS],
            "excluded_models": {GLM_MODEL_ALIAS: STAGE_B3_GLM_EXCLUSION_REASON},
            "verdict": {
                "verdict": "scale_stable_panel",
                "stability_drop_tolerance": 0.02,
            },
            "stage_c_scope": "not_started",
        }
    )

    assert summary["all_panel_models_accounted_for"] is True
    assert summary["stage_c_started"] is False
    assert summary["detector_selected"] is False
    assert summary["objective"] == "proxy_anchor"
    assert summary["negative_budget_ratio"] == 0.5
    assert summary["missing_models"] == []


def test_stage_b3_summary_rejects_detector_or_objective_comparison_language() -> None:
    validate_stage_b3_summary = stage_b3_attr(
        "stage_b3_status",
        "validate_stage_b3_summary",
    )
    base = {
        "stage": "stage_b3",
        "stage_c_started": False,
        "detector_selected": False,
        "objective": "proxy_anchor",
        "negative_budget_ratio": 0.5,
        "negative_budget_seeds": [20260506, 20260507, 20260508],
        "panel_models": list(PANEL_MODELS),
        "evaluated_models": [model for model in PANEL_MODELS if model != GLM_MODEL_ALIAS],
        "excluded_models": {GLM_MODEL_ALIAS: STAGE_B3_GLM_EXCLUSION_REASON},
        "verdict": {"verdict": "scale_stable_panel"},
    }

    with pytest.raises(ValueError, match="Stage C"):
        validate_stage_b3_summary({**base, "stage_c_started": True})

    with pytest.raises(ValueError, match="detector"):
        validate_stage_b3_summary({**base, "detector_selected": True})

    with pytest.raises(ValueError, match="missing"):
        validate_stage_b3_summary({**base, "evaluated_models": PANEL_MODELS[:-2]})

    with pytest.raises(ValueError, match="objective comparison"):
        validate_stage_b3_summary({**base, "objective_comparison": {"winner": "proxy_anchor"}})

    with pytest.raises(ValueError, match="detector|winner"):
        validate_stage_b3_summary({**base, "verdict": {"verdict": "detector_winner"}})

    with pytest.raises(ValueError, match="panel verdict"):
        validate_stage_b3_summary({**base, "verdict": {"verdict": "scale_stability_complete"}})


def test_stage_b3_panel_verdict_labels_are_exact_prompt_labels() -> None:
    scale_stability_verdict = stage_b3_attr("stage_b3_status", "scale_stability_verdict")

    assert scale_stability_verdict(
        [{"model_alias": "a", "verdict": "scale_stable"}]
    )["verdict"] == "scale_stable_panel"
    assert scale_stability_verdict(
        [{"model_alias": "a", "verdict": "scale_sensitive"}]
    )["verdict"] == "scale_sensitive_panel"
    assert scale_stability_verdict(
        [
            {"model_alias": "a", "verdict": "scale_stable"},
            {"model_alias": "b", "verdict": "scale_sensitive"},
        ]
    )["verdict"] == "scale_mixed_panel"
    assert scale_stability_verdict(
        [
            {"model_alias": "a", "verdict": "scale_stable"},
            {"model_alias": "b", "verdict": "scale_stable"},
            {"model_alias": "c", "verdict": "scale_sensitive"},
        ]
    )["verdict"] == "scale_stable_panel"


def test_stage_b3_runner_defaults_are_fixed_to_proxy_anchor_ratio_and_output_root() -> None:
    build_parser = stage_b3_script_attr("stage_b3_run", "build_parser")

    args = build_parser().parse_args([])

    assert args.output_root.as_posix() == "outputs/stageB3"
    assert args.ratio == 0.5
    assert args.seeds == [20260506, 20260507, 20260508]
    assert args.objective == "proxy_anchor"


def test_stage_b3_runner_paths_match_preflight_manifest_report_layout() -> None:
    output_paths = stage_b3_script_attr("stage_b3_run", "_stage_b3_output_paths")

    paths = output_paths(Path("outputs/stageB3"))

    assert paths["preflight_dir"].as_posix() == "outputs/stageB3/preflight"
    assert paths["manifest_dir"].as_posix() == "outputs/stageB3/manifests"
    assert paths["report_dir"].as_posix() == "outputs/stageB3/reports"
    assert paths["preflight_json"].as_posix() == "outputs/stageB3/preflight/stageB3_preflight.json"
    assert paths["metrics_long"].as_posix() == "outputs/stageB3/reports/stageB3_metrics_long.csv"
    assert paths["summary_json"].as_posix() == "outputs/stageB3/reports/STAGE_B3_SUMMARY.json"


def test_stage_b3_metric_row_uses_repope_training_budget_counts() -> None:
    import numpy as np

    metric_row = stage_b3_script_attr("stage_b3_run", "_metric_row")
    row = metric_row(
        model_alias="model-a",
        dataset_family="pope",
        readout="Diag-Classifier",
        labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
        splits=np.asarray(["encoder_train", "encoder_train", "test", "test"]),
        scores=np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        entries=[
            {"stage_b3_split": "encoder_train", "parsed_answer": 0, "label": 0},
            {"stage_b3_split": "encoder_train", "parsed_answer": 1, "label": 0},
            {"stage_b3_split": "test", "parsed_answer": 0, "label": 0},
            {"stage_b3_split": "test", "parsed_answer": 1, "label": 0},
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

    assert row["training_dataset_family"] == "repope"
    assert row["num_encoder_train_correct"] == 100
    assert row["num_encoder_train_hard_hallucination_available"] == 40
    assert row["num_encoder_train_hard_hallucination_used"] == 20
    assert row["num_encoder_train"] == 120
