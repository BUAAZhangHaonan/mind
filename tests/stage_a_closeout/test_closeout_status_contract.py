from __future__ import annotations

from mind.trajectory.stage_a_closeout import (
    CLOSEOUT_VARIANTS,
    decide_sphere_closeout_verdict,
    summarize_closeout_status,
)

from scripts import stage_a_closeout_run

from .conftest import PANEL_MODELS


def test_closeout_uses_exactly_six_variants() -> None:
    assert CLOSEOUT_VARIANTS == (
        "Raw-Static",
        "Sphere-Static",
        "Raw-Traj-MeanPool",
        "Sphere-Traj-MeanPool",
        "Raw-Traj-LSTM",
        "Sphere-Traj-LSTM",
    )


def test_verdict_uses_fixed_labels_only() -> None:
    rows = [
        {
            "model_name": alias,
            "dataset_family": "repope",
            "variant": variant,
            "readout": "Diag-Classifier",
            "eval_split": "test",
            "eval_scope": "pooled",
            "metric_status": "passed",
            "pr_auc": 0.70 if variant == "Sphere-Traj-LSTM" else 0.69,
        }
        for alias in PANEL_MODELS
        for variant in ("Raw-Traj-LSTM", "Sphere-Traj-LSTM")
    ]

    verdict = decide_sphere_closeout_verdict(rows)

    assert verdict["verdict"] in {"beneficial", "neutral", "harmful"}
    assert verdict["verdict"] == "beneficial"


def test_failed_models_are_reported_not_skipped() -> None:
    summary = summarize_closeout_status(
        panel_models=PANEL_MODELS,
        metric_rows=[],
        failures={model: "synthetic failure" for model in PANEL_MODELS},
    )

    assert "model-03" in summary["failed_models"]
    assert summary["all_panel_models_present"] is True
    assert summary["stage_b_started"] is False


def test_failed_metric_rows_cover_all_variants_and_readouts() -> None:
    rows = stage_a_closeout_run._failed_metric_rows("model-03", "repope", "no primary population")

    assert len(rows) == 12
    assert {row["variant"] for row in rows} == set(CLOSEOUT_VARIANTS)
    assert {row["readout"] for row in rows} == {"Diag-Classifier", "Diag-KNN"}
    assert {row["metric_status"] for row in rows} == {"failed"}
