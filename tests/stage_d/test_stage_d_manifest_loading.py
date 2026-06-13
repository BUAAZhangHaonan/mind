from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    STAGE_D_GLM_EXCLUSION_REASON,
    stage_d_attr,
    write_unified_manifest,
)


def test_stage_d_uses_unified_manifest_and_keeps_glm_in_panel(tmp_path: Path) -> None:
    load_stage_d_panel = stage_d_attr("stage_d_manifest", "load_stage_d_panel")
    build_stage_d_preflight = stage_d_attr("stage_d_manifest", "build_stage_d_preflight")

    full_cache_root = tmp_path / "full_cache"
    manifest_path = write_unified_manifest(full_cache_root, use_source_root_for={PANEL_MODELS[-1]})

    panel = load_stage_d_panel(full_cache_root)
    preflight = build_stage_d_preflight(
        panel,
        excluded_models={GLM_MODEL_ALIAS: STAGE_D_GLM_EXCLUSION_REASON},
        split_ready=True,
    )

    assert panel.path == manifest_path
    assert [row["model_alias"] for row in panel.models] == list(PANEL_MODELS)
    assert GLM_MODEL_ALIAS in preflight["panel_models"]
    assert preflight["total_panel_models"] == 16
    assert preflight["evaluable_models"] == 15
    assert preflight["excluded_models"] == {GLM_MODEL_ALIAS: STAGE_D_GLM_EXCLUSION_REASON}
    assert preflight["manifest_source"] == "unified_full_cache_manifest"
    assert preflight["frozen_main_method"] == "Sphere-Traj-LSTM + Proxy Anchor + radius_ball"
    assert preflight["stage_e_started"] is False
    assert preflight["method_redesigned"] is False


def test_stage_d_fixed_plan_rejects_method_drift() -> None:
    validate_stage_d_plan = stage_d_attr("stage_d_manifest", "validate_stage_d_plan")

    plan = validate_stage_d_plan(
        ratio=0.5,
        seeds=[20260506, 20260507, 20260508],
        objective="proxy_anchor",
        encoder_family="Sphere-Traj-LSTM",
        main_detector="radius_ball",
        param_detector="single_vmf",
    )

    assert plan["objective"] == "proxy_anchor"
    assert plan["negative_budget_ratio"] == 0.5
    assert plan["main_detector"] == "radius_ball"
    assert plan["param_detector"] == "single_vmf"

    with pytest.raises(ValueError, match="radius_ball"):
        validate_stage_d_plan(
            ratio=0.5,
            seeds=[20260506, 20260507, 20260508],
            objective="proxy_anchor",
            encoder_family="Sphere-Traj-LSTM",
            main_detector="knn",
            param_detector="single_vmf",
        )
