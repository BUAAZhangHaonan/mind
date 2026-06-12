from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    STAGE_C_GLM_EXCLUSION_REASON,
    stage_c_attr,
    write_unified_manifest,
)


def test_stage_c_uses_unified_manifest_and_excludes_glm_without_blocking(tmp_path: Path) -> None:
    load_stage_c_panel = stage_c_attr("stage_c_manifest", "load_stage_c_panel")
    build_stage_c_preflight = stage_c_attr("stage_c_manifest", "build_stage_c_preflight")

    full_cache_root = tmp_path / "full_cache"
    manifest_path = write_unified_manifest(full_cache_root, use_source_root_for={PANEL_MODELS[-1]})

    panel = load_stage_c_panel(full_cache_root)
    preflight = build_stage_c_preflight(
        panel,
        excluded_models={GLM_MODEL_ALIAS: STAGE_C_GLM_EXCLUSION_REASON},
        split_ready=True,
        primary_dataset_available=True,
    )

    assert panel.path == manifest_path
    assert [row["model_alias"] for row in panel.models] == list(PANEL_MODELS)
    assert GLM_MODEL_ALIAS in preflight["panel_models"]
    assert preflight["total_panel_models"] == 16
    assert preflight["evaluable_models"] == 15
    assert preflight["excluded_models"] == {GLM_MODEL_ALIAS: STAGE_C_GLM_EXCLUSION_REASON}
    assert preflight["manifest_source"] == "unified_full_cache_manifest"
    assert preflight["fixed_objective"] == "proxy_anchor"
    assert preflight["fixed_encoder_family"] == "Sphere-Traj-LSTM"
    assert preflight["fixed_negative_budget_ratio"] == 0.5
    assert preflight["stage_d_started"] is False


def test_stage_c_fixed_plan_rejects_non_frozen_values() -> None:
    validate_stage_c_plan = stage_c_attr("stage_c_manifest", "validate_stage_c_plan")

    plan = validate_stage_c_plan(
        ratio=0.5,
        seeds=[20260506, 20260507, 20260508],
        objective="proxy_anchor",
        encoder_family="Sphere-Traj-LSTM",
    )

    assert plan == {
        "objective": "proxy_anchor",
        "encoder_family": "Sphere-Traj-LSTM",
        "negative_budget_ratio": 0.5,
        "seeds": [20260506, 20260507, 20260508],
    }

    with pytest.raises(ValueError, match="Proxy Anchor"):
        validate_stage_c_plan(
            ratio=0.5,
            seeds=[20260506, 20260507, 20260508],
            objective="supcon",
            encoder_family="Sphere-Traj-LSTM",
        )

    with pytest.raises(ValueError, match="0.5"):
        validate_stage_c_plan(
            ratio=0.25,
            seeds=[20260506, 20260507, 20260508],
            objective="proxy_anchor",
            encoder_family="Sphere-Traj-LSTM",
        )
