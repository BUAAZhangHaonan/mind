from __future__ import annotations

from pathlib import Path

from .conftest import (
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    STAGE_B3_GLM_EXCLUSION_REASON,
    stage_b3_attr,
    write_unified_manifest,
)


def test_stage_b3_uses_unified_manifest_and_keeps_glm_in_panel_but_excluded(tmp_path: Path) -> None:
    load_stage_b3_panel = stage_b3_attr("stage_b3_manifest", "load_stage_b3_panel")
    build_stage_b3_preflight = stage_b3_attr("stage_b3_manifest", "build_stage_b3_preflight")

    full_cache_root = tmp_path / "full_cache"
    manifest_path = write_unified_manifest(full_cache_root, use_source_root_for={PANEL_MODELS[-1]})

    panel = load_stage_b3_panel(full_cache_root)
    preflight = build_stage_b3_preflight(
        panel,
        excluded_models={GLM_MODEL_ALIAS: STAGE_B3_GLM_EXCLUSION_REASON},
        split_ready=True,
        primary_dataset_available=True,
    )

    assert panel.path == manifest_path
    assert [row["model_alias"] for row in panel.models] == list(PANEL_MODELS)
    assert GLM_MODEL_ALIAS in preflight["panel_models"]
    assert preflight["total_panel_models"] == 16
    assert preflight["evaluable_models"] == 15
    assert preflight["excluded_models"] == {GLM_MODEL_ALIAS: STAGE_B3_GLM_EXCLUSION_REASON}
    assert preflight["fixed_objective"] == "proxy_anchor"
    assert preflight["fixed_negative_budget_ratio"] == 0.5
    assert preflight["fixed_seeds"] == [20260506, 20260507, 20260508]
    assert preflight["cache_root_readiness"] == "ready"
