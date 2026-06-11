from __future__ import annotations

from pathlib import Path

from .conftest import GLM_MODEL_ALIAS, PANEL_MODELS, stage_b2_attr, write_unified_manifest


def test_stage_b2_uses_unified_manifest_and_keeps_glm_exclusion_non_blocking(tmp_path: Path) -> None:
    load_stage_b2_panel = stage_b2_attr("stage_b2_manifest", "load_stage_b2_panel")
    build_stage_b2_preflight = stage_b2_attr("stage_b2_manifest", "build_stage_b2_preflight")

    full_cache_root = tmp_path / "full_cache"
    write_unified_manifest(full_cache_root, use_source_root_for={PANEL_MODELS[-1]})

    panel = load_stage_b2_panel(full_cache_root)
    preflight = build_stage_b2_preflight(
        panel,
        excluded_models={GLM_MODEL_ALIAS: "GLM answer_text is not parseable"},
        split_ready=True,
        primary_dataset_available=True,
    )

    assert [row["model_alias"] for row in panel.models] == list(PANEL_MODELS)
    assert preflight["total_panel_models"] == 16
    assert preflight["evaluable_models"] == 15
    assert preflight["excluded_models"][GLM_MODEL_ALIAS] == "GLM answer_text is not parseable"
    assert preflight["cache_root_readiness"] == "ready"
