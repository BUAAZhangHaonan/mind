from __future__ import annotations

import pytest

from mind.trajectory.stage_a_closeout import (
    load_closeout_panel_manifest,
    resolve_model_cache_root,
)

from .conftest import PANEL_MODELS, write_unified_manifest


def test_unified_manifest_is_source_of_cache_roots(tmp_path) -> None:
    full_cache_root = tmp_path / "full_cache"
    write_unified_manifest(
        full_cache_root,
        use_source_root_for={"model-01"},
    )

    manifest = load_closeout_panel_manifest(full_cache_root)

    assert [row["model_alias"] for row in manifest.models] == list(PANEL_MODELS)
    assert resolve_model_cache_root(manifest.models[0], full_cache_root).name == "model-00"
    assert resolve_model_cache_root(manifest.models[1], full_cache_root).name == "model-01"


def test_missing_model_in_manifest_fails(tmp_path) -> None:
    full_cache_root = tmp_path / "full_cache"
    write_unified_manifest(full_cache_root, models=PANEL_MODELS[:-1])

    with pytest.raises(ValueError, match="16 panel models"):
        load_closeout_panel_manifest(full_cache_root)
