from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import PANEL_MODELS, stage_b_attr, write_unified_manifest


def test_unified_full_cache_manifest_is_only_cache_root_source(tmp_path: Path) -> None:
    load_stage_b_panel_manifest = stage_b_attr(
        "stage_b_manifest",
        "load_stage_b_panel_manifest",
    )
    resolve_stage_b_cache_root = stage_b_attr(
        "stage_b_manifest",
        "resolve_stage_b_cache_root",
    )
    full_cache_root = tmp_path / "full_cache"
    manifest_path = write_unified_manifest(
        full_cache_root,
        use_source_root_for={PANEL_MODELS[1]},
    )
    tempting_legacy_root = tmp_path / "stage0_cache_that_must_not_be_used"
    tempting_legacy_root.mkdir()
    (full_cache_root / "manifests" / "cache_manifest.json").write_text(
        json.dumps({"cache_root": str(tempting_legacy_root)}) + "\n",
        encoding="utf-8",
    )

    manifest = load_stage_b_panel_manifest(full_cache_root)

    assert Path(manifest.path) == manifest_path
    assert [row["model_alias"] for row in manifest.models] == list(PANEL_MODELS)
    assert Path(resolve_stage_b_cache_root(manifest.models[0], full_cache_root)) == (
        full_cache_root / "cache" / PANEL_MODELS[0]
    )
    assert Path(resolve_stage_b_cache_root(manifest.models[1], full_cache_root)) == (
        full_cache_root / "source_cache" / PANEL_MODELS[1]
    )


def test_missing_panel_model_in_manifest_fails(tmp_path: Path) -> None:
    load_stage_b_panel_manifest = stage_b_attr(
        "stage_b_manifest",
        "load_stage_b_panel_manifest",
    )
    full_cache_root = tmp_path / "full_cache"
    write_unified_manifest(full_cache_root, models=PANEL_MODELS[:-1])

    with pytest.raises(ValueError, match="16 panel models|panel model"):
        load_stage_b_panel_manifest(full_cache_root)
