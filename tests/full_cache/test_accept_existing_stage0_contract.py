from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import STAGE0_ACCEPT_MODELS, full_cache_attr, write_synthetic_full_cache_root


@pytest.mark.parametrize("model_alias", STAGE0_ACCEPT_MODELS)
def test_accepts_existing_stage0_cache_root_without_copying_tensors(tmp_path: Path, model_alias: str) -> None:
    accept_existing_stage0_cache = full_cache_attr("accept_existing_stage0_cache")
    stage0_cache_root = tmp_path / "stage0-cache" / model_alias
    output_root = tmp_path / "accepted"
    write_synthetic_full_cache_root(stage0_cache_root, model_alias=model_alias, cache_origin="stage0")

    report = accept_existing_stage0_cache(
        model_alias=model_alias,
        stage0_cache_root=stage0_cache_root,
        output_root=output_root,
    )

    assert report["status"] == "accepted_existing_stage0"
    assert report["model_alias"] == model_alias
    assert report["cache_origin"] == "stage0"
    assert report["copied_tensors"] is False
    assert report["source_cache_root"] == str(stage0_cache_root)
    assert not list(output_root.rglob("*.pt"))
    manifest_path = Path(report["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "accepted_existing_stage0"
    assert manifest["copied_tensors"] is False
