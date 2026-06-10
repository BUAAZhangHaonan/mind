from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from .conftest import SEPARATE_ENV_MODELS, full_cache_attr, write_synthetic_full_cache_root


@pytest.mark.parametrize("model_alias", SEPARATE_ENV_MODELS)
def test_accepts_existing_separate_env_cache_without_copying_tensors(tmp_path: Path, model_alias: str) -> None:
    accept_separate_env_cache = full_cache_attr("accept_separate_env_cache")
    separate_env_cache_root = tmp_path / "separate-env-cache" / model_alias
    output_root = tmp_path / "accepted"
    write_synthetic_full_cache_root(
        separate_env_cache_root,
        model_alias=model_alias,
        cache_origin="separate_env",
        extraction_env_name=f"{model_alias}-env",
    )

    report = accept_separate_env_cache(
        model_alias=model_alias,
        separate_env_cache_root=separate_env_cache_root,
        output_root=output_root,
        extraction_env_name=f"{model_alias}-env",
    )

    assert report["status"] == "accepted_existing_separate_env"
    assert report["status"] not in {"extracted_separate_env", "needs_extraction"}
    assert report["model_alias"] == model_alias
    assert report["cache_origin"] == "separate_env"
    assert report["extraction_env_name"] == f"{model_alias}-env"
    assert report["copied_tensors"] is False
    assert not list(output_root.rglob("*.pt"))
    manifest_path = Path(report["manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "accepted_existing_separate_env"


def test_separate_env_acceptance_rejects_schema_mismatch(tmp_path: Path) -> None:
    accept_separate_env_cache = full_cache_attr("accept_separate_env_cache")
    error_type = full_cache_attr("FullCacheValidationError")
    separate_env_cache_root = tmp_path / "separate-env-cache" / "gemma-4-12b-it"
    write_synthetic_full_cache_root(
        separate_env_cache_root,
        model_alias="gemma-4-12b-it",
        cache_origin="separate_env",
        extraction_env_name="gemma4-env",
        sidecar_overrides={"total_layers": 3, "selected_layers": [0, 1, 2], "num_selected_layers": 3},
        entry_overrides={"layer_vectors": torch.ones((2, 2), dtype=torch.float32)},
    )

    with pytest.raises(error_type, match="total_layers|layer_vectors|schema"):
        accept_separate_env_cache(
            model_alias="gemma-4-12b-it",
            separate_env_cache_root=separate_env_cache_root,
            output_root=tmp_path / "accepted",
            extraction_env_name="gemma4-env",
        )


def test_separate_env_acceptance_requires_extraction_env_name(tmp_path: Path) -> None:
    accept_separate_env_cache = full_cache_attr("accept_separate_env_cache")
    error_type = full_cache_attr("FullCacheValidationError")
    separate_env_cache_root = tmp_path / "separate-env-cache" / "molmo-7b-d-0924"
    write_synthetic_full_cache_root(
        separate_env_cache_root,
        model_alias="molmo-7b-d-0924",
        cache_origin="separate_env",
        extraction_env_name="molmo-env",
    )

    with pytest.raises((ValueError, error_type), match="extraction_env_name"):
        accept_separate_env_cache(
            model_alias="molmo-7b-d-0924",
            separate_env_cache_root=separate_env_cache_root,
            output_root=tmp_path / "accepted",
            extraction_env_name="",
        )
