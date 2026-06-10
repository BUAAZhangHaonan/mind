from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .conftest import SYNTHETIC_QUESTION, full_cache_attr, write_synthetic_full_cache_root


def test_valid_synthetic_shard_passes_and_records_question_preservation(tmp_path: Path) -> None:
    validate_full_cache_root = full_cache_attr("validate_full_cache_root")
    cache_root = tmp_path / "cache" / "qwen3-vl-8b"
    output = tmp_path / "manifest.json"
    write_synthetic_full_cache_root(cache_root, model_alias="qwen3-vl-8b", cache_origin="stage0")

    manifest = validate_full_cache_root(
        cache_root=cache_root,
        expected_model_alias="qwen3-vl-8b",
        cache_origin="stage0",
        output=output,
    )

    assert output.is_file()
    assert manifest["status"] == "passed"
    assert manifest["total_entries"] == 3
    assert manifest["question_preservation"]["status"] == "passed"
    assert manifest["question_preservation"]["examples"][0]["question"] == SYNTHETIC_QUESTION


@pytest.mark.parametrize(
    "field",
    (
        "model_alias",
        "model_family",
        "dataset_name",
        "source_dataset",
        "subset",
        "split",
        "sample_id",
        "image_id",
        "image_path",
        "question",
        "label",
        "object_name",
        "answer_text",
        "parsed_answer",
        "selected_layers",
        "layer_vectors",
        "first_token_logits",
        "token_index",
        "prompt_template_id",
        "logit_source",
    ),
)
def test_missing_required_entry_fields_fail(tmp_path: Path, field: str) -> None:
    validate_full_cache_root = full_cache_attr("validate_full_cache_root")
    error_type = full_cache_attr("FullCacheValidationError")
    cache_root = tmp_path / "cache" / field
    write_synthetic_full_cache_root(
        cache_root,
        model_alias="qwen3-vl-8b",
        cache_origin="stage0",
        drop_entry_fields=(field,),
    )

    with pytest.raises(error_type, match=field):
        validate_full_cache_root(
            cache_root=cache_root,
            expected_model_alias="qwen3-vl-8b",
            cache_origin="stage0",
        )


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "cache_type",
        "cache_origin",
        "model_alias",
        "model_family",
        "dataset_name",
        "source_dataset",
        "subset",
        "split",
        "total_layers",
        "selected_layers",
        "num_selected_layers",
        "hidden_dim",
        "token_index",
        "dtype",
        "num_entries",
        "prompt_template_id",
        "logit_source",
    ),
)
def test_missing_required_sidecar_fields_fail(tmp_path: Path, field: str) -> None:
    validate_full_cache_root = full_cache_attr("validate_full_cache_root")
    error_type = full_cache_attr("FullCacheValidationError")
    cache_root = tmp_path / "cache" / field
    write_synthetic_full_cache_root(
        cache_root,
        model_alias="qwen3-vl-8b",
        cache_origin="stage0",
        drop_sidecar_fields=(field,),
    )

    with pytest.raises(error_type, match=field):
        validate_full_cache_root(
            cache_root=cache_root,
            expected_model_alias="qwen3-vl-8b",
            cache_origin="stage0",
        )


def test_non_contiguous_selected_layers_fail(tmp_path: Path) -> None:
    validate_full_cache_root = full_cache_attr("validate_full_cache_root")
    error_type = full_cache_attr("FullCacheValidationError")
    cache_root = tmp_path / "cache" / "non-contiguous"
    write_synthetic_full_cache_root(
        cache_root,
        model_alias="qwen3-vl-8b",
        cache_origin="stage0",
        sidecar_overrides={"total_layers": 3, "selected_layers": [0, 2], "num_selected_layers": 2},
        entry_overrides={
            "selected_layers": [0, 2],
            "layer_vectors": torch.ones((2, 2), dtype=torch.float32),
        },
    )

    with pytest.raises(error_type, match="selected_layers|contiguous"):
        validate_full_cache_root(
            cache_root=cache_root,
            expected_model_alias="qwen3-vl-8b",
            cache_origin="stage0",
        )


def test_layer_vectors_length_must_equal_total_layers(tmp_path: Path) -> None:
    validate_full_cache_root = full_cache_attr("validate_full_cache_root")
    error_type = full_cache_attr("FullCacheValidationError")
    cache_root = tmp_path / "cache" / "layer-length"
    write_synthetic_full_cache_root(
        cache_root,
        model_alias="qwen3-vl-8b",
        cache_origin="stage0",
        sidecar_overrides={"total_layers": 3, "selected_layers": [0, 1, 2], "num_selected_layers": 3},
        entry_overrides={
            "selected_layers": [0, 1, 2],
            "layer_vectors": torch.ones((2, 2), dtype=torch.float32),
        },
    )

    with pytest.raises(error_type, match="layer_vectors|total_layers"):
        validate_full_cache_root(
            cache_root=cache_root,
            expected_model_alias="qwen3-vl-8b",
            cache_origin="stage0",
        )


def test_separate_env_shards_require_separate_env_metadata(tmp_path: Path) -> None:
    validate_full_cache_root = full_cache_attr("validate_full_cache_root")
    error_type = full_cache_attr("FullCacheValidationError")
    cache_root = tmp_path / "cache" / "separate-env"
    write_synthetic_full_cache_root(
        cache_root,
        model_alias="gemma-4-12b-it",
        cache_origin="separate_env",
        extraction_env_name="gemma4-env",
        drop_sidecar_fields=("extraction_env_name",),
    )

    with pytest.raises(error_type, match="extraction_env_name"):
        validate_full_cache_root(
            cache_root=cache_root,
            expected_model_alias="gemma-4-12b-it",
            cache_origin="separate_env",
            extraction_env_name="gemma4-env",
        )
