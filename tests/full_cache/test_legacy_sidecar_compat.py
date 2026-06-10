from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from .conftest import full_cache_attr


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _legacy_entry(
    *,
    model_name: str,
    dataset_name: str,
    subset: str,
    sample_id: str,
    question: str,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "source_dataset": dataset_name,
        "subset": subset,
        "split": subset,
        "sample_id": sample_id,
        "image_id": f"{dataset_name}-{subset}-image",
        "image_path": f"images/{dataset_name}-{subset}.jpg",
        "question": question,
        "label": 1,
        "object_name": "cup",
        "answer_text": "yes",
        "parsed_answer": 1,
        "selected_layers": [0, 1],
        "layer_vectors": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        "first_token_logits": torch.tensor([1.0, -1.0], dtype=torch.float32),
    }


def _legacy_sidecar(
    *,
    model_name: str,
    model_family: str,
    dataset_name: str,
    subset: str,
    records_path: Path,
) -> dict[str, object]:
    return {
        "format": "prefill_cache_shard_v1",
        "metadata_version": 1,
        "cache_type": "full_layer_prefill",
        "stage": "stage0",
        "model_name": model_name,
        "model_family": model_family,
        "dataset_name": dataset_name,
        "source_dataset": dataset_name,
        "subset": subset,
        "split": subset,
        "total_layers": 2,
        "selected_layers": [0, 1],
        "num_selected_layers": 2,
        "hidden_dim": 2,
        "token_index": -1,
        "dtype": "float32",
        "num_entries": 1,
        "records_path": str(records_path),
        "tensor_fields": [
            {"field": "first_token_logits", "dtype": "float32", "shape": [2], "numel": 2},
            {"field": "layer_vectors", "dtype": "float32", "shape": [2, 2], "numel": 4},
        ],
    }


def _write_legacy_shard(
    *,
    cache_root: Path,
    records_root: Path,
    model_name: str,
    model_family: str,
    dataset_name: str,
    subset: str,
    sample_id: str,
    question: str,
) -> tuple[Path, Path]:
    records_path = records_root / dataset_name / f"{subset}.jsonl"
    _write_jsonl(
        records_path,
        [
            {
                "dataset_name": dataset_name,
                "source_dataset": dataset_name,
                "subset": subset,
                "split": subset,
                "sample_id": sample_id,
                "image_id": f"{dataset_name}-{subset}-image",
                "image_path": f"images/{dataset_name}-{subset}.jpg",
                "question": question,
                "label": 1,
                "object_name": "cup",
            }
        ],
    )
    shard_dir = cache_root / dataset_name / subset
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard-00000.pt"
    sidecar_path = shard_dir / "shard-00000.pt.json"
    torch.save(
        [
            _legacy_entry(
                model_name=model_name,
                dataset_name=dataset_name,
                subset=subset,
                sample_id=sample_id,
                question=question,
            )
        ],
        shard_path,
    )
    sidecar_path.write_text(
        json.dumps(
            _legacy_sidecar(
                model_name=model_name,
                model_family=model_family,
                dataset_name=dataset_name,
                subset=subset,
                records_path=records_path,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return shard_path, sidecar_path


@pytest.mark.parametrize(
    ("model_alias", "model_family"),
    (("qwen3-vl-8b", "qwen_vl"), ("internvl3.5-8b", "internvl")),
)
def test_accepts_legacy_stage0_sidecars_and_checks_questions_by_dataset_subset(
    tmp_path: Path,
    model_alias: str,
    model_family: str,
) -> None:
    accept_existing_stage0_cache = full_cache_attr("accept_existing_stage0_cache")
    cache_root = tmp_path / "stage0-cache" / model_alias
    records_root = tmp_path / "records"
    output_root = tmp_path / "accepted"
    first_shard, first_sidecar = _write_legacy_shard(
        cache_root=cache_root,
        records_root=records_root,
        model_name=model_alias,
        model_family=model_family,
        dataset_name="pope",
        subset="popular",
        sample_id="1",
        question="Is there a cup in the image?",
    )
    second_shard, second_sidecar = _write_legacy_shard(
        cache_root=cache_root,
        records_root=records_root,
        model_name=model_alias,
        model_family=model_family,
        dataset_name="pope",
        subset="random",
        sample_id="1",
        question="Is there a mug in the image?",
    )
    sidecars_before = {
        first_sidecar: first_sidecar.read_text(encoding="utf-8"),
        second_sidecar: second_sidecar.read_text(encoding="utf-8"),
    }
    shard_sizes_before = {first_shard: first_shard.stat().st_size, second_shard: second_shard.stat().st_size}

    report = accept_existing_stage0_cache(
        model_alias=model_alias,
        stage0_cache_root=cache_root,
        output_root=output_root,
    )

    assert report["status"] == "accepted_existing_stage0"
    assert report["copied_tensors"] is False
    assert not list(output_root.rglob("*.pt"))
    assert {path: path.read_text(encoding="utf-8") for path in sidecars_before} == sidecars_before
    assert {path: path.stat().st_size for path in shard_sizes_before} == shard_sizes_before
    validation = report["validation"]
    assert validation["status"] == "passed"
    assert validation["question_preservation"]["status"] == "passed"
    assert validation["question_preservation"]["num_records_checked"] == 2
    assert {shard["model_alias"] for shard in validation["shards"]} == {model_alias}


def test_accepts_legacy_gemma4_sidecar_without_source_extraction_env_name(tmp_path: Path) -> None:
    accept_separate_env_cache = full_cache_attr("accept_separate_env_cache")
    cache_root = tmp_path / "separate-env-cache" / "gemma-4-12b-it"
    records_root = tmp_path / "records"
    output_root = tmp_path / "accepted"
    _, sidecar_path = _write_legacy_shard(
        cache_root=cache_root,
        records_root=records_root,
        model_name="gemma-4-12b-it",
        model_family="gemma4_unified",
        dataset_name="pope",
        subset="popular",
        sample_id="1",
        question="Is there a cup in the image?",
    )
    sidecar_before = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "extraction_env_name" not in sidecar_before

    report = accept_separate_env_cache(
        model_alias="gemma-4-12b-it",
        separate_env_cache_root=cache_root,
        output_root=output_root,
        extraction_env_name="mind-gemma4-py311",
    )

    assert report["status"] == "accepted_existing_separate_env"
    assert report["extraction_env_name"] == "mind-gemma4-py311"
    assert report["validation"]["status"] == "passed"
    assert report["validation"]["shards"][0]["extraction_env_name"] == "mind-gemma4-py311"
    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == sidecar_before
