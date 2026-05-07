from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _cache_api() -> tuple[object, type[Exception]]:
    module = importlib.import_module("mind.trajectory.cache")
    return module.validate_stage0_cache, module.CacheValidationError


def _load_script() -> ModuleType:
    script_path = Path("scripts/v2/stage0_validate_cache.py")
    spec = importlib.util.spec_from_file_location("stage0_validate_cache", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "sample_id": "sample-001",
        "image_id": "image-001",
        "image_path": "images/000001.jpg",
        "question": "Is there a cat in the image?",
        "label": "yes",
        "object_name": "cat",
        "answer_text": "yes",
        "parsed_answer": 1,
        "first_token_logits": torch.tensor([0.0, 3.0, -1.0], dtype=torch.float32),
        "selected_layers": [0, 1],
        "layer_vectors": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        ),
    }
    entry.update(overrides)
    return entry


def _base_sidecar(**overrides: object) -> dict[str, object]:
    sidecar: dict[str, object] = {
        "stage": "v2_stage0",
        "cache_type": "full_layer_hidden_states",
        "model_name": "tiny-model",
        "model_id": "org/tiny-model",
        "model_family": "tiny",
        "dataset_name": "pope",
        "subset": "popular",
        "split": "encoder_train",
        "total_layers": 2,
        "selected_layers": [0, 1],
        "num_selected_layers": 2,
        "hidden_dim": 3,
        "token_index": -1,
        "max_new_tokens": 4,
        "dtype": "float32",
        "num_entries": 1,
        "script": "synthetic",
        "git_commit": "deadbeef",
        "created_at_utc": "2026-05-07T00:00:00Z",
        "records_path": "records.jsonl",
        "image_root": "images",
    }
    sidecar.update(overrides)
    return sidecar


def _write_shard(
    cache_root: Path,
    *,
    name: str = "shard-00000.pt",
    entries: list[dict[str, object]] | None = None,
    sidecar: dict[str, object] | None = None,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_path = cache_root / name
    payload = [_base_entry()] if entries is None else entries
    torch.save(payload, shard_path)
    sidecar_payload = _base_sidecar(num_entries=len(payload)) if sidecar is None else sidecar
    (cache_root / f"{name}.json").write_text(
        json.dumps(sidecar_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return shard_path


def _assert_validation_fails(cache_root: Path, match: str) -> dict[str, object]:
    validate_stage0_cache, error_type = _cache_api()
    with pytest.raises(error_type, match=match) as exc_info:
        validate_stage0_cache(cache_root)
    return exc_info.value.manifest  # type: ignore[attr-defined]


def test_valid_shard_passes_and_writes_manifest(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    output = tmp_path / "cache_manifest.json"
    _write_shard(cache_root)
    validate_stage0_cache, _ = _cache_api()

    manifest = validate_stage0_cache(cache_root, output=output)

    assert output.exists()
    assert manifest["status"] == "passed"
    assert manifest["cache_root"] == str(cache_root)
    assert manifest["total_entries"] == 1
    assert manifest["duplicate_keys"] == []
    assert manifest["errors"] == []
    shard = manifest["shards"][0]
    assert shard["path"] == str(cache_root / "shard-00000.pt")
    assert shard["sidecar_path"] == str(cache_root / "shard-00000.pt.json")
    assert shard["num_entries"] == 1
    assert shard["hidden_dim"] == 3
    assert shard["total_layers"] == 2
    assert shard["selected_layers"] == [0, 1]
    assert shard["errors"] == []


def test_missing_layer_vectors_fails(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entry = _base_entry()
    del entry["layer_vectors"]
    _write_shard(cache_root, entries=[entry])

    manifest = _assert_validation_fails(cache_root, "layer_vectors")

    assert manifest["status"] == "failed"
    assert "layer_vectors" in manifest["shards"][0]["errors"][0]


def test_wrong_number_of_layers_fails(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entry = _base_entry(layer_vectors=torch.ones((1, 3), dtype=torch.float32))
    _write_shard(cache_root, entries=[entry])

    manifest = _assert_validation_fails(cache_root, "num_selected_layers")

    assert manifest["status"] == "failed"


def test_non_finite_tensor_fails(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entry = _base_entry(
        layer_vectors=torch.tensor(
            [[0.1, float("inf"), 0.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
    )
    _write_shard(cache_root, entries=[entry])

    manifest = _assert_validation_fails(cache_root, "finite")

    assert manifest["status"] == "failed"


def test_duplicate_sample_ids_fail_across_cache_set(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_shard(cache_root, name="shard-00000.pt")
    _write_shard(cache_root, name="shard-00001.pt")

    manifest = _assert_validation_fails(cache_root, "duplicate")

    assert manifest["duplicate_keys"] == [["pope", "encoder_train", "sample-001"]]


def test_non_contiguous_selected_layers_fail_for_full_layer_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_shard(
        cache_root,
        sidecar=_base_sidecar(
            total_layers=3,
            selected_layers=[0, 2],
            num_selected_layers=2,
        ),
    )

    manifest = _assert_validation_fails(cache_root, "selected_layers")

    assert manifest["status"] == "failed"


def test_missing_required_sidecar_metadata_fails(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    sidecar = _base_sidecar()
    del sidecar["model_id"]
    _write_shard(cache_root, sidecar=sidecar)

    manifest = _assert_validation_fails(cache_root, "model_id")

    assert manifest["status"] == "failed"


def test_dict_payload_is_rejected_for_stage0_validation(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    torch.save({"entries": [_base_entry()]}, cache_root / "shard-00000.pt")
    (cache_root / "shard-00000.pt.json").write_text(
        json.dumps(_base_sidecar(), indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = _assert_validation_fails(cache_root, "list of dicts")

    assert manifest["status"] == "failed"


def test_hidden_dim_mismatch_within_shard_fails(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entries = [
        _base_entry(sample_id="sample-001"),
        _base_entry(
            sample_id="sample-002",
            layer_vectors=torch.ones((2, 4), dtype=torch.float32),
        ),
    ]
    _write_shard(
        cache_root,
        entries=entries,
        sidecar=_base_sidecar(num_entries=2),
    )

    manifest = _assert_validation_fails(cache_root, "hidden dim")

    assert manifest["status"] == "failed"


def test_cli_writes_failed_manifest_and_exits_nonzero(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    output = tmp_path / "manifest.json"
    entry = _base_entry()
    del entry["layer_vectors"]
    _write_shard(cache_root, entries=[entry])
    module = _load_script()

    exit_code = module.main(
        [
            "--cache-root",
            str(cache_root),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "layer_vectors" in manifest["errors"][0]
