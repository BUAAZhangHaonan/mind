from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_script():
    path = Path("scripts/asset_materialize_stage0_normalized.py")
    spec = importlib.util.spec_from_file_location("asset_materialize_stage0_normalized", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(sample_id: str = "s0", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "image_id": 7,
        "image_path": "data/coco/val2014/demo.jpg",
        "question": "Is there a dog in the image?",
        "label": 1,
        "object_name": "dog",
        "source_dataset": "pope",
        "split": "popular",
        "subset": "popular",
        "dataset_name": "pope",
        "prompt_template_id": "qwen",
        "layer_vectors": torch.ones((2, 3)),
        "first_token_logits": torch.ones(4),
        "answer_text": "Yes",
        "parsed_answer": 1,
        "selected_layers": [0, 1],
        "model_name": "qwen3-vl-8b",
        "model_family": "qwen_vl",
        "token_index": 3,
        "hidden_dim": 3,
        "total_layers": 2,
    }
    row.update(overrides)
    return row


def _write_cache(root: Path, model: str, dataset: str, subset: str, entries: list[dict[str, object]]) -> None:
    path = root / "cache" / model / dataset / subset
    path.mkdir(parents=True, exist_ok=True)
    shard = path / "shard-00000.pt"
    torch.save(entries, shard)
    (path / "shard-00000.pt.json").write_text(
        json.dumps({"num_entries": len(entries), "path": str(shard)}) + "\n",
        encoding="utf-8",
    )


def test_materialization_writes_normalized_jsonl_and_excludes_model_fields(tmp_path: Path) -> None:
    module = _load_script()
    stage0_root = tmp_path / "stage0"
    output_root = tmp_path / "assets"
    qwen_entries = [_entry("s0"), _entry("s1", image_id=8)]
    internvl_entries = [dict(row, model_name="internvl3.5-8b") for row in qwen_entries]
    _write_cache(stage0_root, "qwen3-vl-8b", "pope", "popular", qwen_entries)
    _write_cache(stage0_root, "internvl3.5-8b", "pope", "popular", internvl_entries)

    result = module.materialize_dataset_subset(
        stage0_root=stage0_root,
        output_root=output_root,
        dataset_name="pope",
        subset="popular",
        overwrite=False,
    )

    assert result["status"] == "written"
    output_path = stage0_root / "normalized" / "pope" / "popular.jsonl"
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert "layer_vectors" not in rows[0]
    assert "first_token_logits" not in rows[0]
    assert "answer_text" not in rows[0]
    assert "model_name" not in rows[0]
    assert rows[0]["sample_id"] == "s0"
    assert rows[0]["dataset_name"] == "pope"
    assert rows[0]["prompt_template_id"] == "qwen"


def test_cross_model_mismatch_fails(tmp_path: Path) -> None:
    module = _load_script()
    stage0_root = tmp_path / "stage0"
    output_root = tmp_path / "assets"
    _write_cache(stage0_root, "qwen3-vl-8b", "pope", "popular", [_entry("s0")])
    _write_cache(
        stage0_root,
        "internvl3.5-8b",
        "pope",
        "popular",
        [_entry("s0", question="Different question?")],
    )

    result = module.materialize_dataset_subset(
        stage0_root=stage0_root,
        output_root=output_root,
        dataset_name="pope",
        subset="popular",
        overwrite=False,
    )

    assert result["status"] == "failed_mismatched_records"
    assert int(result["num_mismatches"]) == 1
    assert Path(str(result["mismatch_report_path"])).is_file()


def test_missing_sample_in_one_model_fails(tmp_path: Path) -> None:
    module = _load_script()
    stage0_root = tmp_path / "stage0"
    output_root = tmp_path / "assets"
    _write_cache(stage0_root, "qwen3-vl-8b", "pope", "popular", [_entry("s0"), _entry("s1")])
    _write_cache(stage0_root, "internvl3.5-8b", "pope", "popular", [_entry("s0")])

    result = module.materialize_dataset_subset(
        stage0_root=stage0_root,
        output_root=output_root,
        dataset_name="pope",
        subset="popular",
        overwrite=False,
    )

    assert result["status"] == "failed_mismatched_records"
    assert int(result["num_mismatches"]) == 1


def test_existing_identical_file_is_preserved(tmp_path: Path) -> None:
    module = _load_script()
    stage0_root = tmp_path / "stage0"
    output_root = tmp_path / "assets"
    entries = [_entry("s0")]
    _write_cache(stage0_root, "qwen3-vl-8b", "pope", "popular", entries)
    _write_cache(stage0_root, "internvl3.5-8b", "pope", "popular", entries)
    first = module.materialize_dataset_subset(
        stage0_root=stage0_root,
        output_root=output_root,
        dataset_name="pope",
        subset="popular",
        overwrite=False,
    )
    output_path = Path(str(first["output_path"]))
    before = output_path.read_text(encoding="utf-8")

    second = module.materialize_dataset_subset(
        stage0_root=stage0_root,
        output_root=output_root,
        dataset_name="pope",
        subset="popular",
        overwrite=False,
    )

    assert second["status"] == "already_exists_identical"
    assert output_path.read_text(encoding="utf-8") == before


def test_existing_different_file_fails_without_overwrite(tmp_path: Path) -> None:
    module = _load_script()
    stage0_root = tmp_path / "stage0"
    output_root = tmp_path / "assets"
    entries = [_entry("s0")]
    _write_cache(stage0_root, "qwen3-vl-8b", "pope", "popular", entries)
    _write_cache(stage0_root, "internvl3.5-8b", "pope", "popular", entries)
    output_path = stage0_root / "normalized" / "pope" / "popular.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"sample_id":"different"}\n', encoding="utf-8")

    result = module.materialize_dataset_subset(
        stage0_root=stage0_root,
        output_root=output_root,
        dataset_name="pope",
        subset="popular",
        overwrite=False,
    )

    assert result["status"] == "failed_existing_file_differs"
    assert output_path.read_text(encoding="utf-8") == '{"sample_id":"different"}\n'
