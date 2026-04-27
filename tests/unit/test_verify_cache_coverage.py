from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "verify_cache_coverage.py"
    spec = importlib.util.spec_from_file_location("verify_cache_coverage", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_records(path: Path, sample_ids: list[str]) -> None:
    rows = [
        {
            "sample_id": sample_id,
            "image_id": index,
            "image_path": f"{index}.jpg",
            "question": "Q?",
            "label": index % 2,
            "object_name": "dog",
            "split": "val",
            "subset": "popular",
            "source_dataset": "pope",
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _cache_entry(sample_id: str, *, dim: int = 3) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "selected_layers": [8, 13],
        "layer_vectors": torch.zeros((2, dim), dtype=torch.float32),
    }


def test_verify_cache_coverage_reports_missing_and_duplicate_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    records_path = tmp_path / "records.jsonl"
    cache_root = tmp_path / "cache"
    split_dir = cache_root / "qwen3-vl-8b" / "pope" / "popular"
    split_dir.mkdir(parents=True)
    _write_records(records_path, ["sample-1", "sample-2", "sample-3"])
    torch.save(
        [_cache_entry("sample-1"), _cache_entry("sample-1"), _cache_entry("sample-2")],
        split_dir / "shard-00000.pt",
    )

    exit_code = script.main(
        [
            "--records",
            str(records_path),
            "--cache-root",
            str(cache_root),
            "--model",
            "qwen3-vl-8b",
            "--dataset-name",
            "pope",
            "--split",
            "popular",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "total_records: 3" in output
    assert "cached_entries: 3" in output
    assert "missing_count: 1" in output
    assert "duplicate_count: 1" in output
    assert "missing_ids:" in output
    assert "sample-3" in output
    assert "selected_layers: [[8, 13]]" in output
    assert "vector_dims: [3]" in output
    assert "layer_dim_consistent: true" in output


def test_verify_cache_coverage_accepts_split_dir_and_allow_incomplete(tmp_path: Path) -> None:
    script = _load_script()
    records_path = tmp_path / "records.jsonl"
    split_dir = tmp_path / "already_split"
    split_dir.mkdir()
    _write_records(records_path, ["sample-1", "sample-2"])
    torch.save([_cache_entry("sample-1"), _cache_entry("sample-1", dim=4)], split_dir / "shard-00000.pt")

    exit_code = script.main(
        [
            "--records",
            str(records_path),
            "--cache-root",
            str(split_dir),
            "--model",
            "qwen3-vl-8b",
            "--dataset-name",
            "pope",
            "--split",
            "popular",
            "--allow-incomplete",
        ]
    )

    assert exit_code == 0
