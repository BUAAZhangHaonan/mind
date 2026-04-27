from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "build_pooled_bank.py"
SPEC = importlib.util.spec_from_file_location("build_pooled_bank", SCRIPT_PATH)
assert SPEC is not None
pooled_bank = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pooled_bank)


def _write_layer(root: Path, model: str, object_name: str, layer: int, values: list[list[float]]) -> None:
    path = root / model / object_name / f"layer-{layer:02d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(values, dtype=torch.float32), path)


def test_build_pooled_bank_concatenates_object_tensors_and_writes_metadata(tmp_path: Path) -> None:
    reference_root = tmp_path / "object_banks"
    output_root = tmp_path / "pooled_banks"
    model = "toy-model"
    _write_layer(reference_root, model, "cat", 0, [[1.0, 0.0], [2.0, 0.0]])
    _write_layer(reference_root, model, "dog", 0, [[0.0, 1.0]])
    _write_layer(reference_root, model, "__shared__", 0, [[9.0, 9.0]])

    summary = pooled_bank.build_pooled_bank(
        reference_root=reference_root,
        output_root=output_root,
        model_name=model,
    )

    assert summary == [{"layer_index": 0, "pooled_count": 3, "source_count": 3, "object_count": 2}]
    pooled = torch.load(output_root / model / "layer-00.pt", weights_only=True)
    torch.testing.assert_close(
        pooled,
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )

    with (output_root / model / "layer-00.metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["object_name"] for row in rows] == ["cat", "cat", "dog"]
    assert [int(row["pooled_index"]) for row in rows] == [0, 1, 2]
    assert [int(row["source_row_index"]) for row in rows] == [0, 1, 0]

    with (output_root / model / "pooled_counts.csv").open(newline="", encoding="utf-8") as handle:
        count_rows = list(csv.DictReader(handle))
    assert count_rows == [
        {"layer_index": "0", "pooled_count": "3", "source_count": "3", "object_count": "2"}
    ]


def test_build_pooled_bank_rejects_layer_dimension_mismatch(tmp_path: Path) -> None:
    reference_root = tmp_path / "object_banks"
    output_root = tmp_path / "pooled_banks"
    model = "toy-model"
    _write_layer(reference_root, model, "cat", 0, [[1.0, 0.0]])
    _write_layer(reference_root, model, "dog", 0, [[0.0, 1.0, 2.0]])

    with pytest.raises(ValueError, match="dimension"):
        pooled_bank.build_pooled_bank(
            reference_root=reference_root,
            output_root=output_root,
            model_name=model,
        )
