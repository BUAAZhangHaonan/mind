from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from mind.trajectory.stage_a_splits import build_pope_family_split


def _row(
    *,
    model_name: str = "qwen3-vl-8b",
    subset: str = "popular",
    sample_id: str,
    image_id: str,
    label: int = 0,
    parsed_answer: int = 1,
    stage0_split: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": model_name,
        "dataset_name": "pope",
        "subset": subset,
        "sample_id": sample_id,
        "image_id": image_id,
        "object_name": "cat",
        "label": label,
        "parsed_answer": parsed_answer,
    }
    if stage0_split is not None:
        row["stage0_split"] = stage0_split
    return row


def _load_split_script() -> ModuleType:
    script_path = Path("scripts/stage_a_build_family_splits.py")
    spec = importlib.util.spec_from_file_location("stage_a_build_family_splits", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_level_split_has_no_image_id_overlap() -> None:
    entries = [
        _row(sample_id=f"s-{index}", image_id=f"img-{index}")
        for index in range(20)
    ]

    manifest = build_pope_family_split(entries, seed=20260506)

    assert manifest["image_id_overlap_validation"]["valid"] is True


def test_same_image_id_across_subsets_gets_same_split() -> None:
    entries = [
        _row(subset="popular", sample_id="p-1", image_id="shared"),
        _row(subset="random", sample_id="r-1", image_id="shared"),
        _row(subset="adversarial", sample_id="a-1", image_id="other"),
        _row(subset="popular", sample_id="p-2", image_id="third"),
    ]

    manifest = build_pope_family_split(entries, seed=20260506)
    assignments = manifest["assignments"]
    shared_splits = {
        row["split"]
        for row in assignments
        if row["image_id"] == "shared"
    }

    assert len(shared_splits) == 1


def test_conflicting_stage0_subset_split_is_detected_and_reported() -> None:
    entries = [
        _row(subset="popular", sample_id="p-1", image_id="shared", stage0_split="encoder_train"),
        _row(subset="random", sample_id="r-1", image_id="shared", stage0_split="test"),
        _row(subset="adversarial", sample_id="a-1", image_id="other", stage0_split="test"),
    ]

    manifest = build_pope_family_split(entries, seed=20260506)

    assert manifest["stage0_split_conflict_report"]["num_conflicting_image_ids"] == 1
    assert "shared" in manifest["stage0_split_conflict_report"]["conflicts"]
    assert manifest["stage0_split_conflict_report"]["stage_a_action"] == (
        "ignored Stage 0 per-subset assignments and built a new POPE-family split"
    )


def test_model_specific_entries_are_handled_correctly() -> None:
    entries = [
        _row(model_name="qwen3-vl-8b", sample_id="q-1", image_id="img-1"),
        _row(model_name="internvl3.5-8b", sample_id="i-1", image_id="img-1"),
        _row(model_name="internvl3.5-8b", sample_id="i-2", image_id="img-2"),
    ]

    manifest = build_pope_family_split(entries, seed=20260506)

    assert manifest["counts_per_model"]["qwen3-vl-8b"] == 1
    assert manifest["counts_per_model"]["internvl3.5-8b"] == 2
    assert manifest["sample_id_overlap_validation"]["valid"] is True


def test_split_builder_rejects_non_pope_dataset_name(tmp_path: Path) -> None:
    module = _load_split_script()

    with pytest.raises(ValueError, match="Stage A split builder only accepts dataset_names=\\['pope'\\]"):
        module.build_family_splits(
            stage0_root=tmp_path / "stage0",
            output_root=tmp_path / "stageA",
            dataset_names=("repope",),
            write_output=False,
        )


def test_split_builder_rejects_non_stage_a_pope_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_split_script()
    monkeypatch.setattr(
        module,
        "stream_stage0_cache_entries",
        lambda *_, **__: [
            _row(subset="popular", sample_id="p-1", image_id="img-1"),
            _row(subset="other", sample_id="o-1", image_id="img-2"),
        ],
    )

    with pytest.raises(ValueError, match="unsupported Stage A POPE subsets: other"):
        module.build_family_splits(
            stage0_root=tmp_path / "stage0",
            output_root=tmp_path / "stageA",
            dataset_names=("pope",),
            write_output=False,
        )
