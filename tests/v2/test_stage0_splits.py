from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from mind.trajectory.splits import SPLIT_NAMES, build_split_manifest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _synthetic_rows(*, groups: int = 50, rows_per_group: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index in range(groups):
        image_id = f"image-{group_index:03d}"
        for row_index in range(rows_per_group):
            sample_index = group_index * rows_per_group + row_index
            label = sample_index % 2
            object_name = f"object-{group_index % 5}"
            rows.append(
                {
                    "dataset_name": "pope",
                    "subset": "popular",
                    "sample_id": f"sample-{sample_index:03d}",
                    "image_id": image_id,
                    "image_path": f"images/{group_index:06d}.jpg",
                    "object_name": object_name,
                    "label": label,
                    "question": f"Is there a {object_name} in the image?",
                    "ignored_extra": "not preserved",
                }
            )
    return rows


def _manifest_for(tmp_path: Path, rows: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    input_records = tmp_path / "input.jsonl"
    _write_jsonl(input_records, rows)
    return build_split_manifest(
        dataset_name="pope",
        subset="popular",
        input_records=input_records,
        seed=kwargs.pop("seed", 20260506),
        ratios=kwargs.pop("ratios", (0.50, 0.20, 0.10, 0.20)),
        group_key=kwargs.pop("group_key", "image_id"),
        **kwargs,
    )


def _ids_by_split(assignments: list[dict[str, object]], key: str) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        by_split[str(row["split"])].add(str(row[key]))
    return by_split


def _assert_no_overlap(by_split: dict[str, set[str]]) -> None:
    splits = list(by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            assert by_split[left].isdisjoint(by_split[right])


def test_build_split_manifest_creates_four_splits_with_grouped_ratios(tmp_path: Path) -> None:
    manifest = _manifest_for(tmp_path, _synthetic_rows(groups=50, rows_per_group=2))

    assert manifest["split_names"] == list(SPLIT_NAMES)
    assert list(manifest["counts_per_split"]) == list(SPLIT_NAMES)
    assert set(manifest["counts_per_split"].values()) == {10, 20, 50}
    assert manifest["counts_per_split"] == {
        "encoder_train": 50,
        "bank": 20,
        "cal": 10,
        "test": 20,
    }

    assignments = manifest["assignments"]
    assert len(assignments) == 100
    assert {row["split"] for row in assignments} == set(SPLIT_NAMES)
    assert manifest["image_id_overlap_validation"] == {"valid": True, "overlaps": {}}
    assert manifest["sample_id_overlap_validation"] == {"valid": True, "overlaps": {}}

    image_ids = _ids_by_split(assignments, "image_id")
    sample_ids = _ids_by_split(assignments, "sample_id")
    _assert_no_overlap(image_ids)
    _assert_no_overlap(sample_ids)

    first_assignment = assignments[0]
    assert set(first_assignment) == {
        "split",
        "dataset_name",
        "subset",
        "sample_id",
        "image_id",
        "image_path",
        "object_name",
        "label",
        "question",
    }


def test_build_split_manifest_is_seed_reproducible(tmp_path: Path) -> None:
    rows = _synthetic_rows(groups=32, rows_per_group=1)

    first = _manifest_for(tmp_path, rows, seed=11)
    second = _manifest_for(tmp_path, rows, seed=11)
    different = _manifest_for(tmp_path, rows, seed=12)

    first_pairs = [(row["sample_id"], row["split"]) for row in first["assignments"]]
    second_pairs = [(row["sample_id"], row["split"]) for row in second["assignments"]]
    different_pairs = [(row["sample_id"], row["split"]) for row in different["assignments"]]
    assert first_pairs == second_pairs
    assert first_pairs != different_pairs


def test_build_split_manifest_rejects_blank_group_key(tmp_path: Path) -> None:
    rows = _synthetic_rows(groups=4, rows_per_group=1)
    rows[2]["image_id"] = " "

    with pytest.raises(ValueError, match="missing or blank group key 'image_id'.*sample-002"):
        _manifest_for(tmp_path, rows)


def test_build_split_manifest_rejects_mismatched_dataset_source(tmp_path: Path) -> None:
    rows = _synthetic_rows(groups=4, rows_per_group=1)
    rows[1]["dataset_name"] = "repope"

    with pytest.raises(ValueError, match="mismatched dataset_name.*repope"):
        _manifest_for(tmp_path, rows)


def test_cli_main_writes_manifest(tmp_path: Path) -> None:
    script_path = Path("scripts/v2/stage0_build_splits.py")
    spec = importlib.util.spec_from_file_location("stage0_build_splits", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_records = tmp_path / "input.jsonl"
    output = tmp_path / "manifest.json"
    _write_jsonl(input_records, _synthetic_rows(groups=10, rows_per_group=1))

    exit_code = module.main(
        [
            "--dataset-name",
            "pope",
            "--subset",
            "popular",
            "--input-records",
            str(input_records),
            "--output",
            str(output),
            "--seed",
            "7",
            "--ratios",
            "0.50",
            "0.20",
            "0.10",
            "0.20",
            "--group-key",
            "image_id",
        ]
    )

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "pope"
    assert manifest["subset"] == "popular"
    assert manifest["input_records"] == str(input_records)
    assert manifest["seed"] == 7
    assert manifest["group_key"] == "image_id"
    assert manifest["ratios"] == [0.5, 0.2, 0.1, 0.2]
    assert manifest["split_names"] == list(SPLIT_NAMES)
    assert sum(manifest["counts_per_split"].values()) == 10
    assert Counter(row["split"] for row in manifest["assignments"]) == Counter(
        manifest["counts_per_split"]
    )


def test_cli_dry_run_builds_manifest_without_writing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path("scripts/v2/stage0_build_splits.py")
    spec = importlib.util.spec_from_file_location("stage0_build_splits_dry", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_records = tmp_path / "input.jsonl"
    output = tmp_path / "manifests" / "split_manifest.json"
    _write_jsonl(input_records, _synthetic_rows(groups=10, rows_per_group=1))

    exit_code = module.main(
        [
            "--dataset-name",
            "pope",
            "--subset",
            "popular",
            "--input-records",
            str(input_records),
            "--output",
            str(output),
            "--seed",
            "7",
            "--ratios",
            "0.50",
            "0.20",
            "0.10",
            "0.20",
            "--group-key",
            "image_id",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert not output.exists()
    assert not output.parent.exists()
    assert "dry_run=true" in captured.out
    manifest = json.loads(captured.out.strip().splitlines()[-1])
    assert manifest["dry_run"] is True
    assert manifest["output"] == str(output)
    assert manifest["manifest"]["dataset_name"] == "pope"
    assert manifest["manifest"]["subset"] == "popular"
    assert manifest["manifest"]["seed"] == 7
    assert sum(manifest["manifest"]["counts_per_split"].values()) == 10
