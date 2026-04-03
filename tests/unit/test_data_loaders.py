from __future__ import annotations

import json
from pathlib import Path

from mind.data import (
    DatasetUnavailableError,
    apply_repope_labels,
    load_hpope_records,
    load_object_yes_no_records,
    load_pope_records,
)


def test_load_pope_records_reads_jsonl_rows(tmp_path: Path) -> None:
    pope_path = tmp_path / "popular.jsonl"
    rows = [
        {
            "sample_id": "popular-1",
            "image": "COCO_val2014_000000000042.jpg",
            "text": "Is there a dog in the image?",
            "label": "yes",
            "object": "dog",
        },
        {
            "sample_id": "popular-2",
            "image": "COCO_val2014_000000000043.jpg",
            "text": "Is there a train in the image?",
            "label": "no",
            "object": "train",
        },
    ]
    pope_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    records = load_pope_records(pope_path, subset="popular", split="val")

    assert [record.sample_id for record in records] == ["popular-1", "popular-2"]
    assert records[0].question == "Is there a dog in the image?"
    assert records[1].label == 0
    assert records[1].object_name == "train"


def test_load_pope_records_extracts_object_name_from_question_when_missing(tmp_path: Path) -> None:
    pope_path = tmp_path / "popular.jsonl"
    rows = [
        {
            "question_id": 1,
            "image": "COCO_val2014_000000310196.jpg",
            "text": "Is there a snowboard in the image?",
            "label": "yes",
        }
    ]
    pope_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    records = load_pope_records(pope_path, subset="popular", split="val")

    assert records[0].sample_id == "1"
    assert records[0].object_name == "snowboard"


def test_apply_repope_labels_rewrites_labels_by_sample_id() -> None:
    records = load_pope_records(
        [
            {
                "sample_id": "sample-1",
                "image": "COCO_val2014_000000000001.jpg",
                "text": "Is there a bus in the image?",
                "label": "yes",
                "object": "bus",
            }
        ],
        subset="popular",
        split="val",
    )
    relabel_rows = [{"sample_id": "sample-1", "label": "no"}]

    relabeled = apply_repope_labels(records, relabel_rows)

    assert relabeled[0].label == 0
    assert relabeled[0].source_dataset == "repope"


def test_load_hpope_records_raises_when_assets_are_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "hpope.json"

    try:
        load_hpope_records(missing_path, subset="attribute", split="val")
    except DatasetUnavailableError as exc:
        assert "H-POPE" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected DatasetUnavailableError for missing H-POPE assets.")


def test_load_object_yes_no_records_supports_generic_rows_without_questions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dash-b.jsonl"
    rows = [
        {
            "id": "dash-1",
            "image_path": "dash_b/COCO_val2014_000000000314.jpg",
            "answer": "no",
            "object_name": "toaster",
        }
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    records = load_object_yes_no_records(
        dataset_path,
        subset="main",
        split="val",
        source_dataset="dash-b",
        question_template="Can you see a {object_name} in this image?",
    )

    assert records[0].sample_id == "dash-1"
    assert records[0].image_id == 314
    assert records[0].question == "Can you see a toaster in this image?"
    assert records[0].label == 0
    assert records[0].source_dataset == "dash-b"


def test_load_object_yes_no_records_flattens_dash_b_directory_layout(tmp_path: Path) -> None:
    dash_b_root = tmp_path / "dash_b"
    images_dir = dash_b_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "dash_benchmark_neg.json").write_text(
        json.dumps(
            {
                "coco": {
                    "toaster": ["COCO_val2014_000000000314.jpg"],
                }
            }
        ),
        encoding="utf-8",
    )
    (images_dir / "dash_benchmark_pos.json").write_text(
        json.dumps(
            {
                "coco": {
                    "dog": ["COCO_val2014_000000000042.jpg"],
                }
            }
        ),
        encoding="utf-8",
    )

    records = load_object_yes_no_records(
        dash_b_root,
        subset="main",
        split="val",
        source_dataset="dash-b",
    )

    assert [record.object_name for record in records] == ["toaster", "dog"]
    assert [record.label for record in records] == [0, 1]
    assert records[0].image_path == "images/neg/coco/toaster/COCO_val2014_000000000314.jpg"
    assert records[1].image_path == "images/pos/coco/dog/COCO_val2014_000000000042.jpg"
    assert records[0].question == "Can you see a toaster in this image? Please answer only with yes or no."
    assert records[1].question == "Can you see a dog in this image? Please answer only with yes or no."
