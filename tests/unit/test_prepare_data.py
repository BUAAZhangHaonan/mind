from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT_PATH)
prepare_data = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_data)


def test_prepare_data_normalize_pope_writes_canonical_jsonl(tmp_path: Path) -> None:
    source_path = tmp_path / "popular.jsonl"
    output_path = tmp_path / "normalized.jsonl"
    source_rows = [
        {
            "sample_id": "popular-1",
            "image": "COCO_val2014_000000000101.jpg",
            "text": "Is there a dog in the image?",
            "label": "yes",
            "object": "dog",
        }
    ]
    source_path.write_text(
        "\n".join(json.dumps(row) for row in source_rows) + "\n",
        encoding="utf-8",
    )

    exit_code = prepare_data.main(
        [
            "normalize-pope",
            "--source",
            str(source_path),
            "--output",
            str(output_path),
            "--subset",
            "popular",
            "--split",
            "val",
        ]
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert written == [
        {
            "sample_id": "popular-1",
            "image_id": 101,
            "image_path": "COCO_val2014_000000000101.jpg",
            "question": "Is there a dog in the image?",
            "label": 1,
            "object_name": "dog",
            "split": "val",
            "subset": "popular",
            "source_dataset": "pope",
        }
    ]


def test_prepare_data_normalize_pope_allows_source_dataset_override(tmp_path: Path) -> None:
    source_path = tmp_path / "popular.jsonl"
    output_path = tmp_path / "normalized.jsonl"
    source_rows = [
        {
            "sample_id": "popular-1",
            "image": "COCO_val2014_000000000101.jpg",
            "text": "Is there a dog in the image?",
            "label": "yes",
            "object": "dog",
        }
    ]
    source_path.write_text(
        "\n".join(json.dumps(row) for row in source_rows) + "\n",
        encoding="utf-8",
    )

    exit_code = prepare_data.main(
        [
            "normalize-pope",
            "--source",
            str(source_path),
            "--output",
            str(output_path),
            "--subset",
            "popular",
            "--split",
            "val",
            "--source-dataset",
            "repope",
        ]
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert written[0]["source_dataset"] == "repope"


def test_prepare_data_build_reference_can_read_allowed_objects_from_normalized_records(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.json"
    allowed_objects_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "reference.json"

    instances_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "000000000001.jpg"},
                    {"id": 2, "file_name": "000000000002.jpg"},
                ],
                "categories": [
                    {"id": 1, "name": "dog"},
                    {"id": 2, "name": "bus"},
                ],
                "annotations": [
                    {"id": 11, "image_id": 1, "category_id": 1},
                    {"id": 12, "image_id": 2, "category_id": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    allowed_objects_path.write_text(
        "\n".join(
            [
                json.dumps({"object_name": "dog"}),
                json.dumps({"object_name": "dog"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = prepare_data.main(
        [
            "build-reference",
            "--instances-json",
            str(instances_path),
            "--output",
            str(output_path),
            "--allowed-objects-from",
            str(allowed_objects_path),
        ]
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert written == [{"file_name": "000000000001.jpg", "image_id": 1, "object_names": ["dog"]}]


def test_prepare_data_normalize_object_yes_no_writes_canonical_jsonl(tmp_path: Path) -> None:
    source_path = tmp_path / "dash-b.jsonl"
    output_path = tmp_path / "normalized.jsonl"
    source_rows = [
        {
            "id": "dash-1",
            "image_path": "dash_b/COCO_val2014_000000000314.jpg",
            "answer": "no",
            "object_name": "toaster",
        }
    ]
    source_path.write_text(
        "\n".join(json.dumps(row) for row in source_rows) + "\n",
        encoding="utf-8",
    )

    exit_code = prepare_data.main(
        [
            "normalize-object-yes-no",
            "--source",
            str(source_path),
            "--output",
            str(output_path),
            "--subset",
            "main",
            "--split",
            "val",
            "--source-dataset",
            "dash-b",
            "--question-template",
            "Can you see a {object_name} in this image?",
        ]
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert written == [
        {
            "sample_id": "dash-1",
            "image_id": 314,
            "image_path": "dash_b/COCO_val2014_000000000314.jpg",
            "question": "Can you see a toaster in this image?",
            "label": 0,
            "object_name": "toaster",
            "split": "val",
            "subset": "main",
            "source_dataset": "dash-b",
        }
    ]


def test_prepare_data_normalize_object_yes_no_supports_dash_b_directory(tmp_path: Path) -> None:
    dash_b_root = tmp_path / "dash_b"
    images_dir = dash_b_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "dash_benchmark_neg.json").write_text(
        json.dumps({"coco": {"toaster": ["COCO_val2014_000000000314.jpg"]}}),
        encoding="utf-8",
    )
    (images_dir / "dash_benchmark_pos.json").write_text(
        json.dumps({"coco": {"dog": ["COCO_val2014_000000000042.jpg"]}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "normalized.jsonl"

    exit_code = prepare_data.main(
        [
            "normalize-object-yes-no",
            "--source",
            str(dash_b_root),
            "--output",
            str(output_path),
            "--subset",
            "main",
            "--split",
            "val",
            "--source-dataset",
            "dash-b",
        ]
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert [row["image_path"] for row in written] == [
        "images/neg/coco/toaster/COCO_val2014_000000000314.jpg",
        "images/pos/coco/dog/COCO_val2014_000000000042.jpg",
    ]
    assert [row["question"] for row in written] == [
        "Can you see a toaster in this image? Please answer only with yes or no.",
        "Can you see a dog in this image? Please answer only with yes or no.",
    ]
