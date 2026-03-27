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
