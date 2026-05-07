from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from mind.trajectory.audit import run_audit, validate_required_datasets
from mind.trajectory.dataset import DatasetSpec, discover_known_datasets


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_run_audit_writes_counts_and_missing_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "pope" / "popular.jsonl"
    missing_path = tmp_path / "data" / "pope" / "random.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "sample_id": "s1",
                "image_id": 1,
                "image_path": "images/000001.jpg",
                "question": "Is there a cat in the image?",
                "label": "yes",
                "object_name": "cat",
            },
            {
                "sample_id": "s2",
                "image_id": 2,
                "image_path": "images/000002.jpg",
                "question": "Is there a dog in the image?",
                "label": "no",
            },
            {
                "sample_id": "s3",
                "image_id": 3,
                "image_path": "images/000003.jpg",
                "question": "What is shown here?",
                "label": 1,
            },
            {
                "sample_id": "s4",
                "image_id": 4,
                "image_path": "",
                "question": "",
                "label": 0,
                "object_name": "",
            },
            {
                "sample_id": "s5",
                "image_id": 5,
                "image_path": "images/000005.jpg",
                "question": "Is there a bird in the image?",
                "label": "maybe",
                "object_name": "bird",
            },
        ],
    )

    result = run_audit(
        [
            DatasetSpec("pope", "popular", dataset_path),
            DatasetSpec("pope", "random", missing_path),
        ],
        output_root=tmp_path / "outputs",
        cache_root=None,
    )

    dataset_rows = _read_csv(result.audit_dir / "dataset_audit.csv")
    present = next(row for row in dataset_rows if row["subset"] == "popular")
    missing = next(row for row in dataset_rows if row["subset"] == "random")
    assert present["status"] == "present"
    assert present["num_records"] == "5"
    assert present["num_label_yes"] == "2"
    assert present["num_label_no"] == "2"
    assert present["num_missing_image_path"] == "1"
    assert present["num_missing_question"] == "1"
    assert present["num_unknown_object"] == "2"
    assert present["unique_objects"] == "4"
    assert present["unique_images"] == "5"
    assert present["num_duplicate_sample_id"] == "0"
    assert present["num_null_required_fields"] == "3"
    assert present["num_invalid_label"] == "1"
    assert missing["status"] == "missing"
    assert missing["num_records"] == "0"
    assert missing["unique_images"] == "0"
    assert missing["num_duplicate_sample_id"] == "0"
    assert missing["num_null_required_fields"] == "0"
    assert missing["num_invalid_label"] == "0"

    label_rows = _read_csv(result.audit_dir / "label_balance.csv")
    label_present = next(row for row in label_rows if row["subset"] == "popular")
    label_missing = next(row for row in label_rows if row["subset"] == "random")
    assert label_present["num_gt_yes"] == "2"
    assert label_present["num_gt_no"] == "2"
    assert label_present["parsed_answer_status"] == "not_available_before_cache"
    assert label_present["num_parsed_yes"] == ""
    assert label_missing["status"] == "missing"
    assert label_missing["parsed_answer_status"] == "missing"

    object_rows = _read_csv(result.audit_dir / "object_name_audit.csv")
    unknown = next(row for row in object_rows if row["object_name"] == "unknown")
    assert unknown["num_records"] == "2"


def test_overlap_audit_compares_pope_and_repope_by_required_keys(tmp_path: Path) -> None:
    pope_path = tmp_path / "pope" / "popular.jsonl"
    repope_path = tmp_path / "repope" / "popular.jsonl"
    _write_jsonl(
        pope_path,
        [
            {
                "sample_id": "p1",
                "image_id": 1,
                "image_path": "images/a.jpg",
                "question": "Is there a cat in the image?",
                "label": "yes",
                "object_name": "cat",
            },
            {
                "sample_id": "p2",
                "image_id": 2,
                "image_path": "images/b.jpg",
                "question": "Is there a dog in the image?",
                "label": "no",
                "object_name": "dog",
            },
        ],
    )
    _write_jsonl(
        repope_path,
        [
            {
                "sample_id": "p1",
                "image_id": 9,
                "image_path": "images/z.jpg",
                "question": "Is there a bird in the image?",
                "label": "yes",
                "object_name": "bird",
            },
            {
                "sample_id": "r2",
                "image_id": 1,
                "image_path": "images/other.jpg",
                "question": "Is there a cat in the image?",
                "label": "no",
                "object_name": "cat",
            },
            {
                "sample_id": "p2",
                "image_id": 2,
                "image_path": "images/b.jpg",
                "question": "Is there a dog in the image?",
                "label": "yes",
                "object_name": "dog",
            },
        ],
    )

    result = run_audit(
        [
            DatasetSpec("pope", "popular", pope_path),
            DatasetSpec("repope", "popular", repope_path),
        ],
        output_root=tmp_path / "outputs",
        cache_root=None,
    )

    rows = _read_csv(result.audit_dir / "sample_overlap_audit.csv")
    counts = {row["overlap_key"]: row["overlap_count"] for row in rows}
    assert counts == {
        "sample_id": "2",
        "image_id": "2",
        "image_path": "1",
        "image_id_object_name": "2",
        "sample_id_object_name": "1",
    }
    assert {row["left_dataset"] for row in rows} == {"pope"}
    assert {row["right_dataset"] for row in rows} == {"repope"}


def test_label_balance_uses_synthetic_cache_shard(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(
        dataset_path,
        [
            {
                "sample_id": "yes-correct",
                "image_id": 1,
                "image_path": "images/1.jpg",
                "question": "Is there a cat in the image?",
                "label": 1,
                "object_name": "cat",
            },
            {
                "sample_id": "no-hallucinated",
                "image_id": 2,
                "image_path": "images/2.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            },
            {
                "sample_id": "no-unparsed",
                "image_id": 3,
                "image_path": "images/3.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            },
        ],
    )
    cache_root.mkdir()
    (cache_root / "ignored.pt").mkdir()
    (cache_root / "notes.txt").write_text("not a torch shard", encoding="utf-8")
    torch.save(
        [
            {
                "sample_id": "yes-correct",
                "parsed_answer": 1,
                "answer_text": "yes",
                "first_token_logits": torch.tensor([0.0, 5.0, -1.0]),
            },
            {
                "sample_id": "no-hallucinated",
                "parsed_answer": 1,
                "answer_text": "yes",
                "first_token_logits": torch.tensor([0.0, 5.0, -1.0]),
            },
            {
                "sample_id": "no-unparsed",
                "parsed_answer": None,
                "answer_text": "maybe",
                "first_token_logits": torch.tensor([0.0, 0.0, 0.0]),
            },
        ],
        cache_root / "shard-00000.pt",
    )

    result = run_audit(
        [DatasetSpec("pope", "popular", dataset_path)],
        output_root=tmp_path / "outputs",
        cache_root=cache_root,
    )

    label_row = _read_csv(result.audit_dir / "label_balance.csv")[0]
    assert label_row["parsed_answer_status"] == "available"
    assert label_row["num_parsed_yes"] == "2"
    assert label_row["num_parsed_no"] == "0"
    assert label_row["num_parsed_none"] == "1"
    assert label_row["num_correct"] == "1"
    assert label_row["num_hallucination"] == "1"
    assert label_row["hallucination_rate"] == "0.5"

    object_counts = {
        row["object_name"]: row for row in _read_csv(result.audit_dir / "object_name_audit.csv")
    }
    assert object_counts["cat"]["num_correct"] == "1"
    assert object_counts["dog"]["num_hallucination"] == "1"


def test_cache_answers_match_dataset_subset_when_sample_ids_collide(tmp_path: Path) -> None:
    pope_path = tmp_path / "pope" / "popular.jsonl"
    repope_path = tmp_path / "repope" / "popular.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(
        pope_path,
        [
            {
                "sample_id": "shared",
                "image_id": 1,
                "image_path": "images/1.jpg",
                "question": "Is there a cat in the image?",
                "label": 1,
                "object_name": "cat",
            }
        ],
    )
    _write_jsonl(
        repope_path,
        [
            {
                "sample_id": "shared",
                "image_id": 2,
                "image_path": "images/2.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            }
        ],
    )
    cache_root.mkdir()
    torch.save(
        [
            {
                "dataset_name": "repope",
                "subset": "popular",
                "sample_id": "shared",
                "parsed_answer": 0,
                "answer_text": "no",
            },
            {
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "shared",
                "parsed_answer": 1,
                "answer_text": "yes",
            },
        ],
        cache_root / "shard-00000.pt",
    )

    result = run_audit(
        [
            DatasetSpec("pope", "popular", pope_path),
            DatasetSpec("repope", "popular", repope_path),
        ],
        output_root=tmp_path / "outputs",
        cache_root=cache_root,
    )

    label_rows = {
        (row["dataset_name"], row["subset"]): row
        for row in _read_csv(result.audit_dir / "label_balance.csv")
    }
    assert label_rows[("pope", "popular")]["num_parsed_yes"] == "1"
    assert label_rows[("pope", "popular")]["num_parsed_no"] == "0"
    assert label_rows[("repope", "popular")]["num_parsed_yes"] == "0"
    assert label_rows[("repope", "popular")]["num_parsed_no"] == "1"


def test_explicit_bad_cache_root_fails_clearly(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    _write_jsonl(dataset_path, [])

    with pytest.raises(FileNotFoundError, match="Cache root does not exist"):
        run_audit(
            [DatasetSpec("pope", "popular", dataset_path)],
            output_root=tmp_path / "outputs",
            cache_root=tmp_path / "missing-cache",
        )

    file_cache_root = tmp_path / "cache-file"
    file_cache_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="Cache root is not a directory"):
        run_audit(
            [DatasetSpec("pope", "popular", dataset_path)],
            output_root=tmp_path / "outputs",
            cache_root=file_cache_root,
        )


def test_non_object_jsonl_row_reports_path_and_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps({"sample_id": "ok", "label": "yes"}) + "\n"
        + json.dumps(["not", "an", "object"])
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"{dataset_path}.*line 2.*JSON object"):
        run_audit(
            [DatasetSpec("pope", "popular", dataset_path)],
            output_root=tmp_path / "outputs",
            cache_root=None,
        )


def test_required_dataset_validation_and_known_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    normalized_popular = Path("outputs/round2_2026_04/normalized/pope/popular.jsonl")
    raw_random = Path("data/pope/random.jsonl")
    dash_b = Path("outputs/round2_2026_04/normalized/dash-b/main.jsonl")
    _write_jsonl(normalized_popular, [])
    _write_jsonl(raw_random, [])
    _write_jsonl(dash_b, [])

    specs = discover_known_datasets(Path("."))
    by_key = {(spec.dataset_name, spec.subset): spec for spec in specs}

    assert by_key[("pope", "popular")].path == normalized_popular
    assert by_key[("pope", "random")].path == raw_random
    assert by_key[("dash-b", "main")].path == dash_b
    assert ("pope", "adversarial") in by_key
    assert not by_key[("pope", "adversarial")].path.exists()

    with pytest.raises(FileNotFoundError, match="Required dataset is missing: pope/adversarial"):
        validate_required_datasets(specs, [("pope", "adversarial")])
