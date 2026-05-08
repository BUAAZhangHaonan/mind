from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from mind.trajectory.audit import build_cache_label_balance_rows, run_audit, validate_required_datasets
from mind.trajectory.dataset import DatasetSpec, discover_known_datasets

CACHE_LABEL_BALANCE_COLUMNS = [
    "model_name",
    "dataset_name",
    "subset",
    "num_entries",
    "num_gt_yes",
    "num_gt_no",
    "num_parsed_yes",
    "num_parsed_no",
    "num_parsed_none",
    "num_correct",
    "num_hard_hallucination",
    "num_false_negative_error",
    "num_primary_population",
    "hallucination_rate_in_primary_population",
]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_script(path: str, name: str) -> ModuleType:
    script_path = Path(path)
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_cache_label_balance_is_model_scoped_when_cache_has_multiple_models(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(
        dataset_path,
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
    for model_name, parsed_answer in (("model-a", 1), ("model-b", 0)):
        shard_path = cache_root / model_name / "pope" / "popular" / "shard-00000.pt"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [
                {
                    "model_name": model_name,
                    "dataset_name": "pope",
                    "source_dataset": "pope",
                    "subset": "popular",
                    "split": "popular",
                    "sample_id": "shared",
                    "parsed_answer": parsed_answer,
                    "answer_text": "yes" if parsed_answer == 1 else "no",
                }
            ],
            shard_path,
        )

    rows = build_cache_label_balance_rows(
        [DatasetSpec("pope", "popular", dataset_path)],
        cache_root=cache_root,
        model_names=["model-a", "model-b"],
    )

    by_model = {row["model_name"]: row for row in rows}
    assert set(by_model) == {"model-a", "model-b"}
    assert list(by_model["model-a"]) == CACHE_LABEL_BALANCE_COLUMNS
    assert by_model["model-a"]["num_parsed_yes"] == 1
    assert by_model["model-a"]["num_parsed_no"] == 0
    assert by_model["model-a"]["num_correct"] == 1
    assert by_model["model-a"]["num_hard_hallucination"] == 0
    assert by_model["model-a"]["num_false_negative_error"] == 0
    assert by_model["model-a"]["num_primary_population"] == 0
    assert by_model["model-a"]["hallucination_rate_in_primary_population"] is None
    assert by_model["model-b"]["num_parsed_yes"] == 0
    assert by_model["model-b"]["num_parsed_no"] == 1
    assert by_model["model-b"]["num_correct"] == 0
    assert by_model["model-b"]["num_hard_hallucination"] == 0
    assert by_model["model-b"]["num_false_negative_error"] == 1


def test_cli_writes_cache_label_balance_from_synthetic_cache_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("scripts/stage0_audit_data.py", "stage0_audit_data_cache_balance")
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(
        dataset_path,
        [
            {
                "sample_id": "gt-yes",
                "image_id": 1,
                "image_path": "images/1.jpg",
                "question": "Is there a cat in the image?",
                "label": 1,
                "object_name": "cat",
            },
            {
                "sample_id": "gt-no",
                "image_id": 2,
                "image_path": "images/2.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            },
        ],
    )
    for model_name, parsed_answers in {
        "model-a": {"gt-yes": 1, "gt-no": 0},
        "model-b": {"gt-yes": 0, "gt-no": 1},
    }.items():
        shard_path = cache_root / model_name / "pope" / "popular" / "shard-00000.pt"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [
                {
                    "model_name": model_name,
                    "dataset_name": "pope",
                    "subset": "popular",
                    "sample_id": sample_id,
                    "parsed_answer": parsed_answer,
                    "answer_text": "no" if parsed_answer == 1 else "yes",
                }
                for sample_id, parsed_answer in parsed_answers.items()
            ],
            shard_path,
        )

    monkeypatch.chdir(tmp_path)
    exit_code = module.main(
        [
            "--dataset",
            "pope",
            "popular",
            str(dataset_path),
            "--cache-root",
            str(cache_root),
        ]
    )

    assert exit_code == 0
    cache_label_balance_path = Path("outputs/stage0/audit/cache_label_balance.csv")
    with cache_label_balance_path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == CACHE_LABEL_BALANCE_COLUMNS
    rows = _read_csv(cache_label_balance_path)
    by_model = {row["model_name"]: row for row in rows}
    assert set(by_model) == {"model-a", "model-b"}
    assert list(rows[0]) == CACHE_LABEL_BALANCE_COLUMNS
    assert by_model["model-a"]["num_gt_yes"] == "1"
    assert by_model["model-a"]["num_gt_no"] == "1"
    assert by_model["model-a"]["num_entries"] == "2"
    assert by_model["model-a"]["num_parsed_yes"] == "1"
    assert by_model["model-a"]["num_parsed_no"] == "1"
    assert by_model["model-a"]["num_correct"] == "2"
    assert by_model["model-a"]["num_hard_hallucination"] == "0"
    assert by_model["model-a"]["num_false_negative_error"] == "0"
    assert by_model["model-a"]["num_primary_population"] == "1"
    assert by_model["model-a"]["hallucination_rate_in_primary_population"] == "0"
    assert by_model["model-b"]["num_parsed_yes"] == "1"
    assert by_model["model-b"]["num_parsed_no"] == "1"
    assert by_model["model-b"]["num_correct"] == "0"
    assert by_model["model-b"]["num_hard_hallucination"] == "1"
    assert by_model["model-b"]["num_false_negative_error"] == "1"
    assert by_model["model-b"]["num_primary_population"] == "1"
    assert by_model["model-b"]["hallucination_rate_in_primary_population"] == "1"


def test_cache_label_balance_uses_parsed_answer_and_gt_no_primary_population(
    tmp_path: Path,
) -> None:
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
                "sample_id": "yes-false-negative",
                "image_id": 2,
                "image_path": "images/2.jpg",
                "question": "Is there a cat in the image?",
                "label": 1,
                "object_name": "cat",
            },
            {
                "sample_id": "no-correct",
                "image_id": 3,
                "image_path": "images/3.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            },
            {
                "sample_id": "no-hard-hallucination",
                "image_id": 4,
                "image_path": "images/4.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            },
            {
                "sample_id": "no-unparsed",
                "image_id": 5,
                "image_path": "images/5.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "object_name": "dog",
            },
        ],
    )
    shard_path = cache_root / "model-a" / "pope" / "popular" / "shard-00000.pt"
    shard_path.parent.mkdir(parents=True)
    torch.save(
        [
            {
                "model_name": "model-a",
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "yes-correct",
                "parsed_answer": 1,
                "answer_text": "no",
                "first_token_logits": torch.tensor([10.0, -10.0]),
            },
            {
                "model_name": "model-a",
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "yes-false-negative",
                "parsed_answer": 0,
                "answer_text": "yes",
                "first_token_logits": torch.tensor([-10.0, 10.0]),
            },
            {
                "model_name": "model-a",
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "no-correct",
                "parsed_answer": 0,
                "answer_text": "yes",
                "first_token_logits": torch.tensor([-10.0, 10.0]),
            },
            {
                "model_name": "model-a",
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "no-hard-hallucination",
                "parsed_answer": 1,
                "answer_text": "no",
                "first_token_logits": torch.tensor([10.0, -10.0]),
            },
            {
                "model_name": "model-a",
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "no-unparsed",
                "parsed_answer": None,
                "answer_text": "yes",
                "first_token_logits": torch.tensor([-10.0, 10.0]),
            },
        ],
        shard_path,
    )

    rows = build_cache_label_balance_rows(
        [DatasetSpec("pope", "popular", dataset_path)],
        cache_root=cache_root,
        model_names=["model-a"],
    )

    assert rows == [
        {
            "model_name": "model-a",
            "dataset_name": "pope",
            "subset": "popular",
            "num_entries": 5,
            "num_gt_yes": 2,
            "num_gt_no": 3,
            "num_parsed_yes": 2,
            "num_parsed_no": 2,
            "num_parsed_none": 1,
            "num_correct": 2,
            "num_hard_hallucination": 1,
            "num_false_negative_error": 1,
            "num_primary_population": 3,
            "hallucination_rate_in_primary_population": "0.333333",
        }
    ]


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
    normalized_popular = Path("outputs/stage0/normalized/pope/popular.jsonl")
    raw_random = Path("data/pope/random.jsonl")
    dash_b = Path("outputs/stage0/normalized/dash-b/main.jsonl")
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


def test_known_discovery_audits_raw_only_dash_b_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    dash_b_root = Path("data/dash_b")
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

    specs = discover_known_datasets(Path("."))
    by_key = {(spec.dataset_name, spec.subset): spec for spec in specs}

    assert by_key[("dash-b", "all")].path == dash_b_root
    assert not Path("outputs/stage0/normalized/dash-b/all.jsonl").exists()

    result = run_audit(
        [by_key[("dash-b", "all")]],
        output_root=tmp_path / "outputs" / "stage0",
        cache_root=None,
    )

    assert result.dataset_audit_rows[0]["status"] == "present"
    assert result.dataset_audit_rows[0]["num_records"] == 2
    assert result.label_balance_rows[0]["num_gt_yes"] == 1
    assert result.label_balance_rows[0]["num_gt_no"] == 1


def test_cli_dry_run_reads_and_reports_without_writing_audit_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("scripts/stage0_audit_data.py", "stage0_audit_data_dry")
    dataset_path = tmp_path / "data" / "pope" / "popular.jsonl"
    output_root = tmp_path / "outputs"
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
            }
        ],
    )

    exit_code = module.main(
        [
            "--dataset",
            "pope",
            "popular",
            str(dataset_path),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert not output_root.exists()
    assert "dry_run=true" in captured.out
    summary = json.loads(captured.out.strip().splitlines()[-1])
    assert summary["dry_run"] is True
    assert summary["audit_dir"] == str(output_root / "audit")
    assert summary["datasets_present"] == 1
    assert summary["datasets_missing"] == 0
    assert summary["records"] == 1
