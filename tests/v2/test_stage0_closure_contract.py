from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

import pytest
import torch

from mind.data import load_object_yes_no_records
from mind.trajectory.audit import run_audit
from mind.trajectory.cache import validate_stage0_cache
from mind.trajectory.dataset import DatasetSpec, discover_known_datasets


def _load_script(path: str, name: str) -> ModuleType:
    script_path = Path(path)
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage0_run(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    models_package = ModuleType("mind.models")
    model_types = ModuleType("mind.models.types")
    model_types.resolve_torch_dtype = lambda _value: torch.float16  # type: ignore[attr-defined]
    extractor = ModuleType("stage0_extract_full_layer_cache")
    extractor.run_extraction = lambda **_kwargs: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mind.models", models_package)
    monkeypatch.setitem(sys.modules, "mind.models.types", model_types)
    monkeypatch.setitem(sys.modules, "stage0_extract_full_layer_cache", extractor)
    return _load_script("scripts/v2/stage0_run.py", name)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _stage0_record(
    sample_id: str,
    *,
    dataset_name: str = "pope",
    subset: str = "popular",
    image_id: str | int = "image-001",
    label: int = 1,
    object_name: str = "cat",
    question: str | None = None,
) -> dict[str, object]:
    question_text = question or f"Is there a {object_name} in the image?"
    return {
        "dataset_name": dataset_name,
        "source_dataset": dataset_name,
        "subset": subset,
        "split": subset,
        "sample_id": sample_id,
        "image_id": image_id,
        "image_path": f"images/{image_id}.jpg",
        "question": question_text,
        "label": label,
        "object_name": object_name,
    }


def _repope_materializer() -> Callable[..., object]:
    try:
        from mind.trajectory.repope import materialize_repope_cache
    except (ImportError, AttributeError) as exc:
        pytest.fail(
            "Expected future helper mind.trajectory.repope.materialize_repope_cache "
            "for Stage 0 RePOPE cache materialization."
        )
    return materialize_repope_cache


def test_discover_known_datasets_lists_full_closure_surface(tmp_path: Path) -> None:
    for dataset_name in ("pope", "repope"):
        for subset in ("popular", "random", "adversarial"):
            _write_jsonl(
                tmp_path
                / "outputs"
                / "round2_2026_04"
                / "normalized"
                / dataset_name
                / f"{subset}.jsonl",
                [],
            )
    _write_jsonl(
        tmp_path
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / "dash-b"
        / "all.jsonl",
        [],
    )

    specs = discover_known_datasets(tmp_path)

    assert [(spec.dataset_name, spec.subset) for spec in specs] == [
        ("pope", "popular"),
        ("pope", "random"),
        ("pope", "adversarial"),
        ("repope", "popular"),
        ("repope", "random"),
        ("repope", "adversarial"),
        ("dash-b", "all"),
    ]


def test_orchestrator_resolves_dash_b_all_dataset_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage0_run("stage0_run_dash_b_config", monkeypatch)
    records_path = (
        tmp_path
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / "dash-b"
        / "all.jsonl"
    )
    _write_jsonl(records_path, [_stage0_record("dash-1", dataset_name="dash-b", subset="all")])

    specs = module.resolve_dataset_specs(
        datasets=["dash-b"],
        subsets=["all"],
        repo_root=tmp_path,
        require_normalized=True,
    )

    assert specs == [DatasetSpec("dash-b", "all", records_path)]


def test_dash_b_raw_directory_records_have_stage0_required_fields(tmp_path: Path) -> None:
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

    records = load_object_yes_no_records(
        dash_b_root,
        subset="all",
        split="all",
        source_dataset="dash-b",
    )
    rows = [asdict(record) for record in records]

    required_fields = {
        "sample_id",
        "image_id",
        "image_path",
        "question",
        "label",
        "object_name",
        "source_dataset",
        "split",
        "subset",
    }
    assert [row["label"] for row in rows] == [0, 1]
    assert all(required_fields <= set(row) for row in rows)
    assert {(row["source_dataset"], row["subset"], row["split"]) for row in rows} == {
        ("dash-b", "all", "all")
    }
    assert rows[0]["question"] == "Can you see a toaster in this image? Please answer only with yes or no."


def _pope_cache_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "model_name": "tiny-model",
        "dataset_name": "pope",
        "source_dataset": "pope",
        "subset": "popular",
        "split": "popular",
        "sample_id": "sample-001",
        "image_id": "image-001",
        "image_path": "images/image-001.jpg",
        "question": "Is there a cat in the image?",
        "label": 1,
        "object_name": "cat",
        "answer_text": "yes",
        "parsed_answer": 1,
        "first_token_logits": torch.tensor([0.0, 3.0, -1.0], dtype=torch.float32),
        "selected_layers": [0, 1],
        "layer_vectors": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        ),
        "full_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    }
    entry.update(overrides)
    return entry


def _pope_sidecar(**overrides: object) -> dict[str, object]:
    sidecar: dict[str, object] = {
        "stage": "v2_stage0",
        "cache_type": "full_layer_hidden_states",
        "model_name": "tiny-model",
        "model_id": "org/tiny-model",
        "model_family": "tiny",
        "dataset_name": "pope",
        "source_dataset": "pope",
        "subset": "popular",
        "split": "popular",
        "total_layers": 2,
        "selected_layers": [0, 1],
        "num_selected_layers": 2,
        "hidden_dim": 3,
        "token_index": -1,
        "max_new_tokens": 1,
        "dtype": "float32",
        "num_entries": 1,
        "script": "synthetic",
        "git_commit": "deadbeef",
        "created_at_utc": "2026-05-07T00:00:00Z",
        "records_path": "pope.jsonl",
        "image_root": "images",
    }
    sidecar.update(overrides)
    return sidecar


def _write_pope_cache(cache_root: Path, *, entry: dict[str, object] | None = None) -> Path:
    shard_path = cache_root / "tiny-model" / "pope" / "popular" / "shard-00000.pt"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([_pope_cache_entry() if entry is None else entry], shard_path)
    Path(str(shard_path) + ".json").write_text(
        json.dumps(_pope_sidecar(), indent=2) + "\n",
        encoding="utf-8",
    )
    return shard_path


def _aggregate_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    totals = {"encoder_train": 0, "bank": 0, "cal": 0, "test": 0}
    for row in rows:
        counts = row[key]
        assert isinstance(counts, dict)
        for split, count in counts.items():
            totals[str(split)] += int(count)
    return totals


def _aggregate_nested_counts(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    totals = {"encoder_train": {}, "bank": {}, "cal": {}, "test": {}}
    for row in rows:
        split_counts = row[key]
        assert isinstance(split_counts, dict)
        for split, counts in split_counts.items():
            assert isinstance(counts, dict)
            split_total = totals[str(split)]
            for value, count in counts.items():
                split_total[str(value)] = split_total.get(str(value), 0) + int(count)
    return totals


def test_repope_materialization_preserves_hidden_tensors_and_updates_metadata(
    tmp_path: Path,
) -> None:
    materialize_repope_cache = _repope_materializer()
    pope_cache_root = tmp_path / "pope-cache"
    output_root = tmp_path / "repope-cache"
    source_entry = _pope_cache_entry()
    _write_pope_cache(pope_cache_root, entry=source_entry)
    pope_records = tmp_path / "pope.jsonl"
    repope_records = tmp_path / "repope.jsonl"
    _write_jsonl(pope_records, [_stage0_record("sample-001", label=1)])
    _write_jsonl(repope_records, [_stage0_record("sample-001", dataset_name="repope", label=0)])

    materialize_repope_cache(
        source_cache_root=pope_cache_root,
        source_records_path=pope_records,
        target_records_path=repope_records,
        output_root=output_root,
        target_dataset_name="repope",
        target_subset="popular",
    )

    shards = sorted(output_root.rglob("*.pt"))
    assert len(shards) == 1
    payload = torch.load(shards[0], weights_only=False)
    assert len(payload) == 1
    materialized = payload[0]
    assert materialized["dataset_name"] == "repope"
    assert materialized["source_dataset"] == "repope"
    assert materialized["subset"] == "popular"
    assert materialized["label"] == 0
    assert torch.equal(materialized["layer_vectors"], source_entry["layer_vectors"])
    assert torch.equal(materialized["full_hidden_states"], source_entry["full_hidden_states"])

    sidecar = json.loads(Path(str(shards[0]) + ".json").read_text(encoding="utf-8"))
    assert sidecar["dataset_name"] == "repope"
    assert sidecar["source_dataset"] == "repope"
    assert sidecar["subset"] == "popular"
    assert sidecar["split"] == "popular"
    assert sidecar["num_entries"] == 1
    assert sidecar["hidden_dim"] == 3
    manifest = validate_stage0_cache(
        output_root,
        dataset_name="repope",
        split="popular",
        model_name="tiny-model",
    )
    assert manifest["status"] == "passed"


@pytest.mark.parametrize(
    ("field", "target_value"),
    [
        ("sample_id", "sample-999"),
        ("image_id", "image-999"),
        ("question", "Is there a dog in the image?"),
        ("object_name", "dog"),
    ],
)
def test_repope_materialization_requires_exact_source_record_match_before_copy(
    tmp_path: Path,
    field: str,
    target_value: object,
) -> None:
    materialize_repope_cache = _repope_materializer()
    pope_cache_root = tmp_path / "pope-cache"
    output_root = tmp_path / "repope-cache"
    _write_pope_cache(pope_cache_root)
    pope_row = _stage0_record("sample-001", label=1)
    repope_row = _stage0_record("sample-001", dataset_name="repope", label=0)
    repope_row[field] = target_value
    if field == "object_name":
        repope_row["question"] = "Is there a dog in the image?"
    _write_jsonl(tmp_path / "pope.jsonl", [pope_row])
    _write_jsonl(tmp_path / "repope.jsonl", [repope_row])

    with pytest.raises(ValueError, match=field):
        materialize_repope_cache(
            source_cache_root=pope_cache_root,
            source_records_path=tmp_path / "pope.jsonl",
            target_records_path=tmp_path / "repope.jsonl",
            output_root=output_root,
            target_dataset_name="repope",
            target_subset="popular",
        )

    assert not output_root.exists() or not list(output_root.rglob("*.pt"))


def test_top_level_split_manifest_indexes_every_requested_dataset_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage0_run("stage0_run_split_index", monkeypatch)
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    for subset in ("popular", "random"):
        _write_jsonl(
            repo_root
            / "outputs"
            / "round2_2026_04"
            / "normalized"
            / "pope"
            / f"{subset}.jsonl",
            [
                _stage0_record(f"{subset}-001", subset=subset, image_id=f"{subset}-image-1"),
                _stage0_record(
                    f"{subset}-002",
                    subset=subset,
                    image_id=f"{subset}-image-2",
                    label=0,
                    object_name="dog",
                ),
            ],
        )
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)
    smoke_paths: list[Path] = []

    def fake_run_extraction(**kwargs: object) -> list[Path]:
        path = (
            output_root
            / "cache"
            / str(kwargs["dataset_name"])
            / str(kwargs["subset"])
            / "shard-00000.pt"
        )
        smoke_paths.append(path)
        return [path]

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(
        module,
        "validate_stage0_cache",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "shards": [
                {
                    "path": str(path),
                    "model_name": "qwen3-vl-8b",
                    "dataset_name": "pope",
                    "source_dataset": "pope",
                    "subset": path.parent.name,
                    "split": path.parent.name,
                    "status": "passed",
                    "errors": [],
                    "num_entries": 2,
                }
                for path in smoke_paths
            ],
            "total_entries": 4,
            "errors": [],
        },
    )

    args = module.parse_args(
        [
            "--output-root",
            str(output_root),
            "--models",
            "qwen3-vl-8b",
            "--datasets",
            "pope",
            "--subsets",
            "popular",
            "random",
            "--smoke-limit",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float16",
        ]
    )

    assert module.run_orchestration(args, repo_root=repo_root) == 0
    top_level = json.loads(
        (output_root / "manifests" / "split_manifest.json").read_text(encoding="utf-8")
    )

    assert top_level["manifest_type"] == "stage0_split_manifest_index"
    summaries = top_level["dataset_manifests"]
    assert [(row["dataset_name"], row["subset"]) for row in summaries] == [
        ("pope", "popular"),
        ("pope", "random"),
    ]
    assert [row["path"] for row in summaries] == [
        str(output_root / "manifests" / "split_manifest_pope_popular.json"),
        str(output_root / "manifests" / "split_manifest_pope_random.json"),
    ]
    for row in summaries:
        assert "counts_per_split" in row
        assert "label_counts_per_split" in row
        assert "object_counts_per_split" in row
        assert row["image_id_overlap_validation"] == {"valid": True, "overlaps": {}}
        assert row["sample_id_overlap_validation"] == {"valid": True, "overlaps": {}}
    assert top_level["counts_per_split"] == _aggregate_counts(summaries, "counts_per_split")
    assert top_level["label_counts_per_split"] == _aggregate_nested_counts(
        summaries,
        "label_counts_per_split",
    )
    assert top_level["object_counts_per_split"] == _aggregate_nested_counts(
        summaries,
        "object_counts_per_split",
    )
    assert top_level["image_id_overlap_validation"]["valid"] is True
    assert top_level["sample_id_overlap_validation"]["valid"] is True
    assert top_level["total_records"] == 4
    assert "assignments" not in top_level


def test_cache_label_balance_uses_parsed_answer_not_answer_text(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(
        dataset_path,
        [
            _stage0_record("gt-yes-parsed-no", label=1),
            _stage0_record("gt-no-parsed-no", image_id="image-002", label=0, object_name="dog"),
            _stage0_record("gt-no-parsed-yes", image_id="image-003", label=0, object_name="dog"),
        ],
    )
    cache_root.mkdir()
    torch.save(
        [
            {
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "gt-yes-parsed-no",
                "parsed_answer": 0,
                "answer_text": "yes",
            },
            {
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "gt-no-parsed-no",
                "parsed_answer": 0,
                "answer_text": "yes",
            },
            {
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "gt-no-parsed-yes",
                "parsed_answer": 1,
                "answer_text": "no",
            },
        ],
        cache_root / "shard-00000.pt",
    )

    result = run_audit(
        [DatasetSpec("pope", "popular", dataset_path)],
        output_root=tmp_path / "outputs",
        cache_root=cache_root,
    )

    row = result.label_balance_rows[0]
    assert row["num_gt_yes"] == 1
    assert row["num_gt_no"] == 2
    assert row["num_parsed_yes"] == 1
    assert row["num_parsed_no"] == 2
    assert row["num_correct"] == 1
    assert row["num_hallucination"] == 1
    assert row["hallucination_rate"] == "0.5"
    assert row["parsed_answer_status"] == "available"


def _matrix_manifest(
    *,
    output_root: Path,
    models: tuple[str, ...],
    datasets: tuple[str, ...],
    subsets: tuple[str, ...],
    omit: set[tuple[str, str, str]] | None = None,
    num_entries: int = 2,
) -> dict[str, object]:
    omitted = omit or set()
    shards: list[dict[str, object]] = []
    for model_name in models:
        for dataset_name in datasets:
            for subset in subsets:
                if (model_name, dataset_name, subset) in omitted:
                    continue
                shards.append(
                    {
                        "path": str(
                            output_root
                            / "cache"
                            / model_name
                            / dataset_name
                            / subset
                            / "shard-00000.pt"
                        ),
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "subset": subset,
                        "split": subset,
                        "status": "passed",
                        "errors": [],
                        "num_entries": num_entries,
                    }
                )
    return {
        "status": "passed",
        "shards": shards,
        "total_entries": num_entries * len(shards),
        "duplicate_keys": [],
        "errors": [],
    }


def _write_matrix_records(repo_root: Path, *, datasets: tuple[str, ...], subsets: tuple[str, ...]) -> None:
    for dataset_name in datasets:
        for subset in subsets:
            _write_jsonl(
                repo_root
                / "outputs"
                / "round2_2026_04"
                / "normalized"
                / dataset_name
                / f"{subset}.jsonl",
                [
                    _stage0_record(
                        f"{dataset_name}-{subset}-001",
                        dataset_name=dataset_name,
                        subset=subset,
                        image_id=f"{dataset_name}-{subset}-image-1",
                    ),
                    _stage0_record(
                        f"{dataset_name}-{subset}-002",
                        dataset_name=dataset_name,
                        subset=subset,
                        image_id=f"{dataset_name}-{subset}-image-2",
                        label=0,
                        object_name="dog",
                    ),
                ],
            )


def _write_matrix_records_with_count(
    repo_root: Path,
    *,
    dataset_name: str,
    subset: str,
    count: int,
) -> None:
    _write_jsonl(
        repo_root
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / dataset_name
        / f"{subset}.jsonl",
        [
            _stage0_record(
                f"{dataset_name}-{subset}-{index:03d}",
                dataset_name=dataset_name,
                subset=subset,
                image_id=f"{dataset_name}-{subset}-image-{index}",
                label=index % 2,
                object_name="cat" if index % 2 else "dog",
            )
            for index in range(1, count + 1)
        ],
    )


def test_full_run_rejects_pope_only_closure_request_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_stage0_run("stage0_run_pope_only_full_closure", monkeypatch)
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    for subset in ("popular", "random", "adversarial"):
        _write_matrix_records_with_count(repo_root, dataset_name="pope", subset=subset, count=1)
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    def fail_stage0_side_effect(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("full-run closure validation must run before Stage 0 side effects")

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_audit", fail_stage0_side_effect)
    monkeypatch.setattr(module, "write_split_manifest", fail_stage0_side_effect)
    monkeypatch.setattr(module, "run_extraction", fail_stage0_side_effect)
    monkeypatch.setattr(module, "validate_stage0_cache", fail_stage0_side_effect)

    args = module.parse_args(
        [
            "--output-root",
            str(output_root),
            "--models",
            "qwen3-vl-8b",
            "internvl3.5-8b",
            "--datasets",
            "pope",
            "--subsets",
            "popular",
            "random",
            "adversarial",
            "--device",
            "cpu",
            "--dtype",
            "float16",
            "--full-run",
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    summary = json.loads(
        (output_root / "manifests" / "stage0_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert "complete Stage 0 closure matrix" in summary["blocking_issues"][0]
    assert "repope/popular" in summary["blocking_issues"][0]
    assert "dash-b/all" in summary["blocking_issues"][0]
    assert "full-run closure validation" not in capsys.readouterr().err


def test_stage0_summary_fails_when_cache_matrix_entry_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage0_run("stage0_run_missing_cache_matrix", monkeypatch)
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    models = ("qwen3-vl-8b", "internvl3.5-8b")
    datasets = ("pope", "repope")
    subsets = ("popular",)
    _write_matrix_records(repo_root, datasets=datasets, subsets=subsets)
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    def fake_run_extraction(**kwargs: object) -> list[Path]:
        return [
            output_root
            / "cache"
            / str(kwargs["model_config_path"]).split("/")[-1].removesuffix(".yaml")
            / str(kwargs["dataset_name"])
            / str(kwargs["subset"])
            / "shard-00000.pt"
        ]

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(
        module,
        "validate_stage0_cache",
        lambda *_args, **_kwargs: _matrix_manifest(
            output_root=output_root,
            models=models,
            datasets=datasets,
            subsets=subsets,
            omit={("internvl3.5-8b", "repope", "popular")},
        ),
    )

    args = module.parse_args(
        [
            "--output-root",
            str(output_root),
            "--models",
            *models,
            "--datasets",
            *datasets,
            "--subsets",
            *subsets,
            "--smoke-limit",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float16",
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    summary = json.loads(
        (output_root / "manifests" / "stage0_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert "missing cache matrix entry" in summary["blocking_issues"][0]
    assert "internvl3.5-8b/repope/popular" in summary["blocking_issues"][0]


def test_stage0_summary_fails_when_cache_entry_count_is_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage0_run("stage0_run_short_cache_count", monkeypatch)
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    models = ("qwen3-vl-8b",)
    datasets = ("pope",)
    subsets = ("popular",)
    _write_matrix_records_with_count(repo_root, dataset_name="pope", subset="popular", count=3)
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(
        module,
        "run_extraction",
        lambda **kwargs: [
            output_root
            / "cache"
            / str(kwargs["dataset_name"])
            / str(kwargs["subset"])
            / "shard-00000.pt"
        ],
    )
    monkeypatch.setattr(
        module,
        "validate_stage0_cache",
        lambda *_args, **_kwargs: _matrix_manifest(
            output_root=output_root,
            models=models,
            datasets=datasets,
            subsets=subsets,
            num_entries=2,
        ),
    )

    args = module.parse_args(
        [
            "--output-root",
            str(output_root),
            "--models",
            *models,
            "--datasets",
            *datasets,
            "--subsets",
            *subsets,
            "--smoke-limit",
            "3",
            "--device",
            "cpu",
            "--dtype",
            "float16",
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    summary = json.loads(
        (output_root / "manifests" / "stage0_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["completed_cache_matrix"] == []
    assert summary["missing_cache_matrix"][0]["expected_num_records"] == 3
    assert "cache matrix count mismatch" in summary["blocking_issues"][0]
    assert "qwen3-vl-8b/pope/popular expected 3 got 2" in summary["blocking_issues"][0]


def test_stage0_summary_passes_when_cache_matrix_entries_all_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_stage0_run("stage0_run_complete_cache_matrix", monkeypatch)
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    models = ("qwen3-vl-8b", "internvl3.5-8b")
    datasets = ("pope", "repope")
    subsets = ("popular",)
    _write_matrix_records(repo_root, datasets=datasets, subsets=subsets)
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(
        module,
        "run_extraction",
        lambda **kwargs: [
            output_root
            / "cache"
            / str(kwargs["dataset_name"])
            / str(kwargs["subset"])
            / "shard-00000.pt"
        ],
    )
    monkeypatch.setattr(
        module,
        "validate_stage0_cache",
        lambda *_args, **_kwargs: _matrix_manifest(
            output_root=output_root,
            models=models,
            datasets=datasets,
            subsets=subsets,
        ),
    )

    args = module.parse_args(
        [
            "--output-root",
            str(output_root),
            "--models",
            *models,
            "--datasets",
            *datasets,
            "--subsets",
            *subsets,
            "--smoke-limit",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float16",
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code == 0
    summary = json.loads(
        (output_root / "manifests" / "stage0_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "passed"
    assert summary["blocking_issues"] == []
