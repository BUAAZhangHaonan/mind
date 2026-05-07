from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_script(path: str, name: str) -> ModuleType:
    script_path = Path(path)
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _record(sample_id: str = "sample-001", image_id: int = 1) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "image_id": image_id,
        "image_path": f"images/{image_id:06d}.jpg",
        "question": "Is there a cat in the image?",
        "label": 1,
        "object_name": "cat",
        "split": "popular",
        "subset": "popular",
        "source_dataset": "pope",
    }


def _closure_record(dataset_name: str, subset: str, sample_id: str, image_id: int) -> dict[str, object]:
    row = _record(sample_id, image_id)
    row["source_dataset"] = dataset_name
    row["subset"] = subset
    row["split"] = subset
    return row


def _write_complete_stage0_config(path: Path, *, output_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "mode: full",
                f"output_root: {output_root}",
                "device: cpu",
                "dtype: float16",
                "models:",
                "  - qwen3-vl-8b",
                "  - internvl3.5-8b",
                "datasets:",
                "  - family: pope",
                "    subsets: [popular, random, adversarial]",
                "  - family: repope",
                "    subsets: [popular, random, adversarial]",
                "  - family: dash-b",
                "    subset: all",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_normalized_closure_record(
    repo_root: Path,
    *,
    dataset_name: str,
    subset: str,
    sample_id: str,
    image_id: int,
) -> None:
    _write_jsonl(
        repo_root
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / dataset_name
        / f"{subset}.jsonl",
        [_closure_record(dataset_name, subset, sample_id, image_id)],
    )


def _write_raw_pope_style_record(repo_root: Path, *, dataset_name: str, subset: str) -> None:
    _write_jsonl(
        repo_root / "data" / dataset_name / f"{subset}.jsonl",
        [
            {
                "question_id": 1,
                "image": "COCO_val2014_000000310196.jpg",
                "text": "Is there a snowboard in the image?",
                "label": "yes",
            }
        ],
    )


def test_extractor_uses_full_layer_range_from_loaded_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "scripts/v2/stage0_extract_full_layer_cache.py",
        "stage0_extract_full_layer_cache",
    )
    records_path = tmp_path / "records.jsonl"
    _write_jsonl(records_path, [_record("sample-001", 1), _record("sample-002", 2)])

    model_config = SimpleNamespace(
        name="tiny-model",
        model_id="org/tiny-model",
        family="qwen_vl",
    )
    dummy_model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=4))
    )

    class DummyWrapper:
        def load_processor(self) -> object:
            return object()

        def load_model(self, *, device: str) -> object:
            assert device == "cpu"
            return dummy_model

    captured: dict[str, object] = {}
    captured["record_batches"] = []

    def fake_extract_prefill_entries(**kwargs: object) -> list[dict[str, object]]:
        captured["selected_layers"] = list(kwargs["selected_layers"])  # type: ignore[index]
        captured["max_new_tokens"] = kwargs["max_new_tokens"]
        records = kwargs["records"]  # type: ignore[index]
        captured["record_batches"].append(records)  # type: ignore[union-attr]
        return [
            {
                "sample_id": record.sample_id,
                "source_dataset": record.source_dataset,
                "subset": record.subset,
                "split": record.split,
                "selected_layers": list(kwargs["selected_layers"]),  # type: ignore[index]
            }
            for record in records  # type: ignore[union-attr]
        ]

    def fake_save_prefill_cache_shard(
        entries: object,
        output_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["entries"] = entries
        captured["output_path"] = output_path
        captured["save_kwargs"] = kwargs
        return {"actual_file_bytes": 1234}

    monkeypatch.setattr(module, "load_yaml_config", lambda *_args: model_config)
    monkeypatch.setattr(module, "create_model_wrapper", lambda _config: DummyWrapper())
    monkeypatch.setattr(module, "extract_prefill_entries", fake_extract_prefill_entries)
    monkeypatch.setattr(module, "estimate_prefill_cache_tensor_bytes", lambda *_args, **_kwargs: 99)
    monkeypatch.setattr(module, "save_prefill_cache_shard", fake_save_prefill_cache_shard)
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "utc_now_iso", lambda: "2026-05-07T00:00:00Z")

    paths = module.run_extraction(
        records_path=records_path,
        model_config_path=tmp_path / "model.yaml",
        output_root=tmp_path / "cache",
        dataset_name="pope",
        subset="popular",
        split="popular",
        image_root=None,
        device="cpu",
        dtype=module.resolve_torch_dtype("float16"),
        max_new_tokens=1,
        token_index=-1,
        limit=0,
        shard_size=128,
        batch_size=1,
    )

    assert captured["selected_layers"] == [0, 1, 2, 3]
    assert captured["max_new_tokens"] == 1
    assert sum(len(batch) for batch in captured["record_batches"]) == 2  # type: ignore[arg-type]
    entries = captured["entries"]  # type: ignore[assignment]
    assert [entry.get("model_name") for entry in entries] == ["tiny-model", "tiny-model"]  # type: ignore[union-attr]
    assert [entry.get("dataset_name") for entry in entries] == ["pope", "pope"]  # type: ignore[union-attr]
    assert paths == [tmp_path / "cache" / "tiny-model" / "pope" / "popular" / "shard-00000.pt"]
    save_kwargs = captured["save_kwargs"]  # type: ignore[assignment]
    assert save_kwargs["cast_all_floating_tensors"] is False  # type: ignore[index]
    metadata = save_kwargs["metadata"]  # type: ignore[index]
    assert metadata["stage"] == "v2_stage0"
    assert metadata["cache_type"] == "full_layer_prefill"
    assert metadata["total_layers"] == 4
    assert metadata["selected_layers"] == [0, 1, 2, 3]
    assert metadata["num_selected_layers"] == 4
    assert metadata["records_path"] == str(records_path)
    assert metadata["source_dataset"] == "pope"


def test_extractor_dry_run_resolves_layers_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(
        "scripts/v2/stage0_extract_full_layer_cache.py",
        "stage0_extract_full_layer_cache_dry",
    )
    records_path = tmp_path / "records.jsonl"
    model_config_path = tmp_path / "model.yaml"
    _write_jsonl(records_path, [_record()])
    model_config_path.write_text(
        "\n".join(
            [
                "name: tiny-model",
                "model_id: org/tiny-model",
                "family: qwen_vl",
                "num_hidden_layers: 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_create_model_wrapper(_config: object) -> object:
        raise AssertionError("dry-run must not load a model wrapper")

    monkeypatch.setattr(module, "create_model_wrapper", fail_create_model_wrapper)

    exit_code = module.main(
        [
            "--records",
            str(records_path),
            "--model-config",
            str(model_config_path),
            "--dataset-name",
            "pope",
            "--subset",
            "popular",
            "--output-root",
            str(tmp_path / "cache"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dry_run=true" in captured.out
    assert "selected_layers=[0, 1, 2]" in captured.out
    assert not (tmp_path / "cache").exists()


def test_extractor_validates_records_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "scripts/v2/stage0_extract_full_layer_cache.py",
        "stage0_extract_full_layer_cache_validate_first",
    )
    records_path = tmp_path / "raw-pope.jsonl"
    _write_jsonl(
        records_path,
        [
            {
                "question_id": 1,
                "image": "COCO_val2014_000000310196.jpg",
                "text": "Is there a snowboard in the image?",
                "label": "yes",
            }
        ],
    )

    def fail_model_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("record validation must happen before model loading")

    monkeypatch.setattr(module, "load_yaml_config", fail_model_load)
    monkeypatch.setattr(module, "create_model_wrapper", fail_model_load)

    with pytest.raises(ValueError, match="missing required record fields"):
        module.run_extraction(
            records_path=records_path,
            model_config_path=tmp_path / "model.yaml",
            output_root=tmp_path / "cache",
            dataset_name="pope",
            subset="popular",
            split="popular",
            image_root=None,
            device="cpu",
            dtype=module.resolve_torch_dtype("float16"),
            max_new_tokens=1,
            token_index=-1,
            limit=0,
            shard_size=128,
            batch_size=1,
        )
    assert not (tmp_path / "cache").exists()


def test_orchestrator_writes_passed_smoke_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    records_path = repo_root / "outputs" / "round2_2026_04" / "normalized" / "pope" / "popular.jsonl"
    _write_jsonl(records_path, [_record("sample-001", 1), _record("sample-002", 2)])
    image_root = repo_root / "data" / "coco" / "val2014"
    image_root.mkdir(parents=True)

    smoke_path = output_root / "cache" / "qwen3-vl-8b" / "pope" / "popular" / "shard-00000.pt"
    extraction_calls: list[dict[str, object]] = []

    def fake_run_extraction(**kwargs: object) -> list[Path]:
        extraction_calls.append(kwargs)
        return [smoke_path]

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
                    "path": str(smoke_path),
                    "model_name": "qwen3-vl-8b",
                    "dataset_name": "pope",
                    "source_dataset": "pope",
                    "subset": "popular",
                    "split": "popular",
                    "status": "passed",
                    "errors": [],
                    "num_entries": 2,
                }
            ],
            "total_entries": 2,
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
    assert len(extraction_calls) == 1
    assert extraction_calls[0]["image_root"] == image_root
    captured = capsys.readouterr()
    expected_full_run_command = (
        "conda run --no-capture-output -n mind-py311 python scripts/v2/stage0_run.py "
        "--config configs/v2/stage0/stage0_complete.yaml"
    )
    assert captured.out == (
        "Smoke Stage 0 passed. Full-run command:\n"
        f"{expected_full_run_command}\n"
    )
    summary_path = output_root / "manifests" / "stage0_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    old_required_fields = {
        "stage",
        "status",
        "git_commit",
        "models_checked",
        "datasets_checked",
        "smoke_cache_paths",
        "audit_outputs",
        "split_manifest",
        "cache_manifest",
        "blocking_issues",
        "next_recommended_commands",
    }
    closure_required_fields = {
        "split_manifests",
        "required_cache_matrix",
        "completed_cache_matrix",
        "missing_cache_matrix",
        "cache_label_balance",
    }
    assert old_required_fields <= set(summary)
    assert closure_required_fields <= set(summary)
    assert summary["stage"] == "stage0"
    assert summary["status"] == "passed"
    assert summary["git_commit"] == "deadbeef"
    assert summary["models_checked"] == ["qwen3-vl-8b"]
    assert summary["datasets_checked"] == ["pope/popular"]
    assert summary["smoke_cache_paths"] == [str(smoke_path)]
    assert summary["audit_outputs"] == [
        str(output_root / "audit" / "dataset_audit.csv"),
        str(output_root / "audit" / "label_balance.csv"),
        str(output_root / "audit" / "object_name_audit.csv"),
        str(output_root / "audit" / "sample_overlap_audit.csv"),
    ]
    assert summary["blocking_issues"] == []
    assert summary["split_manifest"] == str(output_root / "manifests" / "split_manifest.json")
    assert summary["split_manifests"] == [
        str(output_root / "manifests" / "split_manifest_pope_popular.json"),
        str(output_root / "manifests" / "split_manifest.json"),
    ]
    assert summary["cache_manifest"] == str(output_root / "manifests" / "cache_manifest.json")
    expected_cache_matrix = [
        {
            "model_name": "qwen3-vl-8b",
            "dataset_name": "pope",
            "source_dataset": "pope",
            "subset": "popular",
            "expected_num_records": 2,
        }
    ]
    assert summary["required_cache_matrix"] == expected_cache_matrix
    assert summary["completed_cache_matrix"] == [
        {**expected_cache_matrix[0], "cache_num_entries": 2}
    ]
    assert summary["missing_cache_matrix"] == []
    assert summary["cache_label_balance"] == []
    assert summary["next_recommended_commands"] == [expected_full_run_command]
    assert (output_root / "logs" / "stage0_run.log").exists()


def test_cache_label_balance_summary_uses_cache_schema_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_cache_label_schema")
    dataset_path = tmp_path / "pope" / "popular.jsonl"
    cache_root = tmp_path / "cache"
    _write_jsonl(
        dataset_path,
        [
            {**_record("gt-yes", 1), "label": 1},
            {**_record("gt-no", 2), "label": 0},
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
                "sample_id": "gt-yes",
                "parsed_answer": 0,
            },
            {
                "model_name": "model-a",
                "dataset_name": "pope",
                "subset": "popular",
                "sample_id": "gt-no",
                "parsed_answer": 1,
            },
        ],
        shard_path,
    )

    rows = module.cache_label_balance_summary(
        [module.DatasetSpec("pope", "popular", dataset_path)],
        cache_root=cache_root,
        model_specs=[{"model_name": "model-a"}],
    )

    assert rows == [
        {
            "model_name": "model-a",
            "dataset_name": "pope",
            "subset": "popular",
            "num_entries": 2,
            "num_gt_yes": 1,
            "num_gt_no": 1,
            "num_parsed_yes": 1,
            "num_parsed_no": 1,
            "num_parsed_none": 0,
            "num_correct": 0,
            "num_hard_hallucination": 1,
            "num_false_negative_error": 1,
            "num_primary_population": 1,
            "hallucination_rate_in_primary_population": "1",
        }
    ]


def test_orchestrator_audits_discovered_specs_but_extracts_requested_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_audit_scope")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    records_path = repo_root / "outputs" / "round2_2026_04" / "normalized" / "pope" / "popular.jsonl"
    _write_jsonl(records_path, [_record("sample-001", 1), _record("sample-002", 2)])
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    audit_keys: list[tuple[str, str]] = []
    required_keys: list[tuple[str, str]] = []
    extraction_keys: list[tuple[str, str]] = []
    smoke_path = output_root / "cache" / "qwen3-vl-8b" / "pope" / "popular" / "shard-00000.pt"

    def fake_run_audit(specs: object, **_kwargs: object) -> object:
        audit_keys.extend((spec.dataset_name, spec.subset) for spec in specs)  # type: ignore[attr-defined]
        return SimpleNamespace(audit_dir=output_root / "audit")

    def fake_validate_required_datasets(specs: object, required: object) -> None:
        required_keys.extend((spec.dataset_name, spec.subset) for spec in specs)  # type: ignore[attr-defined]
        assert list(required) == required_keys

    def fake_run_extraction(**kwargs: object) -> list[Path]:
        extraction_keys.append((str(kwargs["dataset_name"]), str(kwargs["subset"])))
        return [smoke_path]

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_audit", fake_run_audit)
    monkeypatch.setattr(module, "validate_required_datasets", fake_validate_required_datasets)
    monkeypatch.setattr(module, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(
        module,
        "validate_stage0_cache",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "shards": [
                {
                    "path": str(smoke_path),
                    "model_name": "qwen3-vl-8b",
                    "dataset_name": "pope",
                    "source_dataset": "pope",
                    "subset": "popular",
                    "split": "popular",
                    "status": "passed",
                    "errors": [],
                    "num_entries": 2,
                }
            ],
            "total_entries": 2,
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
    assert audit_keys == [
        ("pope", "popular"),
        ("pope", "random"),
        ("pope", "adversarial"),
        ("repope", "popular"),
        ("repope", "random"),
        ("repope", "adversarial"),
    ]
    assert required_keys == [("pope", "popular")]
    assert extraction_keys == [("pope", "popular")]


def test_make_verify_env_makes_src_importable_without_editable_install() -> None:
    result = subprocess.run(
        ["make", "-n", "verify-env", "PYTHON=python"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PYTHONPATH=" in result.stdout
    assert "src" in result.stdout
    assert "python scripts/verify_env.py" in result.stdout


def test_orchestrator_dry_run_resolves_plan_without_stage0_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_dry")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    records_path = repo_root / "outputs" / "round2_2026_04" / "normalized" / "pope" / "popular.jsonl"
    _write_jsonl(records_path, [_record("sample-001", 1), _record("sample-002", 2)])
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    def fail_stage0_write_or_extraction(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not call write-heavy Stage 0 operations")

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_audit", fail_stage0_write_or_extraction)
    monkeypatch.setattr(module, "write_split_manifest", fail_stage0_write_or_extraction)
    monkeypatch.setattr(module, "run_extraction", fail_stage0_write_or_extraction)
    monkeypatch.setattr(module, "validate_stage0_cache", fail_stage0_write_or_extraction)

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
            "--smoke-limit",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float16",
            "--dry-run",
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code == 0
    assert not output_root.exists()
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["dry_run"] is True
    assert plan["summary"]["status"] == "dry_run"
    assert plan["summary"]["git_commit"] == "deadbeef"
    assert plan["summary"]["models_checked"] == ["qwen3-vl-8b"]
    assert plan["summary"]["datasets_checked"] == ["pope/popular"]
    assert plan["audit_outputs"] == [
        str(output_root / "audit" / "dataset_audit.csv"),
        str(output_root / "audit" / "label_balance.csv"),
        str(output_root / "audit" / "object_name_audit.csv"),
        str(output_root / "audit" / "sample_overlap_audit.csv"),
    ]
    assert plan["split_manifests"] == [
        str(output_root / "manifests" / "split_manifest_pope_popular.json"),
        str(output_root / "manifests" / "split_manifest.json"),
    ]
    assert len(plan["extraction_commands"]) == 1
    assert "--dry-run" in plan["extraction_commands"][0]
    assert "stage0_extract_full_layer_cache.py" in plan["extraction_commands"][0]
    assert plan["summary"]["next_recommended_commands"] == [
        (
            "conda run --no-capture-output -n mind-py311 python scripts/v2/stage0_run.py "
            "--config configs/v2/stage0/stage0_complete.yaml"
        )
    ]


def test_orchestrator_dry_run_plans_repope_and_dash_b_materialization_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_dry_materialization")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    image_root = repo_root / "data" / "coco" / "val2014"
    image_root.mkdir(parents=True)
    (image_root / "COCO_val2014_000000310196.jpg").write_bytes(b"")
    for dataset_name in ("pope", "repope"):
        for subset in ("popular", "random", "adversarial"):
            _write_raw_pope_style_record(repo_root, dataset_name=dataset_name, subset=subset)
    dash_images = repo_root / "data" / "dash_b" / "images"
    dash_images.mkdir(parents=True)
    (dash_images / "dash_benchmark_neg.json").write_text(
        json.dumps({"coco": {"toaster": ["COCO_val2014_000000000314.jpg"]}}),
        encoding="utf-8",
    )
    (dash_images / "dash_benchmark_pos.json").write_text(
        json.dumps({"coco": {"dog": ["COCO_val2014_000000000042.jpg"]}}),
        encoding="utf-8",
    )

    def fail_stage0_write_or_extraction(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run must not call write-heavy Stage 0 operations")

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_audit", fail_stage0_write_or_extraction)
    monkeypatch.setattr(module, "write_split_manifest", fail_stage0_write_or_extraction)
    monkeypatch.setattr(module, "run_extraction", fail_stage0_write_or_extraction)
    monkeypatch.setattr(module, "validate_stage0_cache", fail_stage0_write_or_extraction)

    config_path = tmp_path / "stage0.yaml"
    _write_complete_stage0_config(config_path, output_root=output_root)
    args = module.parse_args(["--config", str(config_path), "--dry-run"])

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code == 0
    assert not output_root.exists()
    repope_output = (
        repo_root
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / "repope"
        / "random.jsonl"
    )
    dash_b_output = (
        repo_root
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / "dash-b"
        / "all.jsonl"
    )
    assert not repope_output.exists()
    assert not dash_b_output.exists()
    plan = json.loads(capsys.readouterr().out)
    materializations = {
        (row["dataset_name"], row["subset"]): row
        for row in plan["materializations"]
    }
    assert len(materializations) == 7
    assert materializations[("repope", "random")] == {
        "dataset_name": "repope",
        "subset": "random",
        "source_path": str(repo_root / "data" / "repope" / "random.jsonl"),
        "output_path": str(repope_output),
        "record_count": 1,
        "status": "planned",
    }
    assert materializations[("dash-b", "all")] == {
        "dataset_name": "dash-b",
        "subset": "all",
        "source_path": str(repo_root / "data" / "dash_b"),
        "output_path": str(dash_b_output),
        "record_count": 2,
        "status": "planned",
    }


def test_orchestrator_writes_failed_summary_for_missing_required_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_failure")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")

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
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    summary = json.loads(
        (output_root / "manifests" / "stage0_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["smoke_cache_paths"] == []
    assert summary["blocking_issues"]
    assert "Required dataset is missing: pope/popular" in summary["blocking_issues"][0]


def test_orchestrator_fails_before_extraction_when_pope_image_root_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_missing_image_root")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    records_path = repo_root / "outputs" / "round2_2026_04" / "normalized" / "pope" / "popular.jsonl"
    _write_jsonl(records_path, [_record("sample-001", 1)])

    def fail_run_extraction(**_kwargs: object) -> list[Path]:
        raise AssertionError("run_extraction must not run without the POPE image root")

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_extraction", fail_run_extraction)

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
        ]
    )

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    summary = json.loads(
        (output_root / "manifests" / "stage0_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["smoke_cache_paths"] == []
    assert "Image root for pope is missing" in summary["blocking_issues"][0]


def test_full_run_materializes_raw_random_before_stage0_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_raw_random")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    normalized_root = repo_root / "outputs" / "round2_2026_04" / "normalized" / "pope"
    _write_normalized_closure_record(
        repo_root,
        dataset_name="pope",
        subset="popular",
        sample_id="sample-001",
        image_id=1,
    )
    _write_normalized_closure_record(
        repo_root,
        dataset_name="pope",
        subset="adversarial",
        sample_id="sample-002",
        image_id=2,
    )
    for dataset_name in ("repope",):
        for index, subset in enumerate(("popular", "random", "adversarial"), start=3):
            _write_normalized_closure_record(
                repo_root,
                dataset_name=dataset_name,
                subset=subset,
                sample_id=f"{dataset_name}-{subset}",
                image_id=index,
            )
    _write_normalized_closure_record(
        repo_root,
        dataset_name="dash-b",
        subset="all",
        sample_id="dash-b-all",
        image_id=6,
    )
    _write_jsonl(
        repo_root / "data" / "pope" / "random.jsonl",
        [
            {
                "question_id": 1,
                "image": "COCO_val2014_000000310196.jpg",
                "text": "Is there a snowboard in the image?",
                "label": "yes",
            }
        ],
    )
    image_root = repo_root / "data" / "coco" / "val2014"
    image_root.mkdir(parents=True)
    (image_root / "COCO_val2014_000000310196.jpg").write_bytes(b"")

    def stop_after_materialization(specs: object, **_kwargs: object) -> object:
        assert (normalized_root / "random.jsonl").exists()
        assert [
            spec.path
            for spec in specs  # type: ignore[attr-defined]
            if spec.dataset_name == "pope" and spec.subset == "random"
        ] == [
            normalized_root / "random.jsonl"
        ]
        raise RuntimeError("stop after raw materialization")

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_audit", stop_after_materialization)
    monkeypatch.setattr(module, "write_split_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "run_extraction", lambda **_kwargs: [])
    monkeypatch.setattr(module, "validate_stage0_cache", lambda *_args, **_kwargs: {})

    config_path = tmp_path / "stage0.yaml"
    _write_complete_stage0_config(config_path, output_root=output_root)
    args = module.parse_args(["--config", str(config_path)])

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    written = json.loads((normalized_root / "random.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert written == {
        "image_id": 310196,
        "image_path": "COCO_val2014_000000310196.jpg",
        "label": 1,
        "object_name": "snowboard",
        "question": "Is there a snowboard in the image?",
        "sample_id": "1",
        "source_dataset": "pope",
        "split": "random",
        "subset": "random",
    }
    captured = capsys.readouterr()
    assert "stop after raw materialization" in captured.err


def test_full_run_fails_on_missing_random_before_stage0_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("scripts/v2/stage0_run.py", "stage0_run_missing_random")
    repo_root = tmp_path / "repo"
    output_root = repo_root / "outputs" / "v2_stage0"
    normalized_root = repo_root / "outputs" / "round2_2026_04" / "normalized" / "pope"
    _write_normalized_closure_record(
        repo_root,
        dataset_name="pope",
        subset="popular",
        sample_id="sample-001",
        image_id=1,
    )
    _write_normalized_closure_record(
        repo_root,
        dataset_name="pope",
        subset="adversarial",
        sample_id="sample-002",
        image_id=2,
    )
    for dataset_name in ("repope",):
        for index, subset in enumerate(("popular", "random", "adversarial"), start=3):
            _write_normalized_closure_record(
                repo_root,
                dataset_name=dataset_name,
                subset=subset,
                sample_id=f"{dataset_name}-{subset}",
                image_id=index,
            )
    _write_normalized_closure_record(
        repo_root,
        dataset_name="dash-b",
        subset="all",
        sample_id="dash-b-all",
        image_id=6,
    )
    (repo_root / "data" / "coco" / "val2014").mkdir(parents=True)

    def fail_stage0_side_effect(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("full-run input validation must run before Stage 0 side effects")

    monkeypatch.setattr(module, "run_environment_check", lambda *_args, **_kwargs: "env ok")
    monkeypatch.setattr(module, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(module, "run_audit", fail_stage0_side_effect)
    monkeypatch.setattr(module, "write_split_manifest", fail_stage0_side_effect)
    monkeypatch.setattr(module, "run_extraction", fail_stage0_side_effect)
    monkeypatch.setattr(module, "validate_stage0_cache", fail_stage0_side_effect)

    config_path = tmp_path / "stage0.yaml"
    _write_complete_stage0_config(config_path, output_root=output_root)
    args = module.parse_args(["--config", str(config_path)])

    exit_code = module.run_orchestration(args, repo_root=repo_root)

    assert exit_code != 0
    assert not output_root.exists()
    captured = capsys.readouterr()
    assert "Normalized extraction-ready dataset is missing for full-run: pope/random" in captured.err
    assert "raw file exists" not in captured.err
