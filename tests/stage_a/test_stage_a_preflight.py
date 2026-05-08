from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch


MODELS = ("qwen3-vl-8b", "internvl3.5-8b")
DATASETS = {
    "pope": ("popular", "random", "adversarial"),
    "repope": ("popular", "random", "adversarial"),
    "dash-b": ("all",),
}


def _load_script() -> ModuleType:
    script_path = Path("scripts/stage_a_preflight.py")
    spec = importlib.util.spec_from_file_location("stage_a_preflight", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(model_name: str, dataset_name: str, subset: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "source_dataset": dataset_name,
        "subset": subset,
        "split": subset,
        "sample_id": f"{dataset_name}-{subset}-1",
        "image_id": f"{dataset_name}-{subset}-image",
        "image_path": "images/1.jpg",
        "question": "Is there a cat?",
        "label": 0,
        "object_name": "cat",
        "answer_text": "yes",
        "parsed_answer": 1,
        "first_token_logits": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "selected_layers": [0, 1],
        "layer_vectors": torch.ones((2, 3), dtype=torch.float32),
    }


def _write_stage0(root: Path) -> None:
    manifest_dir = root / "manifests"
    audit_dir = root / "audit"
    cache_root = root / "cache"
    manifest_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    required = []
    completed = []
    shards = []
    balance_rows = []
    for model_name in MODELS:
        for dataset_name, subsets in DATASETS.items():
            for subset in subsets:
                row = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "source_dataset": dataset_name,
                    "subset": subset,
                    "expected_num_records": 1,
                }
                required.append(row)
                completed.append({**row, "cache_num_entries": 1})
                cache_dir = cache_root / model_name / dataset_name / subset
                cache_dir.mkdir(parents=True)
                shard_path = cache_dir / "shard-00000.pt"
                torch.save([_entry(model_name, dataset_name, subset)], shard_path)
                (cache_dir / "shard-00000.pt.json").write_text(
                    json.dumps(
                        {
                            "total_layers": 2,
                            "selected_layers": [0, 1],
                            "num_selected_layers": 2,
                            "hidden_dim": 3,
                            "num_entries": 1,
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "source_dataset": dataset_name,
                            "subset": subset,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                shards.append(
                    {
                        "path": str(shard_path),
                        "status": "passed",
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "source_dataset": dataset_name,
                        "subset": subset,
                        "num_entries": 1,
                        "hidden_dim": 3,
                        "total_layers": 2,
                        "selected_layers": [0, 1],
                        "errors": [],
                    }
                )
                balance_rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "subset": subset,
                        "num_entries": 1,
                    }
                )

    (manifest_dir / "stage0_summary.json").write_text(
        json.dumps(
            {
                "stage": "stage0",
                "status": "passed",
                "git_commit": "deadbeef",
                "required_cache_matrix": required,
                "completed_cache_matrix": completed,
                "missing_cache_matrix": [],
                "blocking_issues": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "cache_root": str(cache_root),
                "shards": shards,
                "total_entries": len(shards),
                "duplicate_keys": [],
                "errors": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (audit_dir / "cache_label_balance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "dataset_name", "subset", "num_entries"])
        writer.writeheader()
        writer.writerows(balance_rows)


def _append_cache_shard(
    stage0_root: Path,
    *,
    model_name: str,
    dataset_name: str,
    subset: str,
    entries: list[dict[str, object]],
    hidden_dim: int = 3,
    selected_layers: list[int] | None = None,
) -> Path:
    selected_layers = [0, 1] if selected_layers is None else selected_layers
    cache_dir = stage0_root / "cache" / model_name / dataset_name / subset
    shard_path = cache_dir / "shard-00001.pt"
    torch.save(entries, shard_path)
    (cache_dir / "shard-00001.pt.json").write_text(
        json.dumps(
            {
                "total_layers": len(selected_layers),
                "selected_layers": selected_layers,
                "num_selected_layers": len(selected_layers),
                "hidden_dim": hidden_dim,
                "num_entries": len(entries),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "source_dataset": dataset_name,
                "subset": subset,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path = stage0_root / "manifests" / "stage0_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for rows, count_field in (
        (summary["required_cache_matrix"], "expected_num_records"),
        (summary["completed_cache_matrix"], "cache_num_entries"),
    ):
        for row in rows:
            if (
                row["model_name"] == model_name
                and row["dataset_name"] == dataset_name
                and row["subset"] == subset
            ):
                row[count_field] += len(entries)
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    manifest_path = stage0_root / "manifests" / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"].append(
        {
            "path": str(shard_path),
            "status": "passed",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "source_dataset": dataset_name,
            "subset": subset,
            "num_entries": len(entries),
            "hidden_dim": hidden_dim,
            "total_layers": len(selected_layers),
            "selected_layers": selected_layers,
            "errors": [],
        }
    )
    manifest["total_entries"] += len(entries)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return shard_path


def _rewrite_cache_shard_hidden_dim(
    stage0_root: Path,
    *,
    model_name: str,
    dataset_name: str,
    subset: str,
    hidden_dim: int,
) -> Path:
    shard_path = stage0_root / "cache" / model_name / dataset_name / subset / "shard-00000.pt"
    entry = _entry(model_name, dataset_name, subset)
    entry["layer_vectors"] = torch.ones((2, hidden_dim), dtype=torch.float32)
    torch.save([entry], shard_path)

    sidecar_path = Path(str(shard_path) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["hidden_dim"] = hidden_dim
    sidecar_path.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")

    manifest_path = stage0_root / "manifests" / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        if shard["path"] == str(shard_path):
            shard["hidden_dim"] = hidden_dim
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return shard_path


def test_stage0_summary_passed_is_accepted(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    output_root = tmp_path / "outputs" / "stageA"
    _write_stage0(stage0_root)
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=output_root)

    assert result["status"] == "passed"
    assert (output_root / "audit" / "stage0_acceptance.json").exists()


def test_missing_required_dataset_matrix_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    summary_path = stage0_root / "manifests" / "stage0_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["required_cache_matrix"] = summary["required_cache_matrix"][:-1]
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any("required_cache_matrix missing" in issue for issue in result["issues"])


def test_missing_repope_or_dash_b_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    summary_path = stage0_root / "manifests" / "stage0_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["required_cache_matrix"] = [
        row for row in summary["required_cache_matrix"] if row["dataset_name"] == "pope"
    ]
    summary["completed_cache_matrix"] = [
        row for row in summary["completed_cache_matrix"] if row["dataset_name"] == "pope"
    ]
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any("repope" in issue or "dash-b" in issue for issue in result["issues"])


def test_missing_cache_label_balance_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    (stage0_root / "audit" / "cache_label_balance.csv").unlink()
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any("cache_label_balance.csv" in issue for issue in result["issues"])


def test_missing_cache_dir_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    for path in (stage0_root / "cache" / "qwen3-vl-8b" / "pope" / "popular").glob("*"):
        path.unlink()
    (stage0_root / "cache" / "qwen3-vl-8b" / "pope" / "popular").rmdir()
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any("missing cache directory" in issue for issue in result["issues"])


def test_old_path_references_are_not_required(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "passed"
    assert not any("round2" in issue for issue in result["issues"])


def test_later_required_cache_shard_with_missing_field_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    corrupt_entry = _entry("qwen3-vl-8b", "pope", "popular")
    corrupt_entry["sample_id"] = "pope-popular-2"
    del corrupt_entry["image_path"]
    corrupt_shard = _append_cache_shard(
        stage0_root,
        model_name="qwen3-vl-8b",
        dataset_name="pope",
        subset="popular",
        entries=[corrupt_entry],
    )
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any(str(corrupt_shard) in issue and "image_path" in issue for issue in result["issues"])


def test_later_required_cache_shard_hidden_dim_mismatch_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    corrupt_entry = _entry("qwen3-vl-8b", "pope", "popular")
    corrupt_entry["sample_id"] = "pope-popular-2"
    corrupt_entry["layer_vectors"] = torch.ones((2, 4), dtype=torch.float32)
    corrupt_shard = _append_cache_shard(
        stage0_root,
        model_name="qwen3-vl-8b",
        dataset_name="pope",
        subset="popular",
        entries=[corrupt_entry],
        hidden_dim=4,
    )
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any(str(corrupt_shard) in issue and "hidden_dim 4 inconsistent" in issue for issue in result["issues"])


def test_same_model_hidden_dim_mismatch_across_datasets_fails(tmp_path: Path) -> None:
    stage0_root = tmp_path / "outputs" / "stage0"
    _write_stage0(stage0_root)
    for subset in DATASETS["repope"]:
        _rewrite_cache_shard_hidden_dim(
            stage0_root,
            model_name="qwen3-vl-8b",
            dataset_name="repope",
            subset=subset,
            hidden_dim=4,
        )
    module = _load_script()

    result = module.run_preflight(stage0_root=stage0_root, output_root=tmp_path / "outputs" / "stageA")

    assert result["status"] == "failed"
    assert any(
        "hidden_dim 4 inconsistent" in issue
        and "qwen3-vl-8b/repope/popular" in issue
        and "qwen3-vl-8b/pope/popular" in issue
        for issue in result["issues"]
    )
