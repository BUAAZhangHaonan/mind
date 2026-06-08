from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import torch


def _load_module():
    path = Path("tmp/asset_repair/accept_molmo_separate_env.py")
    spec = importlib.util.spec_from_file_location("accept_molmo_separate_env", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_molmo_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    datasets = [("pope", "popular"), ("repope", "popular"), ("dash-b", "all")]
    (root / "asset_completion_summary.json").write_text(
        json.dumps(
            {
                "model_statuses": {"molmo-7b-d-0924": "verified"},
                "verified_models": ["molmo-7b-d-0924"],
                "smoke_limit": 2,
                "smoke_datasets_used": ["pope", "repope", "dash-b"],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "smoke_extraction_report.csv",
        ["model_alias", "dataset", "subset", "status", "reason", "shard_path", "sidecar_path", "num_records"],
        [
            {
                "model_alias": "molmo-7b-d-0924",
                "dataset": dataset,
                "subset": subset,
                "status": "verified",
                "reason": "smoke extraction completed",
                "shard_path": f"smoke_cache/molmo-7b-d-0924/{dataset}/{subset}/shard-00000.pt",
                "sidecar_path": f"smoke_cache/molmo-7b-d-0924/{dataset}/{subset}/shard-00000.pt.json",
                "num_records": 2,
            }
            for dataset, subset in datasets
        ],
    )
    _write_csv(
        root / "hidden_state_validation_report.csv",
        ["model_alias", "dataset", "subset", "status", "reason", "shard_path", "num_entries"],
        [
            {
                "model_alias": "molmo-7b-d-0924",
                "dataset": dataset,
                "subset": subset,
                "status": "verified",
                "reason": "validation completed",
                "shard_path": f"smoke_cache/molmo-7b-d-0924/{dataset}/{subset}/shard-00000.pt",
                "num_entries": 2,
            }
            for dataset, subset in datasets
        ],
    )
    for dataset, subset in datasets:
        cache_dir = root / "smoke_cache" / "molmo-7b-d-0924" / dataset / subset
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "sample_id": f"{dataset}-0",
                "selected_layers": [0, 1],
                "layer_vectors": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "first_token_logits": torch.tensor([0.1, 0.2]),
                "token_index": 5,
                "prompt_template_id": "molmo_single_image_raw_question_v1",
            },
            {
                "sample_id": f"{dataset}-1",
                "selected_layers": [0, 1],
                "layer_vectors": torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
                "first_token_logits": torch.tensor([0.3, 0.4]),
                "token_index": 6,
                "prompt_template_id": "molmo_single_image_raw_question_v1",
            },
        ]
        torch.save(payload, cache_dir / "shard-00000.pt")
        (cache_dir / "shard-00000.pt.json").write_text(
            json.dumps(
                {
                    "model_alias": "molmo-7b-d-0924",
                    "model_family": "molmo",
                    "total_layers": 2,
                    "hidden_dim": 2,
                    "hidden_state_index_offset": 1,
                    "selected_layers": [0, 1],
                    "token_index": 5,
                    "prompt_template_id": "molmo_single_image_raw_question_v1",
                    "validation_commit": "test-commit",
                    "num_entries": 2,
                }
            ),
            encoding="utf-8",
        )


def test_accepts_verified_separate_env_without_copying_tensors(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "source"
    output_root = tmp_path / "repair"
    _write_valid_molmo_source(source_root)

    report = module.accept_molmo_separate_env(source_root=source_root, output_root=output_root, execute=True)

    assert report["status"] == "verified_separate_env"
    assert report["copied_tensors"] is False
    assert (output_root / "molmo_separate_env_acceptance.json").is_file()
    assert (output_root / "molmo_separate_env_acceptance.md").is_file()
    assert not (output_root / "smoke_cache").exists()


def test_missing_validation_rows_keep_molmo_blocked(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "source"
    output_root = tmp_path / "repair"
    _write_valid_molmo_source(source_root)
    (source_root / "hidden_state_validation_report.csv").write_text(
        "model_alias,dataset,subset,status,reason,shard_path,num_entries\n",
        encoding="utf-8",
    )

    report = module.accept_molmo_separate_env(source_root=source_root, output_root=output_root, execute=True)

    assert report["status"] == "blocked_remove_from_panel"
    assert "missing verified validation row" in report["reason"]


def test_accepts_cwd_relative_paths_that_already_include_source_root(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    source_root = Path("source")
    output_root = Path("repair")
    _write_valid_molmo_source(source_root)
    datasets = [("pope", "popular"), ("repope", "popular"), ("dash-b", "all")]
    _write_csv(
        source_root / "smoke_extraction_report.csv",
        ["model_alias", "dataset", "subset", "status", "reason", "shard_path", "sidecar_path", "num_records"],
        [
            {
                "model_alias": "molmo-7b-d-0924",
                "dataset": dataset,
                "subset": subset,
                "status": "verified",
                "reason": "smoke extraction completed",
                "shard_path": f"source/smoke_cache/molmo-7b-d-0924/{dataset}/{subset}/shard-00000.pt",
                "sidecar_path": f"source/smoke_cache/molmo-7b-d-0924/{dataset}/{subset}/shard-00000.pt.json",
                "num_records": 2,
            }
            for dataset, subset in datasets
        ],
    )

    report = module.accept_molmo_separate_env(source_root=source_root, output_root=output_root, execute=True)

    assert report["status"] == "verified_separate_env"
