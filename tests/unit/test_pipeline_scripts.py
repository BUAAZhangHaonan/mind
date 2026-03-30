from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


cache_reference_states = _load_script("cache_reference_states")
build_manifolds = _load_script("build_manifolds")
compute_drift = _load_script("compute_drift")
run_experiment = _load_script("run_experiment")


def test_build_reference_cache_output_path_uses_model_dataset_split_layout(tmp_path: Path) -> None:
    output_path = cache_reference_states.build_reference_cache_output_path(
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        dataset_name="pope-reference",
        split="train",
        shard_index=2,
    )

    assert output_path == tmp_path / "qwen3-vl-8b" / "pope-reference" / "train" / "shard-00002.pt"


def test_build_feature_output_path_uses_experiment_and_split_layout(tmp_path: Path) -> None:
    output_path = compute_drift.build_feature_output_path(
        output_root=tmp_path,
        experiment_name="smoke-qwen3-vl",
        split="popular",
    )

    assert output_path == tmp_path / "smoke-qwen3-vl" / "popular.parquet"


def test_parse_stage_list_supports_csv_and_all() -> None:
    assert run_experiment.parse_stage_list("prepare,extract,train") == ["prepare", "extract", "train"]
    assert run_experiment.parse_stage_list("all") == run_experiment.DEFAULT_STAGES


def test_build_reference_records_expands_each_candidate_object(tmp_path: Path) -> None:
    records = cache_reference_states.build_reference_records(
        candidates=[
            {
                "image_id": 11,
                "file_name": "COCO_train2017_000000000011.jpg",
                "object_names": ["dog", "bus"],
            }
        ],
        image_root=tmp_path,
        split="train",
        prompt_template="Is there a {object_name} in the image? Answer yes or no.",
    )

    assert [record.sample_id for record in records] == ["ref-11-bus", "ref-11-dog"]
    assert [record.object_name for record in records] == ["bus", "dog"]
    assert records[0].image_path == str(tmp_path / "COCO_train2017_000000000011.jpg")
    assert records[0].question == "Is there a bus in the image? Answer yes or no."
    assert records[0].subset == "reference"
    assert records[0].source_dataset == "coco_reference"


def test_load_cache_entries_supports_directory_inputs(tmp_path: Path) -> None:
    shard_root = tmp_path / "cache"
    shard_root.mkdir()
    torch.save([{"sample_id": "sample-1"}], shard_root / "shard-00000.pt")
    torch.save([{"sample_id": "sample-2"}], shard_root / "shard-00001.pt")

    manifold_entries = build_manifolds.load_cache_entries(shard_root)
    drift_entries = compute_drift.load_cache_entries(shard_root)

    assert [entry["sample_id"] for entry in manifold_entries] == ["sample-1", "sample-2"]
    assert [entry["sample_id"] for entry in drift_entries] == ["sample-1", "sample-2"]


def test_save_reference_bank_writes_stats_and_counts_report(tmp_path: Path) -> None:
    written_paths = build_manifolds.save_reference_bank(
        entries=[
            {
                "sample_id": "yes-1",
                "parsed_answer": 1,
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.0, 0.0, 0.0]]),
            },
            {
                "sample_id": "yes-2",
                "parsed_answer": 1,
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[1.0, 0.0, 0.0]]),
            },
            {
                "sample_id": "no-1",
                "parsed_answer": 0,
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[9.0, 9.0, 9.0]]),
            },
        ],
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        k_neighbors=2,
    )

    stats_path = tmp_path / "qwen3-vl-8b" / "dog" / "stats.pt"
    counts_path = tmp_path / "qwen3-vl-8b" / "reference_counts.csv"

    assert tmp_path / "qwen3-vl-8b" / "dog" / "layer-08.pt" in written_paths
    assert stats_path.exists()
    assert counts_path.exists()
    stats = torch.load(stats_path, weights_only=False)
    counts = pd.read_csv(counts_path)
    assert stats[8]["count"] == 2
    assert counts.loc[0, "object_name"] == "dog"
    assert counts.loc[0, "count"] == 2


def test_build_feature_frame_skips_entries_without_reference_coverage(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference_banks" / "qwen3-vl-8b" / "dog"
    reference_root.mkdir(parents=True)
    torch.save(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        reference_root / "layer-08.pt",
    )
    torch.save({8: {"residual_mean": 0.1, "residual_std": 0.2}}, reference_root / "stats.pt")

    frame = compute_drift.build_feature_frame(
        cache_entries=[
            {
                "sample_id": "covered",
                "image_id": 101,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.3, 0.4, 0.0]]),
            },
            {
                "sample_id": "missing",
                "image_id": 102,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.3, 0.4, 0.0]]),
            },
        ],
        reference_bank=compute_drift.load_reference_bank(tmp_path / "reference_banks", "qwen3-vl-8b"),
        reference_stats=compute_drift.load_reference_stats(tmp_path / "reference_banks", "qwen3-vl-8b"),
    )

    assert list(frame["sample_id"]) == ["covered"]


def test_build_feature_frame_labels_hallucinated_positive_answers(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference_banks" / "qwen3-vl-8b" / "dog"
    reference_root.mkdir(parents=True)
    torch.save(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        reference_root / "layer-08.pt",
    )
    torch.save(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        reference_root / "layer-13.pt",
    )

    frame = compute_drift.build_feature_frame(
        cache_entries=[
            {
                "sample_id": "hallucinated",
                "image_id": 101,
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8, 13],
                "layer_vectors": torch.tensor([[0.3, 0.4, 0.3], [0.3, 0.4, 0.6]]),
            },
            {
                "sample_id": "grounded",
                "image_id": 102,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8, 13],
                "layer_vectors": torch.tensor([[0.3, 0.4, 0.0], [0.3, 0.4, 0.0]]),
            },
        ],
        reference_bank=compute_drift.load_reference_bank(tmp_path / "reference_banks", "qwen3-vl-8b"),
        reference_stats={
            "dog": {
                8: {"residual_mean": 0.1, "residual_std": 0.2},
                13: {"residual_mean": 0.1, "residual_std": 0.2},
            }
        },
    )

    assert list(frame["image_id"]) == [101, 102]
    assert list(frame["ground_truth_label"]) == [0, 1]
    assert list(frame["answer_label"]) == [1, 1]
    assert list(frame["label"]) == [1, 0]
    assert "raw_drift_0" in frame.columns
    assert "raw_mean_drift" in frame.columns
    assert "cal_drift_0" in frame.columns
    assert "cal_approx_energy" in frame.columns
    assert "approx_energy" not in frame.columns


def test_run_experiment_builds_stage_commands_from_flat_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-qwen3.5-4b-popular",
                "model_config: configs/models/qwen3_5_4b.yaml",
                "dataset_config: configs/data/pope.yaml",
                "subset: popular",
                "split: val",
                "limit: 32",
                "selected_layers: 16",
                "detector: logistic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    commands = run_experiment.build_stage_commands(
        config_path=config_path,
        stages=["prepare", "build_reference", "extract_eval"],
    )

    assert commands["prepare"][0].endswith("python")
    assert commands["prepare"][1:] == [
        "scripts/prepare_data.py",
        "normalize-pope",
        "--source",
        "data/pope/popular.jsonl",
        "--output",
        "outputs/normalized/pope/popular.jsonl",
        "--subset",
        "popular",
        "--split",
        "val",
    ]
    assert commands["build_reference"][1] == "scripts/prepare_data.py"
    assert "--allowed-objects-from" in commands["build_reference"]
    assert "outputs/normalized/pope/popular.jsonl" in commands["build_reference"]
    assert commands["extract_eval"][1] == "scripts/extract_eval_states.py"
    assert "--records" in commands["extract_eval"]
    assert "outputs/normalized/pope/popular.jsonl" in commands["extract_eval"]
    assert "--image-root" in commands["extract_eval"]
    assert "data/coco/val2014" in commands["extract_eval"]
