from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
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
compute_baselines_script = _load_script("compute_baselines")
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


def test_save_reference_bank_writes_shared_bank_artifacts(tmp_path: Path) -> None:
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
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[1.0, 0.0, 0.0]]),
            },
        ],
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        k_neighbors=2,
        bank_scope="shared",
    )

    stats_path = tmp_path / "qwen3-vl-8b" / "__shared__" / "stats.pt"
    counts_path = tmp_path / "qwen3-vl-8b" / "reference_counts.csv"

    assert tmp_path / "qwen3-vl-8b" / "__shared__" / "layer-08.pt" in written_paths
    assert stats_path.exists()
    counts = pd.read_csv(counts_path)
    assert counts.loc[0, "object_name"] == "__shared__"
    assert counts.loc[0, "count"] == 2


def test_build_feature_frame_raises_on_entries_without_reference_coverage(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="Missing reference coverage"):
        compute_drift.build_feature_frame(
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
            reference_stats=compute_drift.load_reference_stats(
                tmp_path / "reference_banks",
                "qwen3-vl-8b",
            ),
        )


def test_build_feature_frame_can_use_shared_reference_bank(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference_banks" / "qwen3-vl-8b" / "__shared__"
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
                "sample_id": "dog-sample",
                "image_id": 101,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.3, 0.4, 0.0]]),
            },
            {
                "sample_id": "cat-sample",
                "image_id": 102,
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.4, 0.3, 0.2]]),
            },
        ],
        reference_bank=compute_drift.load_reference_bank(
            tmp_path / "reference_banks",
            "qwen3-vl-8b",
            bank_scope="shared",
        ),
        reference_stats=compute_drift.load_reference_stats(
            tmp_path / "reference_banks",
            "qwen3-vl-8b",
            bank_scope="shared",
        ),
        bank_scope="shared",
    )

    assert sorted(frame["sample_id"].tolist()) == ["cat-sample", "dog-sample"]
    assert "cal_drift_0" in frame.columns


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
        "normalize-object-yes-no",
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


def test_run_experiment_prepare_stage_supports_dash_b_dataset_config(tmp_path: Path) -> None:
    dataset_config = tmp_path / "dash_b.yaml"
    dataset_config.write_text(
        "\n".join(
            [
                "name: dash-b",
                "root: data/dash_b",
                "image_root: data/coco/val2014",
                "splits:",
                "  - main",
                "prompt_template: |",
                "  Answer yes or no based only on the image.",
                "  Question: {question}",
                "question_template: Can you see a {object_name} in this image?",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-llava-onevision-dash-b",
                "model_config: configs/models/qwen3_vl_8b.yaml",
                f"dataset_config: {dataset_config}",
                "subset: main",
                "split: val",
                "selected_layers: 16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    commands = run_experiment.build_stage_commands(
        config_path=config_path,
        stages=["prepare"],
    )

    assert commands["prepare"][1:] == [
        "scripts/prepare_data.py",
        "normalize-object-yes-no",
        "--source",
        "data/dash_b/main.jsonl",
        "--output",
        "outputs/normalized/dash-b/main.jsonl",
        "--subset",
        "main",
        "--split",
        "val",
        "--source-dataset",
        "dash-b",
        "--question-template",
        "Can you see a {object_name} in this image?",
    ]


def test_run_experiment_threads_bank_scope_into_reference_and_drift_stages(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-qwen3.5-4b-popular-shared",
                "model_config: configs/models/qwen3_5_4b.yaml",
                "dataset_config: configs/data/pope.yaml",
                "subset: popular",
                "split: val",
                "selected_layers: 16",
                "bank_scope: shared",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    commands = run_experiment.build_stage_commands(
        config_path=config_path,
        stages=["build_manifolds", "compute_drift"],
    )

    assert commands["build_manifolds"][-2:] == ["--bank-scope", "shared"]
    assert commands["compute_drift"][-2:] == ["--bank-scope", "shared"]


def test_compute_baselines_writes_variant_results_and_uncertainty_artifacts(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "reference"
    reports_root = tmp_path / "reports"
    cache_root.mkdir()
    (reference_root / "qwen3-vl-8b" / "dog").mkdir(parents=True)

    feature_rows = []
    cache_entries = []
    for index in range(8):
        label = index % 2
        parsed_answer = 1 if label == 1 else 0
        feature_rows.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "ground_truth_label": 0 if label == 1 else 1,
                "answer_label": parsed_answer,
                "label": label,
                "subset": "popular",
                "object_name": "dog",
                "raw_drift_0": float(index),
                "raw_drift_1": float(index) + 0.25,
                "cal_drift_0": float(index) * 0.5,
                "cal_drift_1": float(index) * 0.5 + 0.1,
                "cal_mean_drift": float(index) * 0.5 + 0.05,
                "cal_max_drift": float(index) * 0.5 + 0.1,
                "cal_approx_energy": float(index) + 1.0,
                "cal_detail_energy_l1": float(index) + 0.5,
            }
        )
        cache_entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0 if label == 1 else 1,
                "parsed_answer": parsed_answer,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8, 13],
                "layer_vectors": torch.tensor(
                    [
                        [0.1 + 0.1 * index, 0.2, 0.3],
                        [0.2 + 0.1 * index, 0.3, 0.4],
                    ],
                    dtype=torch.float32,
                ),
                "first_token_logits": torch.tensor(
                    [0.0, 3.0 if parsed_answer == 1 else -1.0, 3.0 if parsed_answer == 0 else -1.0],
                    dtype=torch.float32,
                ),
            }
        )

    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    torch.save(cache_entries, cache_root / "shard-00000.pt")
    torch.save(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        reference_root / "qwen3-vl-8b" / "dog" / "layer-08.pt",
    )
    torch.save(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        reference_root / "qwen3-vl-8b" / "dog" / "layer-13.pt",
    )
    torch.save(
        {
            8: {
                "residual_mean": 0.1,
                "residual_std": 0.2,
                "neighbor_residual_mean": 0.1,
                "neighbor_residual_std": 0.2,
            },
            13: {
                "residual_mean": 0.1,
                "residual_std": 0.2,
                "neighbor_residual_mean": 0.1,
                "neighbor_residual_std": 0.2,
            },
        },
        reference_root / "qwen3-vl-8b" / "dog" / "stats.pt",
    )

    exit_code = compute_baselines_script.main(
        [
            "--features-path",
            str(features_path),
            "--cache-path",
            str(cache_root),
            "--reference-root",
            str(reference_root),
            "--model-name",
            "qwen3-vl-8b",
            "--output-root",
            str(reports_root),
            "--experiment-name",
            "smoke-baselines",
            "--split-strategy",
            "image_grouped",
            "--num-folds",
            "2",
            "--bootstrap-resamples",
            "16",
            "--split-seeds",
            "3,5",
            "--yes-token-id",
            "1",
            "--no-token-id",
            "2",
        ]
    )

    baselines_path = reports_root / "smoke-baselines" / "baselines.json"
    ablations_path = reports_root / "smoke-baselines" / "ablations.csv"
    split_sensitivity_path = reports_root / "smoke-baselines" / "split_sensitivity.csv"
    variant_results_root = reports_root / "smoke-baselines" / "variant_results"

    assert exit_code == 0
    assert baselines_path.exists()
    assert ablations_path.exists()
    assert split_sensitivity_path.exists()
    assert (variant_results_root / "full.csv").exists()
    assert (variant_results_root / "output_p_yes.csv").exists()

    payload = json.loads(baselines_path.read_text(encoding="utf-8"))
    assert "full" in payload
    assert "output_p_yes" in payload
    assert "confidence_intervals" in payload["full"]
