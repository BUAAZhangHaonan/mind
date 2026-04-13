from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch


if "mind.models" not in sys.modules:
    _mind_models_shim = types.ModuleType("mind.models")
    _yes_no_pattern = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

    def _parse_yes_no_answer(text: str) -> int | None:
        match = _yes_no_pattern.search(text)
        if match is None:
            return None
        return 1 if match.group(1).lower() == "yes" else 0

    def _resolve_torch_dtype(value: str) -> torch.dtype:
        normalized = value.strip().lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {value}") from exc

    _mind_models_shim.parse_yes_no_answer = _parse_yes_no_answer
    _mind_models_shim.create_model_wrapper = lambda *args, **kwargs: (_ for _ in ()).throw(
        NotImplementedError("create_model_wrapper is only available in the real package")
    )
    _mind_models_shim.resolve_torch_dtype = _resolve_torch_dtype
    sys.modules["mind.models"] = _mind_models_shim


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
run_halp_script = _load_script("run_halp")
run_glsim_script = _load_script("run_glsim_adapted")
train_detector = _load_script("train_detector")


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


def test_run_halp_writes_metrics_and_results_from_tiny_readout_cache(tmp_path: Path) -> None:
    readout_root = tmp_path / "readouts"
    readout_root.mkdir()
    entries = []
    for index in range(12):
        hallucination_label = 1 if index % 2 == 0 else 0
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0,
                "parsed_answer": 1 if hallucination_label else 0,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
                "query_token_index": 2,
                "vision_token_span": [0, 1],
                "vision_features": torch.tensor(
                    [
                        [float(hallucination_label), 0.0],
                        [float(hallucination_label), 1.0],
                    ]
                ),
                "full_hidden_states": torch.zeros((4, 3, 2), dtype=torch.float32),
            }
        )
    torch.save(entries, readout_root / "shard-00000.pt")

    output_root = tmp_path / "reports"
    exit_code = run_halp_script.main(
        [
            "--readout-path",
            str(readout_root),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "smoke-halp",
            "--device",
            "cpu",
            "--epochs",
            "8",
            "--batch-size",
            "4",
            "--hidden-dims",
            "8,4",
            "--bootstrap-resamples",
            "50",
        ]
    )

    assert exit_code == 0
    metrics_path = output_root / "smoke-halp" / "halp.json"
    results_path = output_root / "smoke-halp" / "halp_results.csv"
    selection_path = output_root / "smoke-halp" / "halp_selection.csv"
    assert metrics_path.exists()
    assert results_path.exists()
    assert selection_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_path)
    assert sum(payload["selected_probe_counts"].values()) == 1
    assert results.columns.tolist() == [
        "sample_id",
        "image_id",
        "object_name",
        "subset",
        "label",
        "prediction",
        "score",
        "fold",
        "selected_probe",
    ]


def test_run_halp_writes_metrics_from_compact_readout_cache(tmp_path: Path) -> None:
    readout_root = tmp_path / "readouts"
    readout_root.mkdir()
    entries = []
    for index in range(12):
        hallucination_label = 1 if index % 2 == 0 else 0
        signal = float(hallucination_label)
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0,
                "parsed_answer": 1 if hallucination_label else 0,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
                "vision_features": torch.tensor(
                    [
                        [signal, 0.0],
                        [signal, 1.0],
                    ]
                ),
                "query_hidden_states": torch.full((4, 2), signal, dtype=torch.float32),
                "vision_token_hidden_states": torch.full((4, 2), signal, dtype=torch.float32),
                "readout_format": "compact_comparator_cache_v1",
                "total_layers": 4,
            }
        )
    torch.save(entries, readout_root / "shard-00000.pt")

    output_root = tmp_path / "reports"
    exit_code = run_halp_script.main(
        [
            "--readout-path",
            str(readout_root),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "smoke-halp-compact",
            "--device",
            "cpu",
            "--epochs",
            "8",
            "--batch-size",
            "4",
            "--hidden-dims",
            "8,4",
            "--bootstrap-resamples",
            "50",
        ]
    )

    assert exit_code == 0
    metrics_path = output_root / "smoke-halp-compact" / "halp.json"
    results_path = output_root / "smoke-halp-compact" / "halp_results.csv"
    assert metrics_path.exists()
    assert results_path.exists()


def test_run_halp_only_requests_required_cache_fields(tmp_path: Path, monkeypatch) -> None:
    readout_root = tmp_path / "readouts"
    readout_root.mkdir()
    entries = []
    for index in range(12):
        hallucination_label = 1 if index % 2 == 0 else 0
        signal = float(hallucination_label)
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0,
                "parsed_answer": 1 if hallucination_label else 0,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
                "vision_features": torch.tensor([[signal, 0.0], [signal, 1.0]]),
                "query_hidden_states": torch.full((4, 2), signal, dtype=torch.float32),
                "vision_token_hidden_states": torch.full((4, 2), signal, dtype=torch.float32),
                "readout_format": "compact_comparator_cache_v1",
                "total_layers": 4,
                "glsim_vision_hidden_states": torch.ones((2, 2, 2), dtype=torch.float32),
            }
        )
    torch.save(entries, readout_root / "shard-00000.pt")

    seen: dict[str, object] = {}
    original_load_cache_entries = run_halp_script.load_cache_entries

    def _wrapped_load_cache_entries(path: Path, *, keep_fields=None):
        seen["keep_fields"] = keep_fields
        return original_load_cache_entries(path, keep_fields=keep_fields)

    monkeypatch.setattr(run_halp_script, "load_cache_entries", _wrapped_load_cache_entries)

    output_root = tmp_path / "reports"
    exit_code = run_halp_script.main(
        [
            "--readout-path",
            str(readout_root),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "smoke-halp-required-fields",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--hidden-dims",
            "8,4",
            "--bootstrap-resamples",
            "10",
        ]
    )

    assert exit_code == 0
    assert seen["keep_fields"] == run_halp_script.HALP_REQUIRED_CACHE_FIELDS


def test_run_glsim_writes_metrics_and_results_from_tiny_readout_cache(tmp_path: Path, monkeypatch) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            assert add_special_tokens is False
            return {"dog": [2]}.get(text, [1])

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeWrapper:
        def load_processor(self):
            return FakeProcessor()

        def load_model(self, device: str = "cpu"):
            del device
            head = torch.nn.Linear(2, 4, bias=False)
            with torch.no_grad():
                head.weight.zero_()
                head.weight[2, 0] = 1.0
            return type("FakeModel", (), {"get_output_embeddings": lambda self: head})()

        def prepare_inputs(self, processor, *, question: str, image_path: str | None, device: str):
            del processor, question, image_path, device
            return {"input_ids": torch.tensor([[9, 2, 8]], dtype=torch.long)}

    monkeypatch.setattr(
        run_glsim_script,
        "load_yaml_config",
        lambda path, cls: type("Cfg", (), {"name": "fake", "model_id": "fake", "family": "fake"})(),
    )
    monkeypatch.setattr(run_glsim_script, "create_model_wrapper", lambda config: FakeWrapper())

    readout_root = tmp_path / "readouts"
    readout_root.mkdir()
    entries = []
    for index in range(12):
        hallucination_label = 1 if index % 2 == 0 else 0
        sign = 1.0 if hallucination_label else -1.0
        full_hidden_states = torch.zeros((4, 3, 2), dtype=torch.float32)
        full_hidden_states[0, 0] = torch.tensor([2.0 * sign, 0.0])
        full_hidden_states[0, 1] = torch.tensor([1.0 * sign, 0.0])
        full_hidden_states[0, 2] = torch.tensor([1.0 * sign, 0.0])
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "image_path": "fake.jpg",
                "question": "Is there a dog in the image?",
                "label": 0,
                "parsed_answer": 1 if hallucination_label else 0,
                "subset": "popular",
                "object_name": "dog",
                "query_token_index": 2,
                "vision_token_span": [0, 1],
                "full_hidden_states": full_hidden_states,
            }
        )
    torch.save(entries, readout_root / "shard-00000.pt")

    output_root = tmp_path / "reports"
    exit_code = run_glsim_script.main(
        [
            "--readout-path",
            str(readout_root),
            "--model-config",
            "configs/models/qwen3_vl_8b.yaml",
            "--output-root",
            str(output_root),
            "--experiment-name",
            "smoke-glsim",
            "--split-strategy",
            "image_grouped",
            "--num-folds",
            "3",
            "--image-layers",
            "0,1",
            "--text-layers",
            "0,1",
            "--k-values",
            "1",
            "--w-values",
            "0.5",
            "--bootstrap-resamples",
            "50",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    metrics_path = output_root / "smoke-glsim" / "glsim_adapted.json"
    results_path = output_root / "smoke-glsim" / "glsim_adapted_results.csv"
    selection_path = output_root / "smoke-glsim" / "glsim_adapted_selection.csv"
    assert metrics_path.exists()
    assert results_path.exists()
    assert selection_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_path)
    assert payload["selected_config_counts"]["i0_t0_k1_w0.50"] == 3
    assert results.columns.tolist() == [
        "sample_id",
        "image_id",
        "object_name",
        "subset",
        "label",
        "prediction",
        "score",
        "fold",
        "selected_config",
    ]


def test_run_glsim_writes_metrics_from_compact_readout_cache(tmp_path: Path, monkeypatch) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            assert add_special_tokens is False
            return {"dog": [2]}.get(text, [1])

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeWrapper:
        def load_processor(self):
            return FakeProcessor()

        def load_model(self, device: str = "cpu"):
            del device
            head = torch.nn.Linear(2, 4, bias=False)
            with torch.no_grad():
                head.weight.zero_()
                head.weight[2, 0] = 1.0
            return type("FakeModel", (), {"get_output_embeddings": lambda self: head})()

    monkeypatch.setattr(
        run_glsim_script,
        "load_yaml_config",
        lambda path, cls: type("Cfg", (), {"name": "fake", "model_id": "fake", "family": "fake"})(),
    )
    monkeypatch.setattr(run_glsim_script, "create_model_wrapper", lambda config: FakeWrapper())

    readout_root = tmp_path / "readouts"
    readout_root.mkdir()
    entries = []
    layer_indices = [0, 1]
    for index in range(12):
        hallucination_label = 1 if index % 2 == 0 else 0
        sign = 1.0 if hallucination_label else -1.0
        vision_slice = torch.stack(
            [
                torch.tensor([[2.0 * sign, 0.0], [1.0 * sign, 0.0]], dtype=torch.float32),
                torch.tensor([[2.0 * sign, 0.0], [1.0 * sign, 0.0]], dtype=torch.float32),
            ]
        )
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0,
                "parsed_answer": 1 if hallucination_label else 0,
                "subset": "popular",
                "object_name": "dog",
                "readout_format": "compact_comparator_cache_v1",
                "total_layers": 4,
                "query_hidden_states": torch.tensor(
                    [
                        [1.0 * sign, 0.0],
                        [1.0 * sign, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "object_hidden_states": torch.tensor(
                    [
                        [1.0 * sign, 0.0],
                        [1.0 * sign, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "glsim_layer_indices": layer_indices,
                "glsim_vision_hidden_states": vision_slice,
                "object_token_index": 1,
                "object_token_id": 2,
            }
        )
    torch.save(entries, readout_root / "shard-00000.pt")

    output_root = tmp_path / "reports"
    exit_code = run_glsim_script.main(
        [
            "--readout-path",
            str(readout_root),
            "--model-config",
            "configs/models/qwen3_vl_8b.yaml",
            "--output-root",
            str(output_root),
            "--experiment-name",
            "smoke-glsim-compact",
            "--split-strategy",
            "image_grouped",
            "--num-folds",
            "3",
            "--image-layers",
            "0,1",
            "--text-layers",
            "0,1",
            "--k-values",
            "1",
            "--w-values",
            "0.5",
            "--bootstrap-resamples",
            "50",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    metrics_path = output_root / "smoke-glsim-compact" / "glsim_adapted.json"
    results_path = output_root / "smoke-glsim-compact" / "glsim_adapted_results.csv"
    assert metrics_path.exists()
    assert results_path.exists()


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


def test_save_reference_bank_persists_low_support_layers_for_drift(tmp_path: Path) -> None:
    written_paths = build_manifolds.save_reference_bank(
        entries=[
            {
                "sample_id": "yes-1",
                "parsed_answer": 1,
                "object_name": "backpack",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.0, 0.0, 0.0]]),
            },
            {
                "sample_id": "yes-2",
                "parsed_answer": 1,
                "object_name": "backpack",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[1.0, 0.0, 0.0]]),
            },
        ],
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        k_neighbors=4,
    )

    layer_path = tmp_path / "qwen3-vl-8b" / "backpack" / "layer-08.pt"
    stats_path = tmp_path / "qwen3-vl-8b" / "backpack" / "stats.pt"
    counts_path = tmp_path / "qwen3-vl-8b" / "reference_counts.csv"

    assert layer_path in written_paths
    assert layer_path.exists()
    assert stats_path.exists()
    counts = pd.read_csv(counts_path)
    assert counts.loc[0, "object_name"] == "backpack"
    assert counts.loc[0, "count"] == 2
    assert not bool(counts.loc[0, "supports_manifold"])


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


def test_save_reference_bank_writes_shuffled_object_mapping(tmp_path: Path) -> None:
    written_paths = build_manifolds.save_reference_bank(
        entries=[
            {
                "sample_id": "yes-1",
                "parsed_answer": 1,
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[1.0, 0.0]]),
            },
            {
                "sample_id": "yes-2",
                "parsed_answer": 1,
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[2.0, 0.0]]),
            },
            {
                "sample_id": "yes-3",
                "parsed_answer": 1,
                "object_name": "bus",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[3.0, 0.0]]),
            },
        ],
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        k_neighbors=1,
        bank_scope="shuffled_object",
        shuffle_seed=7,
    )

    mapping_path = tmp_path / "qwen3-vl-8b" / "shuffled_object_map.json"
    assert mapping_path in written_paths
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert sorted(mapping) == ["bus", "cat", "dog"]
    assert all(target != source for target, source in mapping.items())
    for target, source in mapping.items():
        target_tensor = torch.load(
            tmp_path / "qwen3-vl-8b" / target / "layer-08.pt",
            weights_only=False,
        )
        expected_value = {"dog": 1.0, "cat": 2.0, "bus": 3.0}[source]
        assert target_tensor.shape == (1, 2)
        assert float(target_tensor[0, 0]) == expected_value


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


def test_compute_drift_parser_accepts_shuffled_object_scope() -> None:
    args = compute_drift.build_parser().parse_args(
        [
            "--output-root",
            "outputs/features",
            "--experiment-name",
            "smoke-qwen",
            "--split",
            "popular",
            "--bank-scope",
            "shuffled_object",
        ]
    )

    assert args.bank_scope == "shuffled_object"


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

    assert commands["prepare"][0][0].endswith("python")
    assert commands["prepare"][0][1:] == [
        "scripts/prepare_data.py",
        "normalize-object-yes-no",
        "--source",
        "data/pope/popular.jsonl",
        "--output",
        "outputs/round2_2026_04/normalized/pope/popular.jsonl",
        "--subset",
        "popular",
        "--split",
        "val",
    ]
    assert commands["build_reference"][0][1] == "scripts/prepare_data.py"
    assert "--allowed-objects-from" in commands["build_reference"][0]
    assert "outputs/round2_2026_04/normalized/pope/popular.jsonl" in commands["build_reference"][0]
    assert commands["extract_eval"][0][1] == "scripts/extract_eval_states.py"
    assert "--records" in commands["extract_eval"][0]
    assert "outputs/round2_2026_04/normalized/pope/popular.jsonl" in commands["extract_eval"][0]
    assert "--image-root" in commands["extract_eval"][0]
    assert "data/coco/val2014" in commands["extract_eval"][0]


def test_run_experiment_prepare_stage_supports_dash_b_dataset_config(tmp_path: Path) -> None:
    dataset_config = tmp_path / "dash_b.yaml"
    dataset_config.write_text(
        "\n".join(
            [
                "name: dash-b",
                "root: data/dash_b",
                "image_root: data/dash_b",
                "splits:",
                "  - main",
                "prompt_template: '{question}'",
                "question_template: Can you see a {object_name} in this image? Please answer only with yes or no.",
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

    assert commands["prepare"][0][1:] == [
        "scripts/prepare_data.py",
        "normalize-object-yes-no",
        "--source",
        "data/dash_b",
        "--output",
        "outputs/round2_2026_04/normalized/dash-b/main.jsonl",
        "--subset",
        "main",
        "--split",
        "val",
        "--source-dataset",
        "dash-b",
        "--question-template",
        "Can you see a {object_name} in this image? Please answer only with yes or no.",
    ]


def test_run_experiment_prepare_stage_prefers_dash_b_subset_file_over_directory_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]
    dataset_config = tmp_path / "dash_b.yaml"
    dataset_config.write_text(
        "\n".join(
                [
                    "name: dash-b",
                    "root: data/dash_b",
                    "image_root: data/dash_b",
                    "splits:",
                    "  - main",
                    "prompt_template: '{question}'",
                ]
            )
            + "\n",
            encoding="utf-8",
    )
    dash_b_root = tmp_path / "data" / "dash_b"
    dash_b_root.mkdir(parents=True)
    (dash_b_root / "main.jsonl").write_text("", encoding="utf-8")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-llava-onevision-dash-b",
                f"model_config: {repo_root / 'configs/models/qwen3_vl_8b.yaml'}",
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

    assert commands["prepare"][0][4] == "data/dash_b/main.jsonl"


def test_run_experiment_builds_multiple_subset_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-qwen3.5-4b-popular",
                "model_config: configs/models/qwen3_5_4b.yaml",
                "dataset_config: configs/data/pope.yaml",
                "subset: [popular, adversarial]",
                "split: val",
                "selected_layers: 16",
                "detector: logistic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    commands = run_experiment.build_stage_commands(
        config_path=config_path,
        stages=["prepare", "extract_eval", "compute_drift", "baselines", "evaluate", "plot"],
    )

    assert len(commands["prepare"]) == 2
    assert len(commands["extract_eval"]) == 2
    assert len(commands["compute_drift"]) == 2
    assert len(commands["baselines"]) == 2
    assert len(commands["evaluate"]) == 2
    assert len(commands["plot"]) == 2

    assert "outputs/round2_2026_04/normalized/pope/popular.jsonl" in commands["prepare"][0]
    assert "outputs/round2_2026_04/normalized/pope/adversarial.jsonl" in commands["prepare"][1]
    assert "outputs/round2_2026_04/normalized/pope/popular.jsonl" in commands["extract_eval"][0]
    assert "outputs/round2_2026_04/normalized/pope/adversarial.jsonl" in commands["extract_eval"][1]
    assert "--experiment-name" in commands["compute_drift"][0]
    assert "smoke-qwen3.5-4b-popular" in commands["compute_drift"][0]
    assert "--experiment-name" in commands["compute_drift"][1]
    assert "smoke-qwen3.5-4b-popular-adversarial" in commands["compute_drift"][1]
    assert "--experiment-name" in commands["baselines"][0]
    assert "smoke-qwen3.5-4b-popular" in commands["baselines"][0]
    assert "--experiment-name" in commands["baselines"][1]
    assert "smoke-qwen3.5-4b-popular-adversarial" in commands["baselines"][1]
    assert "--experiment-name" in commands["evaluate"][0]
    assert "smoke-qwen3.5-4b-popular" in commands["evaluate"][0]
    assert "--experiment-name" in commands["evaluate"][1]
    assert "smoke-qwen3.5-4b-popular-adversarial" in commands["evaluate"][1]
    assert "--features-path" in commands["plot"][1]
    assert "outputs/round2_2026_04/features/smoke-qwen3.5-4b-popular-adversarial/adversarial.parquet" in commands["plot"][1]


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

    assert commands["build_manifolds"][0][-2:] == ["--bank-scope", "shared"]
    assert commands["compute_drift"][0][-2:] == ["--bank-scope", "shared"]


def test_run_experiment_supports_output_root_and_baselines_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke-qwen3.5-4b-popular",
                "model_config: configs/models/qwen3_5_4b.yaml",
                "dataset_config: configs/data/pope.yaml",
                "subset: popular",
                "split: val",
                "selected_layers: 16",
                "full_variant: raw_plus_calibrated_simple",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    commands = run_experiment.build_stage_commands(
        config_path=config_path,
        stages=["compute_drift", "baselines"],
        output_root=tmp_path / "round2",
    )

    assert str(tmp_path / "round2" / "reference_banks") in commands["compute_drift"][0]
    assert commands["baselines"][0][1] == "scripts/compute_baselines.py"
    assert str(
        tmp_path / "round2" / "features" / "smoke-qwen3.5-4b-popular" / "popular.parquet"
    ) in commands["baselines"][0]
    assert commands["baselines"][0][-2:] == ["--full-variant", "raw_plus_calibrated_simple"]


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
    full_results = pd.read_csv(variant_results_root / "full.csv")
    assert "full" in payload
    assert "output_p_yes" in payload
    assert "confidence_intervals" in payload["full"]
    assert full_results.columns.tolist() == [
        "sample_id",
        "image_id",
        "object_name",
        "subset",
        "label",
        "prediction",
        "score",
        "fold",
    ]


def test_compute_baselines_can_apply_label_overrides_and_full_variant(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "reference"
    reports_root = tmp_path / "reports"
    overrides_path = tmp_path / "repope.jsonl"
    cache_root.mkdir()
    (reference_root / "qwen3-vl-8b" / "dog").mkdir(parents=True)

    feature_rows = []
    cache_entries = []
    for index in range(8):
        label = index % 2
        parsed_answer = 1
        feature_rows.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "ground_truth_label": 1,
                "answer_label": parsed_answer,
                "label": 0,
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
                "label": 1,
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
                "first_token_logits": torch.tensor([0.0, 3.0, -1.0], dtype=torch.float32),
            }
        )

    overrides_path.write_text(
        "\n".join(
            json.dumps({"sample_id": f"sample-{index}", "label": 0 if index % 2 == 0 else 1})
            for index in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    torch.save(cache_entries, cache_root / "shard-00000.pt")
    for layer_index in [8, 13]:
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
            reference_root / "qwen3-vl-8b" / "dog" / f"layer-{layer_index:02d}.pt",
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
            "smoke-baselines-repope",
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
            "--label-overrides",
            str(overrides_path),
            "--full-variant",
            "raw_plus_calibrated_simple",
        ]
    )

    payload = json.loads(
        (reports_root / "smoke-baselines-repope" / "baselines.json").read_text(encoding="utf-8")
    )
    full_results = pd.read_csv(reports_root / "smoke-baselines-repope" / "variant_results" / "full.csv")

    assert exit_code == 0
    assert payload["full"]["roc_auc"] == payload["raw_plus_calibrated_simple"]["roc_auc"]
    assert payload["full"]["pr_auc"] == payload["raw_plus_calibrated_simple"]["pr_auc"]
    assert sorted(full_results["label"].unique().tolist()) == [0, 1]
    assert full_results.loc[full_results["sample_id"] == "sample-0", "label"].item() == 1


def test_compute_baselines_can_merge_selected_variant_runs(tmp_path: Path) -> None:
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
    for layer_index in [8, 13]:
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
            reference_root / "qwen3-vl-8b" / "dog" / f"layer-{layer_index:02d}.pt",
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

    common_args = [
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
        "smoke-baselines-merge",
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

    exit_code_first = compute_baselines_script.main(common_args + ["--variants", "full,output_p_yes"])
    exit_code_second = compute_baselines_script.main(
        common_args + ["--variants", "output_logit_margin,linear_probe"]
    )

    baselines_path = reports_root / "smoke-baselines-merge" / "baselines.json"
    variant_results_root = reports_root / "smoke-baselines-merge" / "variant_results"
    payload = json.loads(baselines_path.read_text(encoding="utf-8"))
    ablations = pd.read_csv(reports_root / "smoke-baselines-merge" / "ablations.csv")
    split_sensitivity = pd.read_csv(
        reports_root / "smoke-baselines-merge" / "split_sensitivity.csv"
    )

    assert exit_code_first == 0
    assert exit_code_second == 0
    assert (variant_results_root / "full.csv").exists()
    assert (variant_results_root / "output_p_yes.csv").exists()
    assert (variant_results_root / "output_logit_margin.csv").exists()
    assert (variant_results_root / "linear_probe.csv").exists()
    assert set(["full", "output_p_yes", "output_logit_margin", "linear_probe"]).issubset(payload)
    assert ablations["variant"].tolist() == [
        "full",
        "linear_probe",
        "output_p_yes",
        "output_logit_margin",
    ]
    assert split_sensitivity.groupby("variant")["random_state"].nunique().to_dict() == {
        "full": 2,
        "linear_probe": 2,
        "output_logit_margin": 2,
        "output_p_yes": 2,
    }


def test_compute_baselines_persists_completed_variants_before_later_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    for layer_index in [8, 13]:
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
            reference_root / "qwen3-vl-8b" / "dog" / f"layer-{layer_index:02d}.pt",
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

    original_evaluate = compute_baselines_script.evaluate_feature_frame
    call_count = {"count": 0}

    def _crash_before_second_variant_metrics(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("boom")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        compute_baselines_script,
        "evaluate_feature_frame",
        _crash_before_second_variant_metrics,
    )

    args = [
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
        "smoke-baselines-partial-persist",
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
        "--variants",
        "full,output_p_yes",
    ]

    with pytest.raises(RuntimeError, match="boom"):
        compute_baselines_script.main(args)

    report_root = reports_root / "smoke-baselines-partial-persist"
    payload = json.loads((report_root / "baselines.json").read_text(encoding="utf-8"))
    ablations = pd.read_csv(report_root / "ablations.csv")
    split_sensitivity = pd.read_csv(report_root / "split_sensitivity.csv")

    assert (report_root / "variant_results" / "full.csv").exists()
    assert "full" in payload
    assert "output_p_yes" not in payload
    assert ablations["variant"].tolist() == ["full"]
    assert split_sensitivity["variant"].tolist() == ["full", "full"]


def test_compute_baselines_skips_unselected_heavy_variant_builders(tmp_path: Path, monkeypatch) -> None:
    features_path = tmp_path / "features.parquet"
    cache_root = tmp_path / "cache"
    reference_root = tmp_path / "reference"
    reports_root = tmp_path / "reports"
    cache_root.mkdir()
    (reference_root / "qwen3-vl-8b" / "dog").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "ground_truth_label": 0 if index % 2 else 1,
                "answer_label": index % 2,
                "label": index % 2,
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
            for index in range(8)
        ]
    ).to_parquet(features_path, index=False)
    torch.save(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0 if index % 2 else 1,
                "parsed_answer": index % 2,
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
                    [0.0, 3.0 if index % 2 else -1.0, 3.0 if index % 2 == 0 else -1.0],
                    dtype=torch.float32,
                ),
            }
            for index in range(8)
        ],
        cache_root / "shard-00000.pt",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("unexpected heavy builder call")

    monkeypatch.setattr(compute_baselines_script, "build_linear_probe_frame", _fail)
    monkeypatch.setattr(compute_baselines_script, "build_no_manifold_feature_frame", _fail)

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
            "smoke-baselines-fast-path",
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
            "--variants",
            "full,output_p_yes",
        ]
    )

    assert exit_code == 0
    variant_results_root = reports_root / "smoke-baselines-fast-path" / "variant_results"
    assert (variant_results_root / "full.csv").exists()
    assert (variant_results_root / "output_p_yes.csv").exists()


def test_compute_baselines_does_not_load_cache_for_feature_only_variants(tmp_path: Path, monkeypatch) -> None:
    features_path = tmp_path / "features.parquet"
    reference_root = tmp_path / "reference"
    reports_root = tmp_path / "reports"
    (reference_root / "qwen3-vl-8b").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "ground_truth_label": 0 if index % 2 else 1,
                "answer_label": index % 2,
                "label": index % 2,
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
            for index in range(8)
        ]
    ).to_parquet(features_path, index=False)

    def _fail(*args, **kwargs):
        raise AssertionError("cache should not be loaded for feature-only variants")

    monkeypatch.setattr(compute_baselines_script, "load_cache_entries", _fail)

    exit_code = compute_baselines_script.main(
        [
            "--features-path",
            str(features_path),
            "--cache-path",
            str(tmp_path / "missing-cache"),
            "--reference-root",
            str(reference_root),
            "--model-name",
            "qwen3-vl-8b",
            "--output-root",
            str(reports_root),
            "--experiment-name",
            "smoke-baselines-feature-only",
            "--split-strategy",
            "image_grouped",
            "--num-folds",
            "2",
            "--bootstrap-resamples",
            "16",
            "--split-seeds",
            "3,5",
            "--variants",
            "full",
        ]
    )

    assert exit_code == 0
    assert (reports_root / "smoke-baselines-feature-only" / "variant_results" / "full.csv").exists()


def test_compute_baselines_prefers_known_token_ids_before_processor_lookup(monkeypatch) -> None:
    yes_token_ids, no_token_ids = compute_baselines_script.resolve_token_ids(
        model_name="qwen3-vl-8b",
        model_id="",
        yes_token_ids=[],
        no_token_ids=[],
    )

    assert yes_token_ids == [9693, 9834, 9454]
    assert no_token_ids == [2152, 902, 2753]
