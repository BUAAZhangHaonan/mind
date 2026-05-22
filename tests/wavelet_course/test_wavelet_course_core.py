from __future__ import annotations

from collections import Counter
import csv
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import types

import numpy as np
import pytest
import torch
import yaml

from mind.wavelet_course import teacher_bagua_features as teacher_bagua


def _load_cli_script() -> ModuleType:
    script_path = Path("scripts/wavelet_course_run.py")
    spec = importlib.util.spec_from_file_location("wavelet_course_run", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(
    *,
    sample_id: str,
    image_id: str,
    subset: str = "popular",
    label: int = 0,
    parsed_answer: int = 1,
    layer_vectors: torch.Tensor | None = None,
    first_token_logits: torch.Tensor | None = None,
) -> dict[str, object]:
    return {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subset": subset,
        "sample_id": sample_id,
        "image_id": image_id,
        "question": "Is there a cat?",
        "object_name": "cat",
        "label": label,
        "parsed_answer": parsed_answer,
        "layer_vectors": layer_vectors
        if layer_vectors is not None
        else torch.arange(36 * 8, dtype=torch.float32).reshape(36, 8),
        "first_token_logits": first_token_logits
        if first_token_logits is not None
        else torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
    }


def test_teacher_bagua_extracts_small_batch_shape_and_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(teacher_bagua, "EXPECTED_LAYER_SHAPE", (8, 3))
    monkeypatch.setattr(teacher_bagua, "NUM_WINDOWS", 2)

    layer_vectors = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)

    result = teacher_bagua.extract_teacher_bagua_features(
        layer_vectors,
        teacher_bagua.TeacherBaguaConfig(wavelet="haar", level=0),
    )

    assert result.features.shape == (2, 3 * teacher_bagua.FEATURES_PER_DIM)
    assert np.isfinite(result.features).all()


def test_teacher_bagua_wavelet_denoise_uses_batched_axis_one_pywt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, ...]]] = []

    class FakeWavelet:
        dec_len = 2

        def __init__(self, _name: str) -> None:
            pass

    fake_pywt = types.SimpleNamespace()
    fake_pywt.Wavelet = FakeWavelet
    fake_pywt.dwt_max_level = lambda _length, _dec_len: 1

    def fake_wavedec(values: np.ndarray, _wavelet: str, *, level: int, axis: int) -> list[np.ndarray]:
        calls.append((axis, tuple(values.shape)))
        assert level == 1
        assert axis == 1
        assert values.shape == (3, 8)
        return [
            np.zeros((3, 4), dtype=np.float64),
            np.asarray(
                [
                    [1.0, 1.0, 1.0, 50.0],
                    [2.0, 2.0, 2.0, 60.0],
                    [3.0, 3.0, 3.0, 70.0],
                ],
                dtype=np.float64,
            ),
        ]

    def fake_threshold(coeff: np.ndarray, threshold: np.ndarray, *, mode: str) -> np.ndarray:
        assert mode == "soft"
        assert threshold.shape == (3, 1)
        return coeff

    def fake_waverec(_coeffs: list[np.ndarray], _wavelet: str, *, axis: int) -> np.ndarray:
        assert axis == 1
        return np.arange(3 * 8, dtype=np.float64).reshape(3, 8)

    fake_pywt.wavedec = fake_wavedec
    fake_pywt.threshold = fake_threshold
    fake_pywt.waverec = fake_waverec
    monkeypatch.setitem(sys.modules, "pywt", fake_pywt)

    denoised = teacher_bagua._wavelet_denoise_matrix(
        np.ones((3, 8), dtype=np.float32),
        teacher_bagua.TeacherBaguaConfig(wavelet="fake", level=1),
    )

    assert calls == [(1, (3, 8))]
    assert denoised.shape == (3, 8)
    assert denoised.dtype == np.float32


def test_teacher_bagua_memmap_writes_entries_via_sample_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(teacher_bagua, "EXPECTED_LAYER_SHAPE", (8, 3))
    monkeypatch.setattr(teacher_bagua, "NUM_WINDOWS", 2)
    calls: list[tuple[int, tuple[int, ...]]] = []

    def fake_extract_batch(
        layer_batch: np.ndarray,
        config: teacher_bagua.TeacherBaguaConfig,
    ) -> teacher_bagua.TeacherBaguaFeatureResult:
        calls.append((config.feature_batch_size, tuple(layer_batch.shape)))
        batch_size = layer_batch.shape[0]
        features = np.arange(
            batch_size * teacher_bagua.NUM_WINDOWS * 3 * teacher_bagua.FEATURES_PER_DIM,
            dtype=np.float32,
        ).reshape(batch_size, teacher_bagua.NUM_WINDOWS, 3 * teacher_bagua.FEATURES_PER_DIM)
        return teacher_bagua.TeacherBaguaFeatureResult(
            features=features,
            epsilon_usage={"time_coeff_var": batch_size},
        )

    monkeypatch.setattr(teacher_bagua, "_extract_teacher_bagua_batch", fake_extract_batch)
    entries = [
        {"layer_vectors": np.full((8, 3), fill_value=float(index), dtype=np.float32)}
        for index in range(3)
    ]
    output_path = tmp_path / "teacher.dat"
    metadata_path = tmp_path / "teacher.json"

    metadata = teacher_bagua.write_teacher_memmap(
        entries,
        [0, 1, 0],
        output_path,
        {"wavelet": "haar", "level": 0, "feature_batch_size": 2},
        metadata_path=metadata_path,
    )

    assert calls == [(2, (2, 8, 3)), (2, (1, 8, 3))]
    assert metadata == {
        "shape": [3, 2, 3 * teacher_bagua.FEATURES_PER_DIM],
        "dtype": "float32",
        "sequence_len": 2,
        "feature_dim": 3 * teacher_bagua.FEATURES_PER_DIM,
        "labels_shape": [3],
        "config": {
            "wavelet": "haar",
            "level": 0,
            "threshold": "universal_soft",
            "epsilon": teacher_bagua.DEFAULT_EPSILON,
        },
        "epsilon_usage": {"time_coeff_var": 3},
    }
    assert metadata_path.exists()
    written = np.memmap(
        output_path,
        dtype=np.float32,
        mode="r",
        shape=tuple(metadata["shape"]),
    )
    assert written.shape == (3, 2, 3 * teacher_bagua.FEATURES_PER_DIM)
    assert np.isfinite(written).all()


def test_teacher_bagua_window_features_match_per_trace_definition() -> None:
    window = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0],
            [0.5, -1.0, 3.0, 7.0],
            [10.0, 9.0, 6.0, 2.0],
        ],
        dtype=np.float32,
    )
    batch_usage: Counter[str] = Counter()
    per_trace_usage: Counter[str] = Counter()

    batch = teacher_bagua._window_features(
        window,
        epsilon=teacher_bagua.DEFAULT_EPSILON,
        usage=batch_usage,
    )
    expected = np.concatenate(
        [
            np.concatenate(
                [
                    teacher_bagua._time_features(trace, epsilon=teacher_bagua.DEFAULT_EPSILON, usage=per_trace_usage),
                    teacher_bagua._frequency_features(
                        trace,
                        epsilon=teacher_bagua.DEFAULT_EPSILON,
                        usage=per_trace_usage,
                    ),
                ]
            )
            for trace in window
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    np.testing.assert_allclose(batch, expected, rtol=1e-6, atol=1e-6)
    assert batch.shape == (3 * teacher_bagua.FEATURES_PER_DIM,)
    assert batch_usage == per_trace_usage


def test_cli_parser_contract_accepts_required_options() -> None:
    module = _load_cli_script()

    args = module.build_parser().parse_args(
        [
            "--config",
            "configs/wavelet_course/repope_qwen3_vl_8b.yaml",
            "--preflight-only",
            "--device",
            "cuda:0",
            "--quick",
            "--allow-cpu",
            "--teacher-bagua-max-train-samples",
            "7",
        ]
    )

    assert args.config == "configs/wavelet_course/repope_qwen3_vl_8b.yaml"
    assert args.preflight_only is True
    assert args.device == "cuda:0"
    assert args.quick is True
    assert args.allow_cpu is True
    assert args.teacher_bagua_max_train_samples == 7


def test_config_uses_exact_requested_names() -> None:
    module = _load_cli_script()
    config = yaml.safe_load(Path("configs/wavelet_course/repope_qwen3_vl_8b.yaml").read_text())

    module.validate_experiment_config_names(config)
    flattened = module.flatten_experiment_configs(config)

    assert config["experiment_name"] == "wavelet_course_repope_qwen3_vl_8b"
    assert config["seed"] == 20260506
    assert config["dataset_name"] == "repope"
    assert config["subsets"] == ["popular", "random", "adversarial"]
    assert config["split_ratios"] == {"train": 0.60, "validation": 0.20, "test": 0.20}
    assert config["require_positive_in_each_split"] is True
    assert config["allow_cpu"] is False
    assert config["allow_no_xgboost"] is True
    assert config["teacher_bagua"]["enabled"] is True
    assert config["teacher_bagua"]["window_size"] == 4
    assert config["teacher_bagua"]["stride"] == 4
    assert config["teacher_bagua"]["num_features_per_window"] == 28
    assert config["teacher_bagua"]["lstm_hidden_dim"] == 64
    assert config["teacher_bagua"]["max_train_samples"] is None
    assert config["teacher_bagua"]["epochs"] == 10
    assert config["teacher_bagua"]["batch_size"] == 16
    assert config["teacher_bagua"]["learning_rate"] == 0.001
    assert list(config["teacher_bagua"]["configs"][0]) == ["name", "wavelet", "level", "threshold"]
    assert config["ours_wavelet"]["enabled"] is True
    assert config["ours_wavelet"]["max_feature_dim"] == 512
    assert config["ours_wavelet"]["trace_names"] == [
        "norm_trace",
        "delta_norm_trace",
        "cos_prev_trace",
        "cos_final_trace",
        "yes_no_margin_trace",
        "yes_no_entropy_trace",
    ]
    assert list(config["ours_wavelet"]["configs"][0]) == ["name", "transform", "wavelet", "level"]
    assert config["classifiers"]["logreg"] == {"max_iter": 5000, "class_weight": "balanced"}
    assert config["classifiers"]["xgboost"] == {
        "enabled": True,
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [100, 300],
        "eval_metric": "aucpr",
    }
    assert config["baselines"] == [
        "final_hidden_logreg",
        "mean_layer_hidden_logreg",
        "norm_traj_logreg",
        "sphere_traj_meanpool_logreg",
    ]
    assert [item["config_name"] for item in flattened] == module.EXPECTED_CONFIG_NAMES
    assert len(flattened) == 13


def test_flatten_accepts_user_style_config_and_expands_final_names() -> None:
    module = _load_cli_script()

    config = {
        "teacher_bagua": {
            "configs": [
                {"name": "teacher_bagua_haar_l1_lstm", "wavelet": "haar", "level": 1, "threshold": "universal_soft"},
                {"name": "teacher_bagua_db2_l1_lstm", "wavelet": "db2", "level": 1, "threshold": "universal_soft"},
                {"name": "teacher_bagua_db4_l1_lstm", "wavelet": "db4", "level": 1, "threshold": "universal_soft"},
            ]
        },
        "ours_wavelet": {
            "configs": [
                {"name": "ours_db2_swt_l2", "transform": "swt", "wavelet": "db2", "level": 2},
                {"name": "ours_db2_swt_l3", "transform": "swt", "wavelet": "db2", "level": 3},
                {"name": "ours_sym4_swt_l2", "transform": "swt", "wavelet": "sym4", "level": 2},
            ]
        },
        "baselines": [
            "final_hidden_logreg",
            "mean_layer_hidden_logreg",
            "norm_traj_logreg",
            "sphere_traj_meanpool_logreg",
        ],
    }

    flattened = module.flatten_experiment_configs(config)

    module.validate_experiment_config_names(config)
    module.validate_experiment_config_names(flattened)
    assert [item["config_name"] for item in flattened] == module.EXPECTED_CONFIG_NAMES
    assert [(item["method_family"], item["classifier"]) for item in flattened[3:9]] == [
        ("ours_wavelet", "logreg"),
        ("ours_wavelet", "logreg"),
        ("ours_wavelet", "logreg"),
        ("ours_wavelet", "xgb"),
        ("ours_wavelet", "xgb"),
        ("ours_wavelet", "xgb"),
    ]


def test_load_local_tokenizer_uses_qwen_model_id_from_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    model_config = tmp_path / "qwen3_vl_8b.yaml"
    model_config.write_text("model_id: Qwen/Qwen3-VL-8B-Instruct\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    tokenizer = object()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls.append({"model_id": model_id, **kwargs})
            return tokenizer

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(module, "DEFAULT_MODEL_CONFIG_PATH", model_config)

    loaded = module.load_local_tokenizer({"model_name": "qwen3-vl-8b", "ours_wavelet": {}})

    assert loaded is tokenizer
    assert calls == [
        {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "local_files_only": True,
            "trust_remote_code": True,
        }
    ]


def test_load_local_tokenizer_uses_explicit_token_ids_without_hardcoded_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()

    class FailingAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("tokenizer should not load when explicit ids exist")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FailingAutoTokenizer))

    assert module.load_local_tokenizer({"ours_wavelet": {"yes_token_id": 1, "no_token_id": 2}}) is None


def test_load_local_tokenizer_falls_back_to_hf_cache_tokenizer_json_after_protobuf_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    hf_home = tmp_path / "hf"
    repo_dir = hf_home / "hub" / "models--Qwen--Qwen3-VL-8B-Instruct"
    snapshot_dir = repo_dir / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    tokenizer_json = snapshot_dir / "tokenizer.json"
    tokenizer_json.write_text('{"fake": true}', encoding="utf-8")
    calls: list[Path] = []

    class FailingAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise ImportError("No module named 'google.protobuf'")

    class FakeEncoding:
        ids = [7, 11]

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> FakeEncoding:
            assert text == "yes"
            assert add_special_tokens is False
            return FakeEncoding()

    class FakeTokenizersTokenizer:
        @staticmethod
        def from_file(path: str) -> FakeTokenizer:
            calls.append(Path(path))
            return FakeTokenizer()

    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FailingAutoTokenizer))
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        types.SimpleNamespace(Tokenizer=FakeTokenizersTokenizer),
    )

    loaded = module.load_local_tokenizer({"model_id": "Qwen/Qwen3-VL-8B-Instruct", "ours_wavelet": {}})

    assert calls == [tokenizer_json]
    assert loaded.source == "local_tokenizer_json"
    assert loaded.encode("yes", add_special_tokens=False) == [7, 11]


def test_load_local_tokenizer_reports_auto_and_tokenizer_json_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()

    class FailingAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise ImportError("No module named 'google.protobuf'")

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FailingAutoTokenizer))

    with pytest.raises(RuntimeError) as exc_info:
        module.load_local_tokenizer({"model_id": "Qwen/Qwen3-VL-8B-Instruct", "ours_wavelet": {}})

    message = str(exc_info.value)
    assert "AutoTokenizer" in message
    assert "google.protobuf" in message
    assert "tokenizer.json" in message
    assert "models--Qwen--Qwen3-VL-8B-Instruct" in message


def test_resolve_config_reads_teacher_lstm_training_params_from_config() -> None:
    module = _load_cli_script()
    args = module.build_parser().parse_args(["--config", "unused"])
    config = yaml.safe_load(Path("configs/wavelet_course/repope_qwen3_vl_8b.yaml").read_text())

    resolved = module.resolve_config(config, args)

    assert resolved["teacher_bagua"]["epochs"] == 10
    assert resolved["teacher_bagua"]["max_train_samples"] is None
    assert resolved["teacher_bagua"]["batch_size"] == 16
    assert resolved["teacher_bagua"]["learning_rate"] == 0.001
    assert resolved["classifiers"]["teacher_lstm"]["epochs"] == 10
    assert resolved["classifiers"]["teacher_lstm"]["batch_size"] == 16
    assert resolved["classifiers"]["teacher_lstm"]["learning_rate"] == 0.001


def test_resolve_config_quick_sets_teacher_lstm_epochs_to_three() -> None:
    module = _load_cli_script()
    args = module.build_parser().parse_args(["--config", "unused", "--quick"])
    config = yaml.safe_load(Path("configs/wavelet_course/repope_qwen3_vl_8b.yaml").read_text())

    resolved = module.resolve_config(config, args)

    assert resolved["teacher_bagua"]["epochs"] == 3
    assert resolved["teacher_bagua"]["max_train_samples"] == 8
    assert resolved["classifiers"]["teacher_lstm"]["epochs"] == 3
    assert resolved["classifiers"]["teacher_lstm"]["batch_size"] == 16


def test_resolve_config_cli_overrides_quick_teacher_max_train_samples() -> None:
    module = _load_cli_script()
    args = module.build_parser().parse_args(
        ["--config", "unused", "--quick", "--teacher-bagua-max-train-samples", "16"]
    )
    config = yaml.safe_load(Path("configs/wavelet_course/repope_qwen3_vl_8b.yaml").read_text())

    resolved = module.resolve_config(config, args)

    assert resolved["teacher_bagua"]["max_train_samples"] == 16


def test_stratified_limit_keeps_two_classes_when_split_prefix_is_single_class() -> None:
    module = _load_cli_script()
    labels = np.asarray([0, 0, 0, 1, 1, 0], dtype=np.int64)
    split_indices = [0, 1, 2, 3, 4, 5]

    first = module.stratified_limit_split_indices(split_indices, labels, limit=2)
    second = module.stratified_limit_split_indices(split_indices, labels, limit=2)

    assert first == [0, 3]
    assert second == first
    assert labels[first].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("quick_run", "expected_sample_ids", "expected_splits", "expected_labels", "expected_scores"),
    [
        (
            True,
            ["train-0", "train-1", "validation-0", "validation-1", "test-0", "test-1"],
            ["train", "train", "validation", "validation", "test", "test"],
            [0, 1, 1, 0, 0, 1],
            {
                "train": [0.2, 0.8],
                "validation": [0.9, 0.1],
                "test": [0.1, 0.7],
            },
        ),
        (
            False,
            [
                "train-0",
                "train-1",
                "validation-0",
                "validation-1",
                "validation-2",
                "test-0",
                "test-1",
                "test-2",
            ],
            ["train", "train", "validation", "validation", "validation", "test", "test", "test"],
            [0, 1, 1, 0, 1, 0, 1, 0],
            {
                "train": [0.2, 0.8],
                "validation": [0.9, 0.1, 0.8],
                "test": [0.1, 0.7, 0.2],
            },
        ),
    ],
)
def test_teacher_max_train_samples_limits_entries_before_memmap_by_quick_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quick_run: bool,
    expected_sample_ids: list[str],
    expected_splits: list[str],
    expected_labels: list[int],
    expected_scores: dict[str, list[float]],
) -> None:
    module = _load_cli_script()
    rows = [
        _entry(sample_id="train-0", image_id="image-a"),
        _entry(sample_id="train-1", image_id="image-b"),
        _entry(sample_id="train-2", image_id="image-c"),
        _entry(sample_id="validation-0", image_id="image-d"),
        _entry(sample_id="validation-1", image_id="image-e"),
        _entry(sample_id="validation-2", image_id="image-f"),
        _entry(sample_id="test-0", image_id="image-g"),
        _entry(sample_id="test-1", image_id="image-h"),
        _entry(sample_id="test-2", image_id="image-i"),
    ]
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    for row, split in zip(
        rows,
        ["train", "train", "train", "validation", "validation", "validation", "test", "test", "test"],
        strict=True,
    ):
        row["wavelet_split"] = split
    preflight = {
        "population": module.WaveletPopulation(
            primary_entries=rows,
            labels=labels.tolist(),
            assignments={module.population_key(row): str(row["wavelet_split"]) for row in rows},
            audit_rows=[],
            split_source="test",
        )
    }
    config = {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subsets": ["popular"],
        "seed": 7,
        "quick_run": quick_run,
        "teacher_bagua": {
            "max_train_samples": 2,
            "feature_shape_json": "features/teacher_bagua_feature_shape.json",
            "configs": [
                {"name": "teacher_bagua_haar_l1_lstm", "wavelet": "haar", "level": 1, "threshold": "universal_soft"}
            ],
        },
    }
    extraction_calls: list[dict[str, object]] = []
    train_calls: list[dict[str, object]] = []

    def fake_write_teacher_memmap(
        entries: list[dict[str, object]],
        labels_for_entries: list[int],
        path: Path,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        extraction_calls.append(
            {
                "sample_ids": [str(entry["sample_id"]) for entry in entries],
                "splits": [str(entry["wavelet_split"]) for entry in entries],
                "labels": list(labels_for_entries),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        features = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(entries), 2, 3))
        features[:] = np.arange(len(entries) * 6, dtype=np.float32).reshape(len(entries), 2, 3)
        features.flush()
        del features
        return {"shape": [len(entries), 2, 3]}

    def fake_train_classifier(
        _experiment: dict[str, object],
        _config: dict[str, object],
        *,
        features: np.ndarray,
        labels: np.ndarray,
        entries: list[dict[str, object]],
        teacher: bool,
    ) -> object:
        train_calls.append(
            {
                "teacher": teacher,
                "feature_shape": tuple(features.shape),
                "sample_ids": [str(entry["sample_id"]) for entry in entries],
                "splits": [str(entry["wavelet_split"]) for entry in entries],
                "labels": labels.tolist(),
            }
        )
        return module.SplitScores(
            train=np.asarray(expected_scores["train"], dtype=np.float32),
            validation=np.asarray(expected_scores["validation"], dtype=np.float32),
            test=np.asarray(expected_scores["test"], dtype=np.float32),
        )

    monkeypatch.setattr(module, "write_teacher_memmap", fake_write_teacher_memmap)
    monkeypatch.setattr(module, "train_classifier", fake_train_classifier)

    rows_out = module.run_configured_experiments(config, preflight=preflight, output_root=tmp_path)

    assert extraction_calls == [
        {
            "sample_ids": expected_sample_ids,
            "splits": expected_splits,
            "labels": expected_labels,
        }
    ]
    assert train_calls == [
        {
            "teacher": True,
            "feature_shape": (len(expected_sample_ids), 2, 3),
            "sample_ids": expected_sample_ids,
            "splits": expected_splits,
            "labels": expected_labels,
        }
    ]
    assert rows_out[0]["status"] == "success"
    assert rows_out[0]["train_samples"] == 2
    assert rows_out[0]["val_samples"] == (2 if quick_run else 3)
    assert rows_out[0]["test_samples"] == (2 if quick_run else 3)
    assert rows_out[0]["feature_shape"] == f"{len(expected_sample_ids)}x2x3"


def test_teacher_max_train_samples_single_class_train_returns_failure_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    rows = [
        _entry(sample_id="train-0", image_id="image-a"),
        _entry(sample_id="train-1", image_id="image-b"),
        _entry(sample_id="train-2", image_id="image-c"),
        _entry(sample_id="validation-0", image_id="image-d"),
        _entry(sample_id="test-0", image_id="image-e"),
    ]
    labels = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)
    for row, split in zip(rows, ["train", "train", "train", "validation", "test"], strict=True):
        row["wavelet_split"] = split
    preflight = {
        "population": module.WaveletPopulation(
            primary_entries=rows,
            labels=labels.tolist(),
            assignments={module.population_key(row): str(row["wavelet_split"]) for row in rows},
            audit_rows=[],
            split_source="test",
        )
    }
    config = {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subsets": ["popular"],
        "seed": 7,
        "quick_run": True,
        "teacher_bagua": {
            "max_train_samples": 2,
            "feature_shape_json": "features/teacher_bagua_feature_shape.json",
            "configs": [
                {"name": "teacher_bagua_haar_l1_lstm", "wavelet": "haar", "level": 1, "threshold": "universal_soft"}
            ],
        },
    }

    def fail_write_teacher_memmap(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("single-class limited train should not reach teacher extraction")

    def fail_train_classifier(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("single-class limited train should not reach teacher training")

    monkeypatch.setattr(module, "write_teacher_memmap", fail_write_teacher_memmap)
    monkeypatch.setattr(module, "train_classifier", fail_train_classifier)

    rows_out = module.run_configured_experiments(config, preflight=preflight, output_root=tmp_path)

    assert len(rows_out) == 1
    assert {
        "config_name": "teacher_bagua_haar_l1_lstm",
        "method_family": "teacher_bagua",
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subset_scope": "popular",
        "classifier": "lstm",
        "wavelet": "haar",
        "wavelet_level": 1,
        "transform": "",
        "status": "failure",
        "failure_reason": "teacher_bagua_train_requires_at_least_two_classes_after_sample_limit",
        "feature_seconds": "",
        "train_eval_seconds": "",
    }.items() <= rows_out[0].items()
    assert rows_out[0]["total_seconds"] != ""


def test_ours_tokenizer_failure_is_preserved_as_failure_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    row = _entry(sample_id="a", image_id="image-a")
    row["wavelet_split"] = "train"
    preflight = {
        "population": module.WaveletPopulation(
            primary_entries=[row],
            labels=[1],
            assignments={module.population_key(row): "train"},
            audit_rows=[],
            split_source="test",
        )
    }
    config = {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subsets": ["popular"],
        "ours_wavelet": {
            "feature_shape_json": "features/ours_wavelet_feature_shape.json",
            "configs": [{"name": "ours_db2_swt_l2", "transform": "swt", "wavelet": "db2", "level": 2}],
        },
    }
    monkeypatch.setattr(
        module,
        "load_local_tokenizer",
        lambda _config: (_ for _ in ()).throw(RuntimeError("failed to load tokenizer for model_id='Qwen/Qwen3-VL-8B-Instruct'")),
    )

    rows = module.run_configured_experiments(config, preflight=preflight, output_root=tmp_path)

    assert [row["config_name"] for row in rows] == ["ours_db2_swt_l2_logreg", "ours_db2_swt_l2_xgb"]
    assert {row["status"] for row in rows} == {"failure"}
    assert all("Qwen/Qwen3-VL-8B-Instruct" in row["failure_reason"] for row in rows)


def test_ours_feature_matrix_records_tokenizer_source_in_config_and_shape_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    expected_tokenizer = types.SimpleNamespace(source="local_tokenizer_json")
    config = {"ours_wavelet": {}}
    shape_metadata = {"ours_wavelet": {}}

    def fake_extract(
        _layer_vectors: object,
        _first_token_logits: object,
        _config: object,
        *,
        tokenizer: object,
    ) -> object:
        assert tokenizer is expected_tokenizer
        return types.SimpleNamespace(
            features=np.asarray([1.0, 2.0], dtype=np.float32),
            feature_names=["a", "b"],
        )

    monkeypatch.setattr(module, "load_local_tokenizer", lambda _config: expected_tokenizer)
    monkeypatch.setattr(module, "extract_ours_wavelet_features", fake_extract)

    features = module.ours_feature_matrix(
        {"config_name": "ours_db2_swt_l2_logreg", "wavelet": "db2", "level": 2},
        config,
        [_entry(sample_id="a", image_id="image-a")],
        tmp_path,
        shape_metadata,
    )

    payload = shape_metadata["ours_wavelet"]["ours_db2_swt_l2_logreg"]
    assert features.shape == (1, 2)
    assert config["ours_wavelet"]["token_id_source"] == "local_tokenizer_json"
    assert payload["config"]["token_id_source"] == "local_tokenizer_json"


def test_cache_entry_shape_validation_uses_repope_layer_vectors() -> None:
    from mind.wavelet_course.cache_loading import validate_first_token_logits, validate_layer_vectors

    validate_layer_vectors(
        torch.zeros(3, 4),
        expected_num_layers=3,
        expected_hidden_dim=4,
        context="fake",
    )
    validate_first_token_logits(torch.zeros(8), context="fake")


def test_cache_entry_shape_validation_fails_closed_on_missing_layer_vectors() -> None:
    from mind.wavelet_course.cache_loading import validate_layer_vectors

    with pytest.raises(ValueError, match="layer_vectors shape"):
        validate_layer_vectors(
            torch.zeros(3, 4),
            expected_num_layers=36,
            expected_hidden_dim=4096,
            context="fake",
        )


def test_population_assignments_use_composite_keys_for_repeated_sample_ids() -> None:
    from mind.wavelet_course.population import build_wavelet_population, population_key

    rows = [
        _entry(sample_id="repeat-1", image_id="image-a", subset="popular", label=0, parsed_answer=1),
        _entry(sample_id="repeat-1", image_id="image-b", subset="random", label=0, parsed_answer=0),
        _entry(sample_id="sample-2", image_id="image-c", subset="popular", label=0, parsed_answer=1),
        _entry(sample_id="sample-3", image_id="image-d", subset="random", label=0, parsed_answer=0),
        _entry(sample_id="sample-4", image_id="image-e", subset="adversarial", label=0, parsed_answer=1),
        _entry(sample_id="sample-5", image_id="image-f", subset="adversarial", label=0, parsed_answer=0),
    ]

    population = build_wavelet_population(
        rows,
        manifest_dir=Path("missing-manifests"),
        subsets=("popular", "random", "adversarial"),
        seed=20260506,
        ratios=(0.50, 0.33, 0.17),
    )

    keys = [population_key(row) for row in rows]
    assert len(keys) == len(set(keys))
    assert set(population.assignments) == set(keys)
    assert population.assignments[population_key(rows[0])] != ""
    assert population.assignments[population_key(rows[0])] != population.assignments[population_key(rows[1])]


def test_grouped_split_keeps_same_image_in_one_split() -> None:
    from mind.wavelet_course.population import build_wavelet_population, population_key

    rows = [
        _entry(sample_id="a-1", image_id="shared-image", subset="popular", label=0, parsed_answer=1),
        _entry(sample_id="a-2", image_id="shared-image", subset="random", label=0, parsed_answer=0),
        _entry(sample_id="b-1", image_id="image-b", subset="popular", label=0, parsed_answer=1),
        _entry(sample_id="c-1", image_id="image-c", subset="adversarial", label=0, parsed_answer=0),
    ]

    population = build_wavelet_population(
        rows,
        manifest_dir=Path("missing-manifests"),
        subsets=("popular", "random", "adversarial"),
        seed=1,
        ratios=(0.50, 0.25, 0.25),
    )

    assert population.assignments[population_key(rows[0])] == population.assignments[population_key(rows[1])]


def test_preflight_writes_audits_and_fails_closed_on_missing_split_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    rows = [
        _entry(sample_id="train-pos", image_id="image-a", label=0, parsed_answer=1),
        _entry(sample_id="train-neg", image_id="image-b", label=0, parsed_answer=0),
        _entry(sample_id="val-neg", image_id="image-c", label=0, parsed_answer=0),
        _entry(sample_id="test-pos", image_id="image-d", label=0, parsed_answer=1),
    ]
    for row, split in zip(rows, ["train", "train", "validation", "test"], strict=True):
        row["wavelet_split"] = split
    monkeypatch.setattr(module, "load_repope_qwen_cache_entries", lambda **_kwargs: rows)
    monkeypatch.setattr(
        module,
        "build_wavelet_population",
        lambda _rows, **_kwargs: module.WaveletPopulation(
            primary_entries=[dict(row) for row in rows],
            labels=[1, 0, 0, 1],
            assignments={module.population_key(row): str(row["wavelet_split"]) for row in rows},
            audit_rows=[
                {
                    "subset": "popular",
                    "total": 4,
                    "primary_pos": 2,
                    "primary_neg": 2,
                    "train_pos": 1,
                    "train_neg": 1,
                    "validation_pos": 0,
                    "validation_neg": 1,
                    "test_pos": 1,
                    "test_neg": 0,
                }
            ],
            split_source="test",
        ),
    )

    with pytest.raises(RuntimeError, match="validation split has no positives"):
        module.run_preflight(
            {
                "stage0_root": "unused",
                "model_name": "qwen3-vl-8b",
                "dataset_name": "repope",
                "subsets": ["popular", "random", "adversarial"],
                "expected_num_layers": 36,
                "expected_hidden_dim": 4096,
                "split_ratios": {"train": 0.60, "validation": 0.20, "test": 0.20},
                "seed": 20260506,
                "require_positive_in_each_split": True,
            },
            audit_dir=tmp_path,
        )

    assert (tmp_path / "cache_acceptance.json").is_file()
    assert (tmp_path / "population_audit.csv").is_file()


def test_ours_feature_output_shape_uses_actual_module_with_small_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mind.wavelet_course.ours_wavelet_features as ours

    monkeypatch.setattr(ours, "EXPECTED_LAYER_SHAPE", (36, 3))
    result = ours.extract_ours_wavelet_features(
        torch.arange(36 * 3, dtype=torch.float32).reshape(36, 3),
        torch.tensor([0.1, 0.7, -0.2], dtype=torch.float32),
        {"wavelet": "db2", "level": 2, "yes_token_id": 1, "no_token_id": 2},
    )

    assert result.features.ndim == 1
    assert len(result.feature_names) == result.features.shape[0]


def test_failure_metrics_row_is_preserved(tmp_path: Path) -> None:
    from mind.wavelet_course.reporting import write_metrics_csv

    output = tmp_path / "metrics.csv"

    write_metrics_csv(
        [
            {
                "config_name": "ours_db2_swt_l2_xgb",
                "method_family": "ours_wavelet",
                "model_name": "qwen3-vl-8b",
                "dataset_name": "repope",
                "status": "failure",
                "failure_reason": "xgboost_not_installed",
                "feature_seconds": 1.25,
                "train_eval_seconds": "",
                "total_seconds": 1.5,
            }
        ],
        output,
    )

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert {"feature_seconds", "train_eval_seconds", "total_seconds"} <= set(rows[0])
    assert rows[0]["status"] == "failure"
    assert rows[0]["failure_reason"] == "xgboost_not_installed"
    assert rows[0]["feature_seconds"] == "1.25"
    assert rows[0]["train_eval_seconds"] == ""
    assert rows[0]["total_seconds"] == "1.5"


def test_best_configs_csv_includes_timing_fields(tmp_path: Path) -> None:
    from mind.wavelet_course.reporting import write_best_configs_csv

    output = tmp_path / "best_configs.csv"

    write_best_configs_csv(
        [
            {
                "config_name": "ours_db2_swt_l2_logreg",
                "method_family": "ours_wavelet",
                "status": "success",
                "pr_auc": 0.4,
                "feature_seconds": 2.0,
                "train_eval_seconds": 3.0,
                "total_seconds": 5.0,
            }
        ],
        output,
    )

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert {"feature_seconds", "train_eval_seconds", "total_seconds"} <= set(rows[0])
    assert rows[0]["feature_seconds"] == "2.0"
    assert rows[0]["train_eval_seconds"] == "3.0"
    assert rows[0]["total_seconds"] == "5.0"


def test_summary_md_reports_completion_configuration_and_full_metrics(tmp_path: Path) -> None:
    from mind.wavelet_course.reporting import write_summary_md

    output = tmp_path / "summary.md"
    config = {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "quick_run": False,
        "expected_num_layers": 36,
        "expected_hidden_dim": 4096,
        "teacher_bagua": {
            "window_size": 4,
            "stride": 4,
            "num_features_per_window": 28,
            "lstm_hidden_dim": 64,
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.001,
            "configs": [
                {"name": "teacher_bagua_haar_l1_lstm", "wavelet": "haar", "level": 1, "threshold": "universal_soft"},
                {"name": "teacher_bagua_db2_l1_lstm", "wavelet": "db2", "level": 1, "threshold": "universal_soft"},
                {"name": "teacher_bagua_db4_l1_lstm", "wavelet": "db4", "level": 1, "threshold": "universal_soft"},
            ],
        },
        "ours_wavelet": {
            "trace_names": [
                "norm_trace",
                "delta_norm_trace",
                "cos_prev_trace",
                "cos_final_trace",
                "yes_no_margin_trace",
                "yes_no_entropy_trace",
            ],
            "token_id_source": "local_tokenizer_or_explicit_config_required",
            "configs": [
                {"name": "ours_db2_swt_l2", "transform": "swt", "wavelet": "db2", "level": 2},
                {"name": "ours_db2_swt_l3", "transform": "swt", "wavelet": "db2", "level": 3},
                {"name": "ours_sym4_swt_l2", "transform": "swt", "wavelet": "sym4", "level": 2},
            ],
        },
        "classifiers": {
            "logreg": {"max_iter": 5000, "class_weight": "balanced"},
            "xgboost": {
                "enabled": True,
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.1],
                "n_estimators": [100, 300],
                "eval_metric": "aucpr",
            },
            "teacher_lstm": {"epochs": 10, "batch_size": 16, "learning_rate": 0.001, "patience": 3},
        },
        "baselines": [
            "final_hidden_logreg",
            "mean_layer_hidden_logreg",
            "norm_traj_logreg",
            "sphere_traj_meanpool_logreg",
        ],
    }
    metrics_rows = [
        {
            "config_name": "teacher_bagua_db2_l1_lstm",
            "method_family": "teacher_bagua",
            "status": "success",
            "classifier": "lstm",
            "wavelet": "db2",
            "wavelet_level": 1,
            "pr_auc": 0.2,
            "test_f1": 0.3,
            "feature_seconds": 10.0,
            "train_eval_seconds": 20.0,
            "total_seconds": 31.0,
            "failure_reason": "",
        },
        {
            "config_name": "ours_db2_swt_l2_logreg",
            "method_family": "ours_wavelet",
            "status": "success",
            "classifier": "logreg",
            "wavelet": "db2",
            "wavelet_level": 2,
            "transform": "swt",
            "pr_auc": 0.4,
            "test_f1": 0.5,
            "feature_seconds": 1.0,
            "train_eval_seconds": 2.0,
            "total_seconds": 3.5,
            "failure_reason": "",
        },
        {
            "config_name": "ours_db2_swt_l2_xgb",
            "method_family": "ours_wavelet",
            "status": "failure",
            "classifier": "xgb",
            "wavelet": "db2",
            "wavelet_level": 2,
            "transform": "swt",
            "feature_seconds": 1.2,
            "train_eval_seconds": "",
            "total_seconds": 1.3,
            "failure_reason": "xgboost_not_installed",
        },
        {
            "config_name": "final_hidden_logreg",
            "method_family": "mind_baseline",
            "status": "success",
            "classifier": "logreg",
            "transform": "final_hidden_logreg",
            "pr_auc": 0.31,
            "test_f1": 0.41,
            "feature_seconds": 0.1,
            "train_eval_seconds": 0.2,
            "total_seconds": 0.3,
            "failure_reason": "",
        },
        {
            "config_name": "mean_layer_hidden_logreg",
            "method_family": "mind_baseline",
            "status": "success",
            "classifier": "logreg",
            "transform": "mean_layer_hidden_logreg",
            "pr_auc": 0.32,
            "test_f1": 0.42,
            "feature_seconds": 0.1,
            "train_eval_seconds": 0.2,
            "total_seconds": 0.3,
            "failure_reason": "",
        },
        {
            "config_name": "norm_traj_logreg",
            "method_family": "halp_like_baseline",
            "status": "success",
            "classifier": "logreg",
            "transform": "norm_traj_logreg",
            "pr_auc": 0.33,
            "test_f1": 0.43,
            "feature_seconds": 0.1,
            "train_eval_seconds": 0.2,
            "total_seconds": 0.3,
            "failure_reason": "",
        },
        {
            "config_name": "sphere_traj_meanpool_logreg",
            "method_family": "halp_like_baseline",
            "status": "success",
            "classifier": "logreg",
            "transform": "sphere_traj_meanpool_logreg",
            "pr_auc": 0.34,
            "test_f1": 0.44,
            "feature_seconds": 0.1,
            "train_eval_seconds": 0.2,
            "total_seconds": 0.3,
            "failure_reason": "",
        },
    ]

    write_summary_md(
        output=output,
        config=config,
        cache_audit={"accepted": True},
        population_summary={"num_primary_population": 3, "num_hard_hallucination": 1},
        metrics_rows=metrics_rows,
        best_rows=[],
        metrics_path=tmp_path / "metrics.csv",
        best_configs_path=tmp_path / "best_configs.csv",
        quick=False,
        failures=[metrics_rows[2]],
    )

    text = output.read_text(encoding="utf-8")
    assert "## Experiment Completion" in text
    assert "- full_run: true" in text
    assert "- quick_run: false" in text
    assert "- cache_accepted: true" in text
    assert "- metrics_rows: 7" in text
    assert "- success_count: 6" in text
    assert "- failure_count: 1" in text
    assert "## Configuration Details" in text
    assert "teacher_bagua_haar_l1_lstm: wavelet=haar level=1 threshold=universal_soft" in text
    assert "LSTM hidden_dim=64 epochs=10 batch_size=16 lr=0.001 patience=3 input_shape=9x114688" in text
    assert "ours_db2_swt_l2: transform=swt wavelet=db2 SWT level=2" in text
    assert "classifier_variants=logreg,xgb" in text
    assert "trace_list=norm_trace, delta_norm_trace, cos_prev_trace, cos_final_trace, yes_no_margin_trace, yes_no_entropy_trace" in text
    assert "final_broadcast=yes token_source=local_tokenizer_or_explicit_config_required" in text
    assert "final_hidden_logreg: feature=final-layer hidden vector classifier=logreg" in text
    assert "mean_layer_hidden_logreg: feature=mean-pooled hidden vector across layers classifier=logreg" in text
    assert "norm_traj_logreg: feature=36-point hidden-norm trajectory classifier=logreg" in text
    assert "sphere_traj_meanpool_logreg: feature=mean-pooled unit-sphere layer trajectory classifier=logreg" in text
    assert "logreg=max_iter=5000 class_weight=balanced" in text
    assert "## Full Metrics" in text
    assert "| teacher_bagua | teacher_bagua_db2_l1_lstm | success | 0.2 | 0.3 | 10.0 | 20.0 | 31.0 |  |" in text
    assert "| ours_wavelet | ours_db2_swt_l2_xgb | failure |  |  | 1.2 |  | 1.3 | xgboost_not_installed |" in text
    for baseline_name in [
        "final_hidden_logreg",
        "mean_layer_hidden_logreg",
        "norm_traj_logreg",
        "sphere_traj_meanpool_logreg",
    ]:
        assert baseline_name in text
    assert "## Timing" in text
    assert "teacher_bagua: configs=1 feature_seconds=10.000000 train_eval_seconds=20.000000 total_seconds=31.000000" in text
    assert "ours_wavelet: configs=2 feature_seconds=2.200000 train_eval_seconds=2.000000 total_seconds=4.800000" in text
    assert "Teacher-Bagua total_seconds=31.000000 avg_total_seconds=31.000000" in text
    assert "Ours-Wavelet total_seconds=4.800000 avg_total_seconds=2.400000" in text


def test_run_one_config_success_records_timing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    rows = [
        _entry(sample_id="train-0", image_id="image-a", label=0),
        _entry(sample_id="train-1", image_id="image-b", label=1),
        _entry(sample_id="validation-0", image_id="image-c", label=1),
        _entry(sample_id="validation-1", image_id="image-d", label=0),
        _entry(sample_id="test-0", image_id="image-e", label=0),
        _entry(sample_id="test-1", image_id="image-f", label=1),
    ]
    labels = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int64)
    for row, split in zip(rows, ["train", "train", "validation", "validation", "test", "test"], strict=True):
        row["wavelet_split"] = split
    ticks = iter([100.0, 101.0, 102.0, 105.0, 106.0])

    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(module, "baseline_feature_matrix", lambda _experiment, _entries: np.ones((6, 2), dtype=np.float32))
    monkeypatch.setattr(
        module,
        "train_classifier",
        lambda *_args, **_kwargs: module.SplitScores(
            train=np.asarray([0.1, 0.9], dtype=np.float32),
            validation=np.asarray([0.8, 0.2], dtype=np.float32),
            test=np.asarray([0.3, 0.7], dtype=np.float32),
        ),
    )

    row = module.run_one_config(
        {"config_name": "final_hidden_logreg", "method_family": "mind_baseline", "classifier": "logreg"},
        {"model_name": "qwen3-vl-8b", "dataset_name": "repope", "subsets": ["popular"]},
        entries=rows,
        labels=labels,
        features_dir=tmp_path,
        shape_metadata={"teacher_bagua": {}, "ours_wavelet": {}},
    )

    assert row["status"] == "success"
    assert row["feature_seconds"] == 1.0
    assert row["train_eval_seconds"] == 3.0
    assert row["total_seconds"] == 6.0


def test_run_one_config_failure_records_available_timing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli_script()
    row = _entry(sample_id="train-0", image_id="image-a", label=0)
    row["wavelet_split"] = "train"
    ticks = iter([200.0, 202.0, 205.0, 205.0])

    def fail_feature(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("feature exploded")

    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(module, "baseline_feature_matrix", fail_feature)

    result = module.run_one_config(
        {"config_name": "final_hidden_logreg", "method_family": "mind_baseline", "classifier": "logreg"},
        {"model_name": "qwen3-vl-8b", "dataset_name": "repope", "subsets": ["popular"]},
        entries=[row],
        labels=np.asarray([0], dtype=np.int64),
        features_dir=tmp_path,
        shape_metadata={"teacher_bagua": {}, "ours_wavelet": {}},
    )

    assert result["status"] == "failure"
    assert result["failure_reason"] == "feature exploded"
    assert result["feature_seconds"] == 3.0
    assert result["train_eval_seconds"] == ""
    assert result["total_seconds"] == 5.0


def test_best_configs_select_expected_method_families() -> None:
    from mind.wavelet_course.reporting import best_config_rows

    rows = [
        {"config_name": "teacher_bagua_db2_l1_lstm", "method_family": "teacher_bagua", "status": "success", "pr_auc": 0.2},
        {"config_name": "ours_db2_swt_l2_logreg", "method_family": "ours_wavelet", "status": "success", "pr_auc": 0.4},
        {"config_name": "final_hidden_logreg", "method_family": "mind_baseline", "status": "success", "pr_auc": 0.3},
    ]

    ranks = {row["rank_name"]: row["config_name"] for row in best_config_rows(rows)}

    assert ranks == {
        "best_teacher_bagua": "teacher_bagua_db2_l1_lstm",
        "best_ours_wavelet": "ours_db2_swt_l2_logreg",
        "best_baseline": "final_hidden_logreg",
        "overall_best": "ours_db2_swt_l2_logreg",
    }
