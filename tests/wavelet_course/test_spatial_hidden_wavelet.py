from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from mind.wavelet_course.common_classifiers import SplitScores
from mind.wavelet_course.common_feature_protocols import stat28
from mind.wavelet_course.population import WaveletPopulation


def _fake_pywt_for_reconstruction(reconstructed: np.ndarray) -> types.SimpleNamespace:
    calls: list[tuple[str, int, tuple[int, ...]]] = []

    class FakeWavelet:
        dec_len = 2

        def __init__(self, _name: str) -> None:
            pass

    def wavedec(values: np.ndarray, _wavelet: str, *, mode: str, level: int, axis: int) -> list[np.ndarray]:
        calls.append(("wavedec", axis, tuple(values.shape)))
        assert axis == 1
        assert mode == "symmetric"
        assert level == 1
        return [
            np.zeros((values.shape[0], values.shape[1] // 2), dtype=np.float64),
            np.ones((values.shape[0], values.shape[1] // 2), dtype=np.float64),
        ]

    def threshold(coeff: np.ndarray, value: np.ndarray, *, mode: str) -> np.ndarray:
        calls.append(("threshold", -1, tuple(coeff.shape)))
        assert mode == "soft"
        assert value.shape == (coeff.shape[0], 1)
        return coeff

    def waverec(_coeffs: list[np.ndarray], _wavelet: str, *, mode: str, axis: int) -> np.ndarray:
        calls.append(("waverec", axis, tuple(reconstructed.shape)))
        assert axis == 1
        assert mode == "symmetric"
        return reconstructed.astype(np.float64, copy=True)

    return types.SimpleNamespace(
        Wavelet=FakeWavelet,
        dwt_max_level=lambda _length, _dec_len: 2,
        wavedec=wavedec,
        threshold=threshold,
        waverec=waverec,
        calls=calls,
    )


def test_spatial_dwt_stat28_sequence_reconstructs_hidden_axis_and_returns_layer_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.spatial_wavelet_features import SpatialWaveletConfig, spatial_dwt_stat28_sequence

    reconstructed = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 10.0
    fake_pywt = _fake_pywt_for_reconstruction(reconstructed)
    monkeypatch.setitem(sys.modules, "pywt", fake_pywt)

    features = spatial_dwt_stat28_sequence(
        np.ones((4, 8), dtype=np.float32),
        SpatialWaveletConfig(wavelet="fake", level=1, expected_num_layers=4, expected_hidden_dim=8),
    )

    assert features.shape == (4, 28)
    np.testing.assert_allclose(features, stat28(reconstructed), rtol=1e-6, atol=1e-6)
    assert fake_pywt.calls[0] == ("wavedec", 1, (4, 8))
    assert fake_pywt.calls[-1] == ("waverec", 1, (4, 8))


def test_spatial_dwt_stat28_sequence_batch_uses_one_hidden_axis_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.spatial_wavelet_features import (
        SpatialWaveletConfig,
        spatial_dwt_stat28_sequence_batch,
    )

    reconstructed = np.arange(2 * 4 * 8, dtype=np.float32).reshape(8, 8) / 10.0
    fake_pywt = _fake_pywt_for_reconstruction(reconstructed)
    monkeypatch.setitem(sys.modules, "pywt", fake_pywt)

    features = spatial_dwt_stat28_sequence_batch(
        np.ones((2, 4, 8), dtype=np.float32),
        SpatialWaveletConfig(wavelet="fake", level=1, expected_num_layers=4, expected_hidden_dim=8),
    )

    assert features.shape == (2, 4, 28)
    expected = stat28(reconstructed.reshape(2, 4, 8))
    np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)
    assert fake_pywt.calls[0] == ("wavedec", 1, (8, 8))


def test_spatial_dwt_stat28_sequence_fails_closed_for_bad_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.spatial_wavelet_features import SpatialWaveletConfig, spatial_dwt_stat28_sequence

    fake_pywt = _fake_pywt_for_reconstruction(np.ones((4, 8), dtype=np.float32))
    monkeypatch.setitem(sys.modules, "pywt", fake_pywt)
    config = SpatialWaveletConfig(wavelet="fake", level=1, expected_num_layers=4, expected_hidden_dim=8)

    with pytest.raises(ValueError, match="layer_vectors shape"):
        spatial_dwt_stat28_sequence(np.ones((4, 7), dtype=np.float32), config)

    values = np.ones((4, 8), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        spatial_dwt_stat28_sequence(values, config)


def test_spatial_dwt_stat28_sequence_requires_pywt(monkeypatch: pytest.MonkeyPatch) -> None:
    from mind.wavelet_course.spatial_wavelet_features import SpatialWaveletConfig, spatial_dwt_stat28_sequence

    monkeypatch.setitem(sys.modules, "pywt", None)

    with pytest.raises(ImportError, match="requires pywt"):
        spatial_dwt_stat28_sequence(
            np.ones((4, 8), dtype=np.float32),
            SpatialWaveletConfig(expected_num_layers=4, expected_hidden_dim=8),
        )


def _spatial_population() -> WaveletPopulation:
    splits = ("train", "train", "validation", "validation", "test", "test")
    labels = [0, 1, 0, 1, 0, 1]
    entries = []
    for index, (split, label) in enumerate(zip(splits, labels, strict=True)):
        entries.append(
            {
                "wavelet_population_key": f"row-{index}",
                "wavelet_split": split,
                "wavelet_label": label,
                "image_id": f"image-{index}",
                "subset": "popular",
                "layer_vectors": np.full((4, 8), fill_value=float(index + 1), dtype=np.float32),
            }
        )
    return WaveletPopulation(
        primary_entries=entries,
        labels=labels,
        assignments={f"row-{index}": split for index, split in enumerate(splits)},
        audit_rows=[],
        split_source="unit-test",
    )


def test_spatial_runner_writes_metrics_and_uses_quick_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course import spatial_wavelet_runner as runner

    captured: dict[str, object] = {}

    def fake_features(layer_vectors: object, config: object) -> np.ndarray:
        del config
        values = np.asarray(layer_vectors, dtype=np.float32)
        assert values.shape[1:] == (4, 8)
        return np.repeat(values.mean(axis=2, keepdims=True), 28, axis=2).astype(np.float32)

    def fake_train_sequence_model(
        model_name: str,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        *,
        test_x: np.ndarray,
        device: str,
        batch_size: int,
        max_epochs: int,
        patience: int,
        learning_rate: float,
        seed: int,
        **_kwargs: object,
    ) -> types.SimpleNamespace:
        captured.update(
            {
                "model_name": model_name,
                "train_shape": train_x.shape,
                "validation_shape": validation_x.shape,
                "test_shape": test_x.shape,
                "device": device,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "learning_rate": learning_rate,
                "seed": seed,
                "train_y": train_y.tolist(),
                "validation_y": validation_y.tolist(),
            }
        )
        return types.SimpleNamespace(
            scores=SplitScores(
                train=np.array([0.1, 0.9], dtype=np.float32),
                validation=np.array([0.2, 0.8], dtype=np.float32),
                test=np.array([0.3, 0.7], dtype=np.float32),
            ),
            training_curve=[{"epoch": 1.0, "val_pr_auc": 1.0}],
            best_epoch=1,
            best_validation_pr_auc=1.0,
            epochs_ran=1,
            early_stopped=False,
            converged=False,
            max_epoch_reached=False,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
        )

    monkeypatch.setattr(runner, "spatial_dwt_stat28_sequence_batch", fake_features)
    monkeypatch.setattr(runner, "train_sequence_model", fake_train_sequence_model)

    status = runner.run_spatial_hidden_wavelet_experiment(
        config={
            "seed": 123,
            "device": "cuda:0",
            "quick_run": True,
            "expected_num_layers": 4,
            "expected_hidden_dim": 8,
        },
        preflight={"population": _spatial_population()},
        output_root=tmp_path,
    )

    metrics_path = tmp_path / "reports" / "spatial_hidden_wavelet_metrics.csv"
    summary_path = tmp_path / "reports" / "spatial_hidden_wavelet_summary.md"
    assert status["status"] == "success"
    assert status["metrics_path"] == str(metrics_path)
    assert status["summary_path"] == str(summary_path)
    assert captured["model_name"] == "lstm_projected"
    assert captured["train_shape"] == (2, 4, 28)
    assert captured["max_epochs"] == 5
    assert captured["batch_size"] == 32
    assert captured["device"] == "cuda:0"

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "F_spatial_dwt_db2_l2_universal_soft_stat28_sequence_lstm_projected"
    assert row["status"] == "success"
    assert row["feature_shape"] == "6x4x28"
    assert row["max_epochs"] == "5"
    assert row["pr_auc"] != ""
    assert "F_spatial_dwt_db2_l2_universal_soft_stat28_sequence_lstm_projected" in summary_path.read_text(
        encoding="utf-8"
    )


def test_spatial_runner_fails_closed_when_train_split_has_one_class(tmp_path: Path) -> None:
    from mind.wavelet_course.spatial_wavelet_runner import run_spatial_hidden_wavelet_experiment

    population = _spatial_population()
    labels = [0, 0, 0, 1, 0, 1]
    for entry, label in zip(population.primary_entries, labels, strict=True):
        entry["wavelet_label"] = label
    population = WaveletPopulation(
        primary_entries=population.primary_entries,
        labels=labels,
        assignments=population.assignments,
        audit_rows=[],
        split_source="unit-test",
    )

    with pytest.raises(RuntimeError, match="train split.*two classes"):
        run_spatial_hidden_wavelet_experiment(
            config={"expected_num_layers": 4, "expected_hidden_dim": 8, "allow_cpu": True, "device": "cpu"},
            preflight={"population": population},
            output_root=tmp_path,
        )


def test_spatial_quick_limits_samples_stratified_by_split() -> None:
    from mind.wavelet_course.spatial_wavelet_runner import _limit_entries_for_quick

    entries = []
    labels = []
    for split in ("train", "validation", "test"):
        for index, label in enumerate((0, 1, 0, 1)):
            entries.append({"wavelet_split": split, "sample_id": f"{split}-{index}"})
            labels.append(label)

    limited_entries, limited_labels = _limit_entries_for_quick(
        entries,
        np.asarray(labels, dtype=np.int64),
        {
            "quick": {
                "max_train_samples": 2,
                "max_validation_samples": 2,
                "max_test_samples": 2,
            }
        },
    )

    assert [entry["sample_id"] for entry in limited_entries] == [
        "train-0",
        "train-1",
        "validation-0",
        "validation-1",
        "test-0",
        "test-1",
    ]
    assert limited_labels.tolist() == [0, 1, 0, 1, 0, 1]


def _load_spatial_cli_script() -> ModuleType:
    script_path = Path("scripts/wavelet_course_spatial_run.py")
    spec = importlib.util.spec_from_file_location("wavelet_course_spatial_run", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spatial_cli_quick_resolves_five_epochs() -> None:
    module = _load_spatial_cli_script()

    args = module.build_parser().parse_args(["--quick"])
    config = module.resolve_config({}, args)

    assert config["quick_run"] is True
    assert config["spatial_hidden_wavelet"]["max_epochs"] == 5
    assert config["spatial_hidden_wavelet"]["batch_size"] == 32
    assert config["device"] == "cuda:0"
