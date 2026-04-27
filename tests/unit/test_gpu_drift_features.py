from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "build_gpu_drift_features.py"
SPEC = importlib.util.spec_from_file_location("build_gpu_drift_features", SCRIPT_PATH)
assert SPEC is not None
gpu_features = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gpu_features)


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU drift feature tests")
    return torch.device("cuda")


def test_prompt_full_curve_features_use_cuda_summary_recipe() -> None:
    device = _cuda_or_skip()
    raw_curve = torch.tensor([1.0, 2.0, 4.0], device=device)
    calibrated_curve = torch.tensor([-1.0, 0.0, 2.0], device=device)

    features = gpu_features.build_prompt_full_curve_features_gpu(
        raw_curve=raw_curve,
        calibrated_curve=calibrated_curve,
    )

    assert set(features) == {
        "raw_drift_0",
        "raw_drift_1",
        "raw_drift_2",
        "cal_mean_drift",
        "cal_max_drift",
        "cal_final_drift",
        "cal_drift_slope",
        "cal_drift_variance",
    }
    assert features["raw_drift_2"] == pytest.approx(4.0)
    assert features["cal_mean_drift"] == pytest.approx(1.0 / 3.0)
    assert features["cal_max_drift"] == pytest.approx(2.0)
    assert features["cal_final_drift"] == pytest.approx(2.0)
    assert features["cal_drift_slope"] == pytest.approx(1.5)
    assert features["cal_drift_variance"] == pytest.approx(14.0 / 9.0)


def test_no_manifold_frame_computes_nearest_neighbor_residual_on_cuda() -> None:
    device = _cuda_or_skip()
    entry = {
        "sample_id": "sample-1",
        "image_id": 7,
        "label": 0,
        "parsed_answer": 1,
        "subset": "main",
        "object_name": "cat",
        "selected_layers": [0],
        "layer_vectors": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    }
    reference_bank = {
        "cat": {
            0: torch.tensor(
                [
                    [0.0, 0.0],
                    [3.0, 0.0],
                    [10.0, 0.0],
                ],
                dtype=torch.float32,
            )
        }
    }
    reference_stats = {
        "cat": {
            0: {
                "residual_mean": 0.0,
                "residual_std": 1.0,
                "neighbor_residual_mean": 0.0,
                "neighbor_residual_std": 1.0,
            }
        }
    }

    frame = gpu_features.build_gpu_feature_frame(
        cache_entries=[entry],
        reference_bank=reference_bank,
        reference_stats=reference_stats,
        bank_scope="object",
        curve_type="no_manifold",
        batch_size=1,
        k_neighbors=3,
        device=device,
    )

    assert len(frame) == 1
    assert frame.loc[0, "sample_id"] == "sample-1"
    assert frame.loc[0, "image_id"] == 7
    assert frame.loc[0, "label"] == 1
    assert frame.loc[0, "raw_drift_0"] == pytest.approx(0.25)
    assert frame.loc[0, "raw_max_drift"] == pytest.approx(0.25)
    assert frame.loc[0, "raw_mean_drift"] == pytest.approx(0.25)
    assert frame.loc[0, "raw_peak_layer_index"] == pytest.approx(0.0)
    assert frame.loc[0, "cal_mean_drift"] == pytest.approx(0.25)


def test_cuda_device_is_required() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        gpu_features.resolve_cuda_device("cpu")
