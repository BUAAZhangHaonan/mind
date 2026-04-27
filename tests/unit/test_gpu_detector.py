from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "train_gpu_detector.py"
SPEC = importlib.util.spec_from_file_location("train_gpu_detector", SCRIPT_PATH)
assert SPEC is not None
gpu_detector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gpu_detector)


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU detector tests")
    return torch.device("cuda")


def test_full_curve_resolves_prompt_defined_recipe() -> None:
    frame = pd.DataFrame(
        [
            {
                "label": 0,
                "raw_drift_1": 0.1,
                "raw_drift_0": 0.0,
                "cal_drift_0": 9.0,
                "cal_mean_drift": 0.2,
                "cal_max_drift": 0.3,
                "cal_final_drift": 0.4,
                "cal_drift_slope": 0.5,
                "cal_drift_variance": 0.6,
            }
        ]
    )

    columns = gpu_detector.resolve_columns(frame, "auto", "full_curve")

    assert columns == [
        "raw_drift_0",
        "raw_drift_1",
        "cal_mean_drift",
        "cal_max_drift",
        "cal_final_drift",
        "cal_drift_slope",
        "cal_drift_variance",
    ]


def test_drift_only_keeps_raw_summary_columns_when_available() -> None:
    frame = pd.DataFrame(
        [
            {
                "label": 0,
                "raw_drift_1": 0.1,
                "raw_drift_0": 0.0,
                "raw_max_drift": 0.1,
                "raw_mean_drift": 0.05,
                "raw_peak_layer_index": 1.0,
                "cal_mean_drift": 0.2,
            }
        ]
    )

    columns = gpu_detector.resolve_columns(frame, "auto", "drift_only")

    assert columns == [
        "raw_drift_0",
        "raw_drift_1",
        "raw_max_drift",
        "raw_mean_drift",
        "raw_peak_layer_index",
    ]


def test_all_features_excludes_metadata_for_linear_probe() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "image_id": 1,
                "label": 0,
                "subset": "popular",
                "object_name": "cat",
                "ground_truth_label": 1,
                "answer_label": 1,
                "hidden_0": 0.1,
                "hidden_1": 0.2,
            }
        ]
    )

    assert gpu_detector.resolve_columns(frame, "all_features", "full_curve") == ["hidden_0", "hidden_1"]


def test_torch_auc_metrics_match_known_ranking_on_cuda() -> None:
    device = _cuda_or_skip()
    labels = torch.tensor([0, 1, 0, 1], device=device)
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8], device=device)

    assert torch.isclose(gpu_detector.roc_auc_torch(labels, scores), torch.tensor(1.0, device=device))
    assert torch.isclose(gpu_detector.average_precision_torch(labels, scores), torch.tensor(1.0, device=device))


def test_cuda_training_scoring_and_bootstrap_stay_on_cuda() -> None:
    device = _cuda_or_skip()
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "object_name": "dog" if index < 8 else "cat",
                "label": index % 2,
                "raw_drift_0": float(index % 2) + 0.02 * index,
                "raw_drift_1": float(index % 2) - 0.01 * index,
                "cal_mean_drift": 0.5 * float(index % 2) + 0.01 * index,
            }
            for index in range(16)
        ]
    )

    metrics, predictions, states = gpu_detector.evaluate_frame(
        frame,
        columns=["raw_drift_0", "raw_drift_1", "cal_mean_drift"],
        split_strategy="image_grouped",
        device=device,
        num_folds=2,
        bootstrap_resamples=8,
        random_state=3,
    )

    assert predictions["fold"].nunique() == 2
    assert {"roc_auc", "pr_auc", "roc_auc_ci_lower", "pr_auc_ci_upper"}.issubset(metrics)
    assert metrics["roc_auc"] > 0.95
    assert all(state.weights.device.type == "cuda" for state in states)
    assert all(state.mean.device.type == "cuda" for state in states)
    assert all(state.scale.device.type == "cuda" for state in states)
    assert set(predictions["image_id"]) == set(frame["image_id"])


def test_cpu_inputs_raise_for_cuda_only_helpers() -> None:
    cpu_labels = torch.tensor([0, 1])
    cpu_scores = torch.tensor([0.2, 0.8])

    with pytest.raises(ValueError, match="CUDA"):
        gpu_detector._require_cuda_tensor("labels", cpu_labels)
    with pytest.raises(ValueError, match="CUDA"):
        gpu_detector.compute_metrics_torch(cpu_labels, cpu_scores, require_cuda=True)
