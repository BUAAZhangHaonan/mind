from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mind.detectors import fit_logistic_detector
from mind.drift import compute_drift_curve, standardize_drift_curve
from mind.evaluation import compute_object_hallucination_label
from mind.wavelets import extract_wavelet_features


TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_detector.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_detector", TRAIN_SCRIPT)
train_detector = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC is not None and TRAIN_SPEC.loader is not None
TRAIN_SPEC.loader.exec_module(train_detector)


def test_compute_drift_curve_scores_each_layer_against_reference_bank() -> None:
    reference_bank = {
        "dog": {
            8: torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            13: torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
        }
    }
    layer_vectors = torch.tensor(
        [
            [0.3, 0.4, 0.0],
            [0.3, 0.4, 0.6],
        ]
    )

    drift = compute_drift_curve(
        layer_vectors=layer_vectors,
        selected_layers=[8, 13],
        object_name="dog",
        reference_bank=reference_bank,
        k_neighbors=4,
    )

    assert len(drift) == 2
    assert drift[0] < 1e-5
    assert drift[1] > 0.4


def test_standardize_drift_curve_returns_zero_mean_unit_scale() -> None:
    curve = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    standardized = standardize_drift_curve(curve)

    assert np.isclose(float(standardized.mean()), 0.0, atol=1e-6)
    assert np.isclose(float(standardized.std()), 1.0, atol=1e-6)


def test_extract_wavelet_features_returns_raw_curve_and_energy_terms() -> None:
    curve = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)

    features = extract_wavelet_features(curve)

    assert features["drift_0"] == 0.0
    assert features["drift_7"] == 3.0
    assert "approx_energy" in features
    assert "detail_energy_l1" in features
    assert features["max_drift"] == 3.0


def test_fit_logistic_detector_learns_simple_separable_problem() -> None:
    features = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.2],
            [2.0, 2.1],
            [2.1, 2.2],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    detector = fit_logistic_detector(features, labels)
    predictions = detector.predict(features)

    assert predictions.tolist() == [0, 0, 1, 1]


def test_compute_object_hallucination_label_only_marks_unsupported_positive_answers() -> None:
    assert compute_object_hallucination_label(ground_truth_label=0, answer_label=1) == 1
    assert compute_object_hallucination_label(ground_truth_label=1, answer_label=1) == 0
    assert compute_object_hallucination_label(ground_truth_label=0, answer_label=0) == 0
    assert compute_object_hallucination_label(ground_truth_label=1, answer_label=0) == 0


def test_build_feature_output_path_uses_experiment_and_split(tmp_path: Path) -> None:
    output_path = train_detector.build_feature_output_path(
        output_root=tmp_path,
        experiment_name="smoke-qwen3-vl",
        split="popular",
    )

    assert output_path == tmp_path / "smoke-qwen3-vl" / "popular.parquet"


def test_feature_columns_excludes_label_metadata_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "label": 1,
                "subset": "popular",
                "object_name": "dog",
                "ground_truth_label": 0,
                "answer_label": 1,
                "drift_0": 0.5,
                "approx_energy": 1.2,
            }
        ]
    )

    columns = train_detector.feature_columns(frame)

    assert columns == ["drift_0", "approx_energy"]


def test_split_train_eval_frame_preserves_both_classes() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": f"sample-{index}", "label": index % 2, "drift_0": float(index)}
            for index in range(10)
        ]
    )

    train_frame, eval_frame = train_detector.split_train_eval_frame(
        frame,
        test_size=0.4,
        random_state=7,
    )

    assert len(train_frame) == 6
    assert len(eval_frame) == 4
    assert sorted(train_frame["label"].unique().tolist()) == [0, 1]
    assert sorted(eval_frame["label"].unique().tolist()) == [0, 1]
