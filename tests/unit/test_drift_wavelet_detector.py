from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mind.detectors import fit_logistic_detector
from mind.drift import build_drift_features, calibrate_drift_curve, compute_drift_curve
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


def test_calibrate_drift_curve_uses_reference_stats_not_self_normalization() -> None:
    curve = np.array([10.0, 12.0], dtype=np.float32)

    calibrated = calibrate_drift_curve(
        curve,
        selected_layers=[8, 13],
        layer_stats={
            8: {"residual_mean": 8.0, "residual_std": 2.0},
            13: {"residual_mean": 10.0, "residual_std": 4.0},
        },
    )

    assert np.allclose(calibrated, np.array([1.0, 0.5], dtype=np.float32))
    assert not np.isclose(float(calibrated.mean()), 0.0, atol=1e-6)


def test_extract_wavelet_features_returns_raw_curve_and_energy_terms() -> None:
    curve = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)

    features = extract_wavelet_features(curve)

    assert features["drift_0"] == 0.0
    assert features["drift_7"] == 3.0
    assert "approx_energy" in features
    assert "detail_energy_l1" in features
    assert features["max_drift"] == 3.0


def test_build_drift_features_preserves_raw_magnitude_and_wavelet_terms_only_for_calibrated_curve() -> None:
    raw_curve = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    calibrated_curve = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)

    features = build_drift_features(raw_curve=raw_curve, calibrated_curve=calibrated_curve)

    assert features["raw_drift_0"] == 2.0
    assert features["raw_drift_3"] == 8.0
    assert features["raw_mean_drift"] == 5.0
    assert features["raw_max_drift"] == 8.0
    assert features["cal_drift_0"] == 0.0
    assert features["cal_drift_3"] == 1.5
    assert "cal_approx_energy" in features
    assert "cal_detail_energy_l1" in features
    assert "raw_approx_energy" not in features


def test_build_drift_features_preserves_magnitude_for_same_shape_curves() -> None:
    small = build_drift_features(
        raw_curve=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        calibrated_curve=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )
    large = build_drift_features(
        raw_curve=np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
        calibrated_curve=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )

    assert large["raw_mean_drift"] > small["raw_mean_drift"]
    assert large["raw_max_drift"] > small["raw_max_drift"]


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
                "image_id": 101,
                "label": 1,
                "subset": "popular",
                "object_name": "dog",
                "ground_truth_label": 0,
                "answer_label": 1,
                "raw_drift_0": 0.5,
                "cal_approx_energy": 1.2,
            }
        ]
    )

    columns = train_detector.feature_columns(frame)

    assert columns == ["raw_drift_0", "cal_approx_energy"]


def test_build_train_eval_splits_keeps_image_groups_disjoint() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "object_name": "dog" if index < 4 else "cat",
                "label": index % 2,
                "raw_drift_0": float(index),
            }
            for index in range(8)
        ]
    )

    splits = train_detector.build_train_eval_splits(
        frame,
        split_strategy="image_grouped",
        random_state=7,
        num_folds=2,
    )

    assert len(splits) == 2
    for _, train_frame, eval_frame in splits:
        assert set(train_frame["image_id"]).isdisjoint(set(eval_frame["image_id"]))
        assert sorted(train_frame["label"].unique().tolist()) == [0, 1]
        assert sorted(eval_frame["label"].unique().tolist()) == [0, 1]


def test_build_train_eval_splits_keeps_object_groups_disjoint() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": 100 + index,
                "object_name": "dog" if index < 2 else "cat" if index < 4 else "bus" if index < 6 else "chair",
                "label": 0 if index in {0, 2, 4, 6} else 1,
                "raw_drift_0": float(index),
            }
            for index in range(8)
        ]
    )

    splits = train_detector.build_train_eval_splits(
        frame,
        split_strategy="object_heldout",
        random_state=11,
        num_folds=2,
    )

    assert len(splits) == 2
    for _, train_frame, eval_frame in splits:
        assert set(train_frame["object_name"]).isdisjoint(set(eval_frame["object_name"]))
        assert sorted(train_frame["label"].unique().tolist()) == [0, 1]
        assert sorted(eval_frame["label"].unique().tolist()) == [0, 1]


def test_train_detector_frame_assigns_fold_column_for_grouped_evaluation() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "object_name": "dog" if index < 4 else "cat",
                "label": index % 2,
                "raw_drift_0": float(index),
                "raw_mean_drift": float(index) / 2.0,
            }
            for index in range(8)
        ]
    )

    _, results = train_detector.train_detector_frame(
        frame,
        columns=["raw_drift_0", "raw_mean_drift"],
        split_strategy="image_grouped",
        random_state=5,
        num_folds=2,
    )

    assert len(results) == len(frame)
    assert sorted(results["fold"].unique().tolist()) == [0, 1]
    assert sorted(results["sample_id"].tolist()) == sorted(frame["sample_id"].tolist())
