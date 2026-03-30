from __future__ import annotations

import pandas as pd
import torch

from mind.evaluation.baselines import (
    build_no_manifold_feature_frame,
    build_raw_model_yes_no_baseline,
    evaluate_feature_frame,
)


def test_build_raw_model_yes_no_baseline_tracks_counts_and_metrics() -> None:
    baseline = build_raw_model_yes_no_baseline(
        [
            {
                "sample_id": "sample-1",
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
            },
            {
                "sample_id": "sample-2",
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
            },
            {
                "sample_id": "sample-3",
                "label": 1,
                "parsed_answer": None,
                "subset": "popular",
                "object_name": "dog",
            },
        ]
    )

    assert baseline["rows_total"] == 3
    assert baseline["rows_parsed"] == 2
    assert baseline["rows_unparsed"] == 1
    assert baseline["hallucination_positives"] == 1
    assert baseline["yes_no"]["accuracy"] == 0.5


def test_build_no_manifold_feature_frame_builds_wavelet_features() -> None:
    reference_bank = {
        "dog": {
            8: torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            13: torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        }
    }

    frame = build_no_manifold_feature_frame(
        cache_entries=[
            {
                "sample_id": "sample-1",
                "image_id": 11,
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8, 13],
                "layer_vectors": torch.tensor([[0.2, 0.2, 0.1], [0.2, 0.2, 0.3]]),
            }
        ],
        reference_bank=reference_bank,
        reference_stats={
            "dog": {
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
            }
        },
    )

    assert isinstance(frame, pd.DataFrame)
    assert list(frame["label"]) == [1]
    assert "cal_approx_energy" in frame.columns
    assert "cal_detail_energy_l1" in frame.columns
    assert "raw_drift_0" in frame.columns
    assert frame.loc[0, "object_name"] == "dog"


def test_build_no_manifold_feature_frame_skips_entries_without_reference_coverage() -> None:
    reference_bank = {
        "dog": {
            8: torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        }
    }

    frame = build_no_manifold_feature_frame(
        cache_entries=[
            {
                "sample_id": "covered",
                "image_id": 21,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.2, 0.1]]),
            },
            {
                "sample_id": "missing",
                "image_id": 22,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.2, 0.1]]),
            },
        ],
        reference_bank=reference_bank,
        reference_stats={
            "dog": {
                8: {
                    "residual_mean": 0.1,
                    "residual_std": 0.2,
                    "neighbor_residual_mean": 0.1,
                    "neighbor_residual_std": 0.2,
                },
            }
        },
    )

    assert list(frame["sample_id"]) == ["covered"]


def test_build_no_manifold_feature_frame_can_use_shared_bank() -> None:
    reference_bank = {
        "__shared__": {
            8: torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        }
    }

    frame = build_no_manifold_feature_frame(
        cache_entries=[
            {
                "sample_id": "dog-sample",
                "image_id": 21,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.2, 0.1, 0.0]]),
            },
            {
                "sample_id": "cat-sample",
                "image_id": 22,
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.4, 0.2, 0.3]]),
            },
        ],
        reference_bank=reference_bank,
        reference_stats={
            "__shared__": {
                8: {
                    "residual_mean": 0.1,
                    "residual_std": 0.2,
                    "neighbor_residual_mean": 0.1,
                    "neighbor_residual_std": 0.2,
                },
            }
        },
        bank_scope="shared",
    )

    assert sorted(frame["sample_id"].tolist()) == ["cat-sample", "dog-sample"]
    assert "raw_drift_0" in frame.columns


def test_evaluate_feature_frame_uses_image_grouped_out_of_fold_results() -> None:
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

    metrics, results = evaluate_feature_frame(
        frame,
        columns=["raw_drift_0", "raw_mean_drift"],
        split_strategy="image_grouped",
        random_state=17,
        num_folds=2,
    )

    assert len(results) == len(frame)
    assert sorted(results["fold"].unique().tolist()) == [0, 1]
    assert sorted(results["sample_id"].tolist()) == sorted(frame["sample_id"].tolist())
    assert "pr_auc" in metrics
    assert "tpr_at_fpr_0.01" in metrics
