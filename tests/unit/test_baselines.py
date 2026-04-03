from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from mind.evaluation.baselines import (
    build_feature_variant_frames,
    build_no_manifold_feature_frame,
    build_output_baseline_frame,
    build_raw_model_yes_no_baseline,
    compute_bootstrap_confidence_intervals,
    evaluate_feature_frame,
    evaluate_feature_frame_across_random_states,
    resolve_yes_no_token_ids,
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


def test_build_no_manifold_feature_frame_raises_on_entries_without_reference_coverage() -> None:
    reference_bank = {
        "dog": {
            8: torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        }
    }

    with pytest.raises(ValueError, match="Missing reference coverage"):
        build_no_manifold_feature_frame(
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


def test_resolve_yes_no_token_ids_keeps_single_token_variants_only() -> None:
    class FakeTokenizer:
        mapping = {
            "yes": [10],
            " yes": [11],
            "Yes": [12],
            "no": [20],
            " no": [21],
            "No": [22],
        }

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return self.mapping.get(text, [99, 100])

    token_ids = resolve_yes_no_token_ids(FakeTokenizer())

    assert token_ids == {"yes": [10, 11, 12], "no": [20, 21, 22]}


def test_build_output_baseline_frame_uses_first_token_logits_and_hallucination_labels() -> None:
    frame = build_output_baseline_frame(
        [
            {
                "sample_id": "grounded-yes",
                "image_id": 1,
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "first_token_logits": torch.tensor([0.0, 3.0, 1.0, -2.0], dtype=torch.float32),
            },
            {
                "sample_id": "hallucinated-yes",
                "image_id": 2,
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "first_token_logits": torch.tensor([0.0, 1.5, 0.5, -1.0], dtype=torch.float32),
            },
            {
                "sample_id": "grounded-no",
                "image_id": 3,
                "label": 0,
                "parsed_answer": 0,
                "subset": "popular",
                "object_name": "dog",
                "first_token_logits": torch.tensor([0.0, -1.0, 2.0, 0.5], dtype=torch.float32),
            },
        ],
        yes_token_ids=[1],
        no_token_ids=[2],
    )

    assert list(frame["sample_id"]) == ["grounded-yes", "hallucinated-yes", "grounded-no"]
    assert list(frame["label"]) == [0, 1, 0]
    assert "p_yes" in frame.columns
    assert "yes_logit_margin" in frame.columns
    assert "chosen_answer_confidence" in frame.columns
    assert frame.loc[0, "p_yes"] > frame.loc[2, "p_yes"]
    assert frame.loc[0, "yes_logit_margin"] > 0.0
    assert frame.loc[2, "yes_logit_margin"] < 0.0
    assert frame.loc[0, "chosen_answer_confidence"] > frame.loc[2, "chosen_answer_confidence"]


def test_build_feature_variant_frames_creates_clean_ablation_views() -> None:
    features = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "image_id": 1,
                "ground_truth_label": 0,
                "answer_label": 1,
                "label": 1,
                "subset": "popular",
                "object_name": "dog",
                "raw_drift_0": 0.2,
                "raw_drift_1": 0.4,
                "cal_drift_0": 1.0,
                "cal_drift_1": 3.0,
                "cal_mean_drift": 2.0,
                "cal_max_drift": 3.0,
                "cal_approx_energy": 5.0,
                "cal_detail_energy_l1": 1.5,
                "cal_detail_energy_l2": 0.5,
            }
        ]
    )

    variants = build_feature_variant_frames(features)

    assert set(variants) == {
        "raw_curve_only",
        "raw_plus_calibrated_simple",
        "raw_plus_calibrated_full_curve",
        "raw_plus_calibrated_haar",
    }
    assert "raw_drift_0" in variants["raw_curve_only"].columns
    assert "cal_drift_0" not in variants["raw_curve_only"].columns
    assert {
        "cal_final_drift",
        "cal_drift_slope",
        "cal_drift_variance",
    }.issubset(variants["raw_plus_calibrated_simple"].columns)
    assert {"cal_drift_0", "cal_drift_1"}.issubset(variants["raw_plus_calibrated_full_curve"].columns)
    assert "cal_approx_energy" in variants["raw_plus_calibrated_haar"].columns
    assert "cal_drift_0" not in variants["raw_plus_calibrated_haar"].columns
    assert math.isclose(variants["raw_plus_calibrated_simple"].loc[0, "cal_final_drift"], 3.0)


def test_compute_bootstrap_confidence_intervals_returns_metric_bounds() -> None:
    results = pd.DataFrame(
        [
            {"sample_id": "a1", "image_id": 1, "object_name": "dog", "label": 0, "prediction": 0, "score": 0.1},
            {"sample_id": "a2", "image_id": 1, "object_name": "dog", "label": 1, "prediction": 1, "score": 0.9},
            {"sample_id": "b1", "image_id": 2, "object_name": "cat", "label": 0, "prediction": 0, "score": 0.2},
            {"sample_id": "b2", "image_id": 2, "object_name": "cat", "label": 1, "prediction": 1, "score": 0.8},
        ]
    )

    intervals = compute_bootstrap_confidence_intervals(
        results,
        group_column="image_id",
        n_resamples=16,
        random_state=7,
    )

    assert "roc_auc" in intervals
    assert intervals["roc_auc"]["lower"] <= intervals["roc_auc"]["upper"]
    assert intervals["pr_auc"]["lower"] <= intervals["pr_auc"]["upper"]


def test_evaluate_feature_frame_across_random_states_reports_each_seed() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "object_name": "dog" if index < 4 else "cat",
                "label": index % 2,
                "score_feature": float(index),
            }
            for index in range(8)
        ]
    )

    summary = evaluate_feature_frame_across_random_states(
        frame,
        columns=["score_feature"],
        split_strategy="image_grouped",
        random_states=[3, 5],
        num_folds=2,
    )

    assert summary["random_state"].tolist() == [3, 5]
    assert {"roc_auc", "pr_auc", "tpr_at_fpr_0.01"}.issubset(summary.columns)
