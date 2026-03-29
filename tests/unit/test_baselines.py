from __future__ import annotations

import pandas as pd
import torch

from mind.evaluation.baselines import (
    build_no_manifold_feature_frame,
    build_raw_model_yes_no_baseline,
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
                "label": 0,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8, 13],
                "layer_vectors": torch.tensor([[0.2, 0.2, 0.1], [0.2, 0.2, 0.3]]),
            }
        ],
        reference_bank=reference_bank,
    )

    assert isinstance(frame, pd.DataFrame)
    assert list(frame["label"]) == [1]
    assert "approx_energy" in frame.columns
    assert "detail_energy_l1" in frame.columns
    assert "drift_0" in frame.columns
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
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "dog",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.2, 0.1]]),
            },
            {
                "sample_id": "missing",
                "label": 1,
                "parsed_answer": 1,
                "subset": "popular",
                "object_name": "cat",
                "selected_layers": [8],
                "layer_vectors": torch.tensor([[0.2, 0.1]]),
            },
        ],
        reference_bank=reference_bank,
    )

    assert list(frame["sample_id"]) == ["covered"]
