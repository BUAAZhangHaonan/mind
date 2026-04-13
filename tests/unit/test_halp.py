from __future__ import annotations

import pandas as pd
import pytest
import torch

import mind.comparators.halp as halp_module
from mind.comparators.halp import (
    HALPProbeConfig,
    build_halp_probe_frame,
    build_halp_probe_frames,
    evaluate_halp_nested_from_readout_entries,
    evaluate_halp_nested,
    resolve_halp_probe_names,
    resolve_halp_layer_indices,
)


def test_resolve_halp_layer_indices_match_official_five_layer_schedule() -> None:
    assert resolve_halp_layer_indices(5) == [0, 1, 2, 3, 4]
    assert resolve_halp_layer_indices(8) == [0, 2, 4, 6, 7]
    assert resolve_halp_layer_indices(32) == [0, 8, 16, 24, 31]
    assert resolve_halp_layer_indices(1) == [0]


def test_resolve_halp_probe_names_returns_exactly_eleven_official_probes() -> None:
    entries = [
        {
            "sample_id": "sample-1",
            "image_id": 10,
            "label": 0,
            "parsed_answer": 1,
            "subset": "popular",
            "object_name": "dog",
            "query_hidden_states": torch.zeros((32, 2), dtype=torch.float32),
            "vision_token_hidden_states": torch.zeros((32, 2), dtype=torch.float32),
            "vision_features": torch.zeros((2, 2), dtype=torch.float32),
        }
    ]

    probe_names = resolve_halp_probe_names(entries)

    assert len(probe_names) == 11
    assert probe_names[0] == "vision_only"
    assert probe_names[1:] == [
        "vision_token_layer_0",
        "query_token_layer_0",
        "vision_token_layer_8",
        "query_token_layer_8",
        "vision_token_layer_16",
        "query_token_layer_16",
        "vision_token_layer_24",
        "query_token_layer_24",
        "vision_token_layer_31",
        "query_token_layer_31",
    ]


def test_build_halp_probe_frames_extracts_vf_vt_and_qt_vectors() -> None:
    entries = [
        {
            "sample_id": "sample-1",
            "image_id": 10,
            "label": 0,
            "parsed_answer": 1,
            "subset": "popular",
            "object_name": "dog",
            "full_hidden_states": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
            "query_token_index": 2,
            "vision_token_span": [0, 1],
            "vision_features": torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32),
        }
    ]

    frames = build_halp_probe_frames(entries, layer_indices=[0, 3])

    assert set(frames) == {
        "vision_only",
        "vision_token_layer_0",
        "vision_token_layer_3",
        "query_token_layer_0",
        "query_token_layer_3",
    }
    vision_row = frames["vision_only"].iloc[0]
    assert vision_row["feature_0"] == 11.0
    assert vision_row["feature_1"] == 12.0
    vt_row = frames["vision_token_layer_0"].iloc[0]
    assert vt_row["feature_0"] == 2.0
    assert vt_row["feature_1"] == 3.0
    qt_row = frames["query_token_layer_3"].iloc[0]
    assert qt_row["feature_0"] == 22.0
    assert qt_row["feature_1"] == 23.0


def test_evaluate_halp_nested_selects_best_probe() -> None:
    rows: list[dict[str, object]] = []
    for index in range(12):
        label = 1 if index % 2 == 0 else 0
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "ground_truth_label": 0,
                "answer_label": 1 if label else 0,
                "label": label,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
            }
        )

    metadata = pd.DataFrame(rows)
    good_frame = metadata.assign(feature_0=[float(row["label"]) for row in rows])
    bad_frame = metadata.assign(feature_0=[0.5 for _ in rows])

    metrics, results, selection = evaluate_halp_nested(
        {
            "good_probe": good_frame,
            "bad_probe": bad_frame,
        },
        split_strategy="image_grouped",
        num_folds=3,
        random_state=13,
        inner_candidate_folds=(3, 2),
        probe_config=HALPProbeConfig(hidden_dims=(8, 4), batch_size=4, epochs=10, random_state=7),
    )

    assert metrics["roc_auc"] > 0.95
    assert set(results["selected_probe"]) == {"good_probe"}
    assert set(selection["selected_probe"]) == {"good_probe"}


def test_evaluate_halp_nested_filters_to_supported_objects() -> None:
    rows: list[dict[str, object]] = []
    for index in range(16):
        label = 1 if index % 2 == 0 else 0
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "ground_truth_label": 0,
                "answer_label": 1 if label else 0,
                "label": label,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
            }
        )

    metadata = pd.DataFrame(rows)
    probe_frame = metadata.assign(feature_0=[float(row["label"]) for row in rows])

    metrics, results, selection = evaluate_halp_nested(
        {"probe": probe_frame},
        split_strategy="object_heldout",
        num_folds=2,
        random_state=13,
        inner_candidate_folds=(2,),
        probe_config=HALPProbeConfig(hidden_dims=(8, 4), batch_size=4, epochs=10, random_state=7),
        supported_object_names=["object-0", "object-1", "object-2", "object-3"],
    )

    assert metrics["roc_auc"] > 0.95
    assert set(results["object_name"]) <= {"object-0", "object-1", "object-2", "object-3"}
    assert set(selection["selected_probe"]) == {"probe"}


def test_build_halp_probe_frame_builds_one_probe_without_materializing_all() -> None:
    entries = [
        {
            "sample_id": "sample-1",
            "image_id": 10,
            "label": 0,
            "parsed_answer": 1,
            "subset": "popular",
            "object_name": "dog",
            "query_hidden_states": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "vision_token_hidden_states": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            "vision_features": torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32),
        }
    ]

    frame = build_halp_probe_frame(entries, "vision_token_layer_1")

    assert frame.loc[0, "feature_0"] == 7.0
    assert frame.loc[0, "feature_1"] == 8.0


def test_evaluate_halp_nested_from_readout_entries_matches_frame_path() -> None:
    entries: list[dict[str, object]] = []
    for index in range(12):
        label = 1 if index % 2 == 0 else 0
        informative_value = 1.0 if label else 0.0
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index // 2,
                "label": 0,
                "parsed_answer": 0 if label else 1,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
                "query_hidden_states": torch.tensor([[informative_value]], dtype=torch.float32),
                "vision_token_hidden_states": torch.tensor([[0.5]], dtype=torch.float32),
                "vision_features": torch.tensor([[informative_value]], dtype=torch.float32),
            }
        )

    frame_metrics, frame_results, frame_selection = evaluate_halp_nested(
        build_halp_probe_frames(entries, layer_indices=[0]),
        split_strategy="image_grouped",
        num_folds=3,
        random_state=13,
        inner_candidate_folds=(3, 2),
        probe_config=HALPProbeConfig(hidden_dims=(8, 4), batch_size=4, epochs=10, random_state=7),
    )
    lazy_metrics, lazy_results, lazy_selection = evaluate_halp_nested_from_readout_entries(
        entries,
        split_strategy="image_grouped",
        num_folds=3,
        random_state=13,
        inner_candidate_folds=(3, 2),
        probe_config=HALPProbeConfig(hidden_dims=(8, 4), batch_size=4, epochs=10, random_state=7),
        layer_indices=[0],
    )

    assert lazy_metrics["roc_auc"] == frame_metrics["roc_auc"]
    assert lazy_metrics["pr_auc"] == frame_metrics["pr_auc"]
    assert set(lazy_results["selected_probe"]) == set(frame_results["selected_probe"])
    assert set(lazy_selection["selected_probe"]) == set(frame_selection["selected_probe"])


def test_evaluate_halp_nested_from_readout_entries_uses_highest_valid_outer_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries: list[dict[str, object]] = []
    for index in range(12):
        label = 1 if index % 2 == 0 else 0
        informative_value = 1.0 if label else 0.0
        entries.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index,
                "label": 0,
                "parsed_answer": 0 if label else 1,
                "subset": "popular",
                "object_name": f"object-{index // 2}",
                "query_hidden_states": torch.tensor([[informative_value]], dtype=torch.float32),
                "vision_token_hidden_states": torch.tensor([[0.5]], dtype=torch.float32),
                "vision_features": torch.tensor([[informative_value]], dtype=torch.float32),
            }
        )

    captured_num_folds: list[int] = []
    original_build_splits = halp_module._build_sample_id_splits

    def _capture_build_splits(frame, *, split_strategy, test_size, random_state, num_folds):
        captured_num_folds.append(int(num_folds))
        return original_build_splits(
            frame,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=random_state,
            num_folds=num_folds,
        )

    monkeypatch.setattr(halp_module, "_build_sample_id_splits", _capture_build_splits)
    monkeypatch.setattr(halp_module, "_resolve_outer_num_folds", lambda *args, **kwargs: 2)

    metrics, results, selection = evaluate_halp_nested_from_readout_entries(
        entries,
        split_strategy="object_heldout",
        num_folds=5,
        random_state=13,
        inner_candidate_folds=(2,),
        probe_config=HALPProbeConfig(hidden_dims=(8, 4), batch_size=4, epochs=10, random_state=7),
        layer_indices=[0],
    )

    assert "roc_auc" in metrics
    assert captured_num_folds[0] == 2
    assert set(results["fold"]) == {0, 1}
    assert set(selection["outer_fold"]) == {0, 1}
