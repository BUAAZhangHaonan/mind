from __future__ import annotations

import pandas as pd
import torch

from mind.comparators.halp import (
    HALPProbeConfig,
    build_halp_probe_frames,
    evaluate_halp_nested,
    resolve_halp_layer_indices,
)


def test_resolve_halp_layer_indices_uses_all_layers() -> None:
    assert resolve_halp_layer_indices(5) == [0, 1, 2, 3, 4]
    assert resolve_halp_layer_indices(1) == [0]


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


def test_evaluate_halp_nested_filters_object_heldout_to_supported_objects() -> None:
    rows: list[dict[str, object]] = []
    objects = ["dog", "dog", "cat", "cat", "bus", "bus", "cup", "cup", "tree", "tree"]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    for index, (object_name, label) in enumerate(zip(objects, labels)):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "image_id": index,
                "ground_truth_label": 0,
                "answer_label": label,
                "label": label,
                "subset": "popular",
                "object_name": object_name,
            }
        )

    metadata = pd.DataFrame(rows)
    good_frame = metadata.assign(feature_0=[float(label) for label in labels])
    bad_frame = metadata.assign(feature_0=[0.5 for _ in labels])

    metrics, results, _selection = evaluate_halp_nested(
        {
            "good_probe": good_frame,
            "bad_probe": bad_frame,
        },
        split_strategy="object_heldout",
        num_folds=2,
        random_state=13,
        inner_candidate_folds=(2,),
        supported_object_names=["dog", "cat", "bus", "cup"],
        probe_config=HALPProbeConfig(hidden_dims=(8, 4), batch_size=2, epochs=10, random_state=7),
    )

    assert metrics["roc_auc"] > 0.95
    assert set(results["object_name"]) == {"dog", "cat", "bus", "cup"}


def test_evaluate_halp_nested_rejects_supported_object_mismatch() -> None:
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
