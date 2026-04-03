from __future__ import annotations

import pandas as pd

from mind.comparators.glsim import (
    evaluate_glsim_nested,
    find_subsequence_start,
    resolve_glsim_layer_indices,
)


def test_resolve_glsim_layer_indices_matches_quarter_depths() -> None:
    assert resolve_glsim_layer_indices(32) == [0, 8, 16, 24, 31]


def test_find_subsequence_start_returns_first_match() -> None:
    assert find_subsequence_start([1, 7, 8, 9, 7, 8], [7, 8]) == 1
    assert find_subsequence_start([1, 2, 3], [4]) is None


def test_evaluate_glsim_nested_selects_best_config() -> None:
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
                "global_i0_t0": float(label),
                "local_i0_t0_k1": float(label),
                "global_i1_t1": 0.5,
                "local_i1_t1_k1": 0.5,
            }
        )
    score_frame = pd.DataFrame(rows)

    metrics, results, selection = evaluate_glsim_nested(
        score_frame,
        image_layers=[0, 1],
        text_layers=[0, 1],
        k_values=[1],
        w_values=[0.5],
        split_strategy="image_grouped",
        num_folds=3,
        random_state=13,
    )

    assert metrics["roc_auc"] > 0.95
    assert set(results["selected_config"]) == {"i0_t0_k1_w0.50"}
    assert set(selection["selected_config"]) == {"i0_t0_k1_w0.50"}
