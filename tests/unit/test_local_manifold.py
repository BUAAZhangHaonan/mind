from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from mind.manifolds import (
    build_reference_bank,
    clean_reference_entries,
    compute_reference_bank_stats,
    fit_local_pca_manifold,
    normalized_normal_residual,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_manifolds.py"
SPEC = importlib.util.spec_from_file_location("build_manifolds", SCRIPT_PATH)
build_manifolds = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(build_manifolds)


def test_fit_local_pca_manifold_keeps_plane_queries_near_zero_residual() -> None:
    reference_vectors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.7, 0.0],
        ]
    )
    query_vector = torch.tensor([0.3, 0.4, 0.0])

    manifold = fit_local_pca_manifold(
        reference_vectors,
        query_vector,
        k_neighbors=4,
        variance_threshold=0.9,
    )

    residual = normalized_normal_residual(query_vector, manifold)
    assert residual < 1e-5


def test_normalized_normal_residual_grows_when_query_leaves_local_plane() -> None:
    reference_vectors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.7, 0.0],
        ]
    )
    query_vector = torch.tensor([0.3, 0.4, 0.6])

    manifold = fit_local_pca_manifold(
        reference_vectors,
        query_vector,
        k_neighbors=4,
        variance_threshold=0.9,
    )

    residual = normalized_normal_residual(query_vector, manifold)
    assert residual > 0.4


def test_fit_local_pca_manifold_accepts_float16_inputs() -> None:
    reference_vectors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float16,
    )
    query_vector = torch.tensor([0.3, 0.4, 0.0], dtype=torch.float16)

    manifold = fit_local_pca_manifold(reference_vectors, query_vector, k_neighbors=4)
    residual = normalized_normal_residual(query_vector, manifold)

    assert manifold.mean.dtype == torch.float32
    assert manifold.components.dtype == torch.float32
    assert residual < 1e-5


def test_build_reference_bank_groups_vectors_by_object_and_layer() -> None:
    entries = [
        {
            "object_name": "dog",
            "selected_layers": [8, 13],
            "layer_vectors": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        {
            "object_name": "dog",
            "selected_layers": [8, 13],
            "layer_vectors": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        },
    ]

    bank = build_reference_bank(entries)

    assert sorted(bank) == ["dog"]
    assert sorted(bank["dog"]) == [8, 13]
    assert bank["dog"][8].shape == (2, 2)


def test_clean_reference_entries_keeps_only_parsed_yes_rows() -> None:
    cleaned = clean_reference_entries(
        [
            {"sample_id": "yes", "parsed_answer": 1, "object_name": "dog"},
            {"sample_id": "no", "parsed_answer": 0, "object_name": "dog"},
            {"sample_id": "unparsed", "parsed_answer": None, "object_name": "dog"},
        ]
    )

    assert [entry["sample_id"] for entry in cleaned] == ["yes"]


def test_build_reference_bank_excludes_layers_below_min_support() -> None:
    entries = [
        {
            "sample_id": "sample-1",
            "parsed_answer": 1,
            "object_name": "dog",
            "selected_layers": [8],
            "layer_vectors": torch.tensor([[1.0, 2.0]]),
        },
        {
            "sample_id": "sample-2",
            "parsed_answer": 1,
            "object_name": "dog",
            "selected_layers": [8],
            "layer_vectors": torch.tensor([[3.0, 4.0]]),
        },
        {
            "sample_id": "sample-3",
            "parsed_answer": 1,
            "object_name": "dog",
            "selected_layers": [8],
            "layer_vectors": torch.tensor([[5.0, 6.0]]),
        },
    ]

    bank = build_reference_bank(entries, min_points=4)

    assert bank == {}


def test_compute_reference_bank_stats_returns_layerwise_counts_and_radius_summaries() -> None:
    entries = [
        {
            "sample_id": "sample-1",
            "parsed_answer": 1,
            "object_name": "dog",
            "selected_layers": [8],
            "layer_vectors": torch.tensor([[0.0, 0.0, 0.0]]),
        },
        {
            "sample_id": "sample-2",
            "parsed_answer": 1,
            "object_name": "dog",
            "selected_layers": [8],
            "layer_vectors": torch.tensor([[1.0, 0.0, 0.0]]),
        },
        {
            "sample_id": "sample-3",
            "parsed_answer": 1,
            "object_name": "dog",
            "selected_layers": [8],
            "layer_vectors": torch.tensor([[0.0, 1.0, 0.0]]),
        },
    ]

    stats = compute_reference_bank_stats(entries, k_neighbors=2)

    assert stats["dog"][8]["count"] == 3
    assert "residual_mean" in stats["dog"][8]
    assert "residual_std" in stats["dog"][8]
    assert "neighbor_radius_mean" in stats["dog"][8]
    assert "neighbor_radius_std" in stats["dog"][8]
    assert "neighbor_radius_q10" in stats["dog"][8]
    assert "neighbor_radius_q50" in stats["dog"][8]
    assert "neighbor_radius_q90" in stats["dog"][8]


def test_build_output_path_uses_object_and_layer_subdirectories(tmp_path: Path) -> None:
    output_path = build_manifolds.build_output_path(
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        object_name="dog",
        layer_index=8,
    )

    assert output_path == tmp_path / "qwen3-vl-8b" / "dog" / "layer-08.pt"
