from __future__ import annotations

import pytest
import torch

from mind.geometry.local_neighborhood import compute_local_features, select_local_references


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for local neighborhood tests")
    return torch.device("cuda")


def test_select_local_references_returns_angular_knn_indices_and_vectors() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([1.0, 0.0, 0.0], device=device)
    bank = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        device=device,
    )

    indices, vectors = select_local_references(query, bank, k=2)

    assert indices.device.type == "cuda"
    assert vectors.device.type == "cuda"
    assert indices.tolist() == [1, 2]
    torch.testing.assert_close(vectors, bank[indices])


def test_compute_local_features_matches_simple_line_geometry() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([2.5, 1.0, 0.0], device=device)
    local_references = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        device=device,
    )

    features = compute_local_features(query, local_references)

    assert set(features) == {
        "centroid_angular_distance",
        "local_pca_residual",
        "mean_angular_distance",
        "std_angular_distance",
        "centroid_euclidean_distance",
    }
    for value in features.values():
        assert value.device.type == "cuda"
        assert value.ndim == 0
        assert torch.isfinite(value)

    expected_angle = torch.atan2(torch.tensor(1.0, device=device), torch.tensor(2.5, device=device))
    torch.testing.assert_close(features["centroid_angular_distance"], expected_angle, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(features["mean_angular_distance"], expected_angle, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(features["std_angular_distance"], torch.tensor(0.0, device=device), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(features["centroid_euclidean_distance"], torch.tensor(1.0, device=device))
    torch.testing.assert_close(features["local_pca_residual"], torch.tensor(1.0, device=device), atol=1e-5, rtol=1e-5)


def test_local_neighborhood_requires_cuda_inputs() -> None:
    query = torch.ones(3)
    bank = torch.ones((4, 3))

    with pytest.raises(ValueError, match="CUDA"):
        select_local_references(query, bank, k=2)
    with pytest.raises(ValueError, match="CUDA"):
        compute_local_features(query, bank)
