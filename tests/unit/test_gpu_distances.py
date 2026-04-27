from __future__ import annotations

import pytest
import torch

from mind.geometry.gpu_distances import (
    batch_angular_distance,
    batch_euclidean_distance,
    centroid_angular_distance_gpu,
    centroid_euclidean_distance_gpu,
    knn_angular_distance_gpu,
)


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU distance primitive tests")
    return torch.device("cuda")


def _cpu_angular_distance(query: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    query_normalized = torch.nn.functional.normalize(query.cpu(), dim=1)
    reference_normalized = torch.nn.functional.normalize(reference.cpu(), dim=1)
    cosine = query_normalized @ reference_normalized.T
    return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))


def _cpu_euclidean_distance(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    diff = query.cpu()[:, None, :] - reference.cpu()[None, :, :]
    return torch.sqrt((diff * diff).sum(dim=-1))


def test_batch_angular_distance_matches_cpu_reference_with_chunks() -> None:
    device = _cuda_or_skip()
    query = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        device=device,
    )
    reference = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        device=device,
    )

    distances = batch_angular_distance(
        query,
        reference,
        query_chunk_size=2,
        reference_chunk_size=2,
    )

    assert distances.device.type == "cuda"
    torch.testing.assert_close(
        distances.cpu(),
        _cpu_angular_distance(query, reference),
        atol=1e-6,
        rtol=1e-6,
    )


def test_batch_euclidean_distance_matches_cpu_reference_with_chunks() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 0.5]], device=device)
    reference = torch.tensor([[1.0, 0.0], [-2.0, 4.0], [3.0, -1.0]], device=device)

    distances = batch_euclidean_distance(
        query,
        reference,
        query_chunk_size=1,
        reference_chunk_size=2,
    )

    assert distances.device.type == "cuda"
    torch.testing.assert_close(
        distances.cpu(),
        _cpu_euclidean_distance(query, reference),
        atol=1e-6,
        rtol=1e-6,
    )


def test_knn_angular_distance_gpu_matches_exact_cpu_topk_mean_across_chunks() -> None:
    device = _cuda_or_skip()
    query = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        device=device,
    )
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        device=device,
    )

    distances = knn_angular_distance_gpu(
        query,
        reference,
        k=3,
        query_chunk_size=1,
        reference_chunk_size=2,
    )
    expected = torch.topk(
        _cpu_angular_distance(query, reference),
        k=3,
        dim=1,
        largest=False,
    ).values.mean(dim=1)

    assert distances.device.type == "cuda"
    torch.testing.assert_close(distances.cpu(), expected, atol=1e-6, rtol=1e-6)


def test_centroid_distances_match_cpu_reference() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], device=device)
    reference = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], device=device)
    centroid = reference.cpu().mean(dim=0, keepdim=True)

    angular = centroid_angular_distance_gpu(query, reference, query_chunk_size=2)
    euclidean = centroid_euclidean_distance_gpu(query, reference, query_chunk_size=2)

    torch.testing.assert_close(
        angular.cpu(),
        _cpu_angular_distance(query, centroid).squeeze(1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        euclidean.cpu(),
        _cpu_euclidean_distance(query, centroid).squeeze(1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_cpu_inputs_raise() -> None:
    query = torch.ones((2, 3))
    reference = torch.ones((4, 3))

    with pytest.raises(ValueError, match="CUDA"):
        batch_angular_distance(query, reference)
    with pytest.raises(ValueError, match="CUDA"):
        batch_euclidean_distance(query, reference)
    with pytest.raises(ValueError, match="CUDA"):
        knn_angular_distance_gpu(query, reference, k=2)
    with pytest.raises(ValueError, match="CUDA"):
        centroid_angular_distance_gpu(query, reference)
    with pytest.raises(ValueError, match="CUDA"):
        centroid_euclidean_distance_gpu(query, reference)
