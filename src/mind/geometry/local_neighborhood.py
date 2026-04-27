"""Query-local reference selection and feature computation on CUDA tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mind.geometry.gpu_distances import (
    DEFAULT_ACOS_EPS,
    DEFAULT_KNN_REFERENCE_CHUNK_SIZE,
    batch_angular_distance,
    centroid_angular_distance_gpu,
    centroid_euclidean_distance_gpu,
)


@dataclass(frozen=True)
class LocalFeatureConfig:
    variance_threshold: float = 0.9
    max_components: int = 32


def _validate_cuda_vector(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank 1")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def _validate_cuda_matrix(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def _validate_query_and_bank(query_vector: torch.Tensor, bank: torch.Tensor) -> None:
    _validate_cuda_vector("query_vector", query_vector)
    _validate_cuda_matrix("pooled_bank", bank)
    if query_vector.device != bank.device:
        raise ValueError("query_vector and pooled_bank must be on the same CUDA device")
    if query_vector.shape[0] != bank.shape[1]:
        raise ValueError("query_vector and pooled_bank must have the same feature dimension")
    if bank.shape[0] == 0:
        raise ValueError("pooled_bank must contain at least one row")


def _chunk_ranges(num_rows: int, chunk_size: int) -> range:
    return range(0, num_rows, chunk_size)


def select_local_references(
    query_vector: torch.Tensor,
    pooled_bank: torch.Tensor,
    *,
    k: int = 30,
    reference_chunk_size: int = DEFAULT_KNN_REFERENCE_CHUNK_SIZE,
    eps: float = DEFAULT_ACOS_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CUDA indices and vectors for the angular kNN local neighborhood.

    A custom CUDA kernel is not needed because exact query-local selection is a
    direct composition of PyTorch normalize, matmul, acos, and topk. Chunked
    top-k merging keeps the pairwise distance block bounded by GPU memory.
    """
    _validate_query_and_bank(query_vector, pooled_bank)
    if k < 1:
        raise ValueError("k must be positive")
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be positive")
    if not 0.0 <= eps < 1.0:
        raise ValueError("eps must satisfy 0 <= eps < 1")

    neighbor_count = min(int(k), pooled_bank.shape[0])
    query = query_vector.to(dtype=torch.float32).unsqueeze(0)
    best_distances: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None

    for start in _chunk_ranges(pooled_bank.shape[0], reference_chunk_size):
        stop = min(start + reference_chunk_size, pooled_bank.shape[0])
        distances = batch_angular_distance(
            query,
            pooled_bank[start:stop].to(dtype=torch.float32),
            query_chunk_size=1,
            reference_chunk_size=stop - start,
            eps=eps,
        ).squeeze(0)
        local_k = min(neighbor_count, distances.shape[0])
        local = torch.topk(distances, k=local_k, largest=False, sorted=True)
        local_indices = local.indices + start

        if best_distances is None or best_indices is None:
            best_distances = local.values
            best_indices = local_indices
        else:
            merged_distances = torch.cat((best_distances, local.values), dim=0)
            merged_indices = torch.cat((best_indices, local_indices), dim=0)
            merged = torch.topk(merged_distances, k=neighbor_count, largest=False, sorted=True)
            best_distances = merged.values
            best_indices = merged_indices[merged.indices]

    if best_indices is None:
        raise RuntimeError("failed to select local references")
    return best_indices, pooled_bank[best_indices]


def _local_pca_normal_residual(
    query_vector: torch.Tensor,
    local_references: torch.Tensor,
    *,
    variance_threshold: float,
    max_components: int,
) -> torch.Tensor:
    mean = local_references.mean(dim=0)
    centered_references = local_references - mean.unsqueeze(0)
    centered_query = query_vector - mean
    radius = torch.linalg.norm(centered_references, dim=1).mean().clamp_min(1e-8)

    if local_references.shape[0] == 1:
        return torch.linalg.norm(centered_query) / radius

    _, singular_values, vh = torch.linalg.svd(centered_references, full_matrices=False)
    variances = singular_values.square()
    variance_ratio = variances / variances.sum().clamp_min(1e-8)
    cumulative = torch.cumsum(variance_ratio, dim=0)
    threshold = torch.tensor(variance_threshold, device=query_vector.device)
    component_count = int(torch.searchsorted(cumulative, threshold).item() + 1)
    component_count = max(1, min(component_count, int(max_components), vh.shape[0]))
    components = vh[:component_count]
    coefficients = components @ centered_query
    projection = components.T @ coefficients
    residual = centered_query - projection
    return torch.linalg.norm(residual) / radius


def compute_local_features(
    query_vector: torch.Tensor,
    local_references: torch.Tensor,
    *,
    config: LocalFeatureConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute query-local geometry features on CUDA."""
    _validate_query_and_bank(query_vector, local_references)
    feature_config = config or LocalFeatureConfig()
    if not 0.0 < feature_config.variance_threshold <= 1.0:
        raise ValueError("variance_threshold must satisfy 0 < threshold <= 1")
    if feature_config.max_components < 1:
        raise ValueError("max_components must be positive")

    query = query_vector.to(dtype=torch.float32)
    references = local_references.to(dtype=torch.float32)
    angular_distances = batch_angular_distance(
        query.unsqueeze(0),
        references,
        query_chunk_size=1,
        reference_chunk_size=references.shape[0],
    ).squeeze(0)
    centroid_query = query.unsqueeze(0)
    return {
        "centroid_angular_distance": centroid_angular_distance_gpu(centroid_query, references).squeeze(0),
        "local_pca_residual": _local_pca_normal_residual(
            query,
            references,
            variance_threshold=feature_config.variance_threshold,
            max_components=feature_config.max_components,
        ),
        "mean_angular_distance": angular_distances.mean(),
        "std_angular_distance": angular_distances.std(unbiased=False),
        "centroid_euclidean_distance": centroid_euclidean_distance_gpu(centroid_query, references).squeeze(0),
    }
