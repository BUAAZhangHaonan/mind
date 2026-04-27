"""CUDA distance primitives for geometry-based scoring."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DEFAULT_MAX_DISTANCE_ELEMENTS = 16_777_216
DEFAULT_KNN_QUERY_CHUNK_SIZE = 1024
DEFAULT_KNN_REFERENCE_CHUNK_SIZE = 16_384
DEFAULT_ACOS_EPS = 1e-7


def _validate_cuda_matrix(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def _validate_pair(query: torch.Tensor, reference: torch.Tensor) -> None:
    _validate_cuda_matrix("query", query)
    _validate_cuda_matrix("reference", reference)
    if query.device != reference.device:
        raise ValueError("query and reference must be on the same CUDA device")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same feature dimension")
    if query.shape[0] == 0:
        raise ValueError("query must contain at least one row")
    if reference.shape[0] == 0:
        raise ValueError("reference must contain at least one row")


def _validate_chunk_size(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive")


def _resolve_distance_chunks(
    num_query: int,
    num_reference: int,
    query_chunk_size: int | None,
    reference_chunk_size: int | None,
    max_elements: int,
) -> tuple[int, int]:
    _validate_chunk_size("query_chunk_size", query_chunk_size)
    _validate_chunk_size("reference_chunk_size", reference_chunk_size)
    if max_elements < 1:
        raise ValueError("max_elements must be positive")

    if query_chunk_size is None and reference_chunk_size is None:
        query_chunk_size = min(num_query, int(math.sqrt(max_elements)) or 1)
        reference_chunk_size = min(num_reference, max(1, max_elements // query_chunk_size))
    elif query_chunk_size is None:
        reference_chunk_size = min(num_reference, int(reference_chunk_size))
        query_chunk_size = min(num_query, max(1, max_elements // reference_chunk_size))
    elif reference_chunk_size is None:
        query_chunk_size = min(num_query, int(query_chunk_size))
        reference_chunk_size = min(num_reference, max(1, max_elements // query_chunk_size))
    else:
        query_chunk_size = min(num_query, int(query_chunk_size))
        reference_chunk_size = min(num_reference, int(reference_chunk_size))

    return query_chunk_size, reference_chunk_size


def _row_ranges(num_rows: int, chunk_size: int) -> range:
    return range(0, num_rows, chunk_size)


def batch_angular_distance(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_chunk_size: int | None = None,
    reference_chunk_size: int | None = None,
    max_elements: int = DEFAULT_MAX_DISTANCE_ELEMENTS,
    eps: float = DEFAULT_ACOS_EPS,
) -> torch.Tensor:
    """Return pairwise angular distances between CUDA query and reference rows."""
    _validate_pair(query, reference)
    if not 0.0 <= eps < 1.0:
        raise ValueError("eps must satisfy 0 <= eps < 1")
    query_chunk_size, reference_chunk_size = _resolve_distance_chunks(
        query.shape[0],
        reference.shape[0],
        query_chunk_size,
        reference_chunk_size,
        max_elements,
    )

    distances = torch.empty((query.shape[0], reference.shape[0]), dtype=query.dtype, device=query.device)
    reference_normalized = F.normalize(reference, dim=1)
    for query_start in _row_ranges(query.shape[0], query_chunk_size):
        query_end = min(query.shape[0], query_start + query_chunk_size)
        query_normalized = F.normalize(query[query_start:query_end], dim=1)
        for reference_start in _row_ranges(reference.shape[0], reference_chunk_size):
            reference_end = min(reference.shape[0], reference_start + reference_chunk_size)
            cosine = query_normalized @ reference_normalized[reference_start:reference_end].T
            distances[query_start:query_end, reference_start:reference_end] = torch.acos(
                cosine.clamp(-1.0 + eps, 1.0 - eps)
            )
    return distances


def batch_euclidean_distance(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_chunk_size: int | None = None,
    reference_chunk_size: int | None = None,
    max_elements: int = DEFAULT_MAX_DISTANCE_ELEMENTS,
) -> torch.Tensor:
    """Return pairwise Euclidean distances between CUDA query and reference rows."""
    _validate_pair(query, reference)
    query_chunk_size, reference_chunk_size = _resolve_distance_chunks(
        query.shape[0],
        reference.shape[0],
        query_chunk_size,
        reference_chunk_size,
        max_elements,
    )

    distances = torch.empty((query.shape[0], reference.shape[0]), dtype=query.dtype, device=query.device)
    reference_norm = (reference * reference).sum(dim=1)
    for query_start in _row_ranges(query.shape[0], query_chunk_size):
        query_end = min(query.shape[0], query_start + query_chunk_size)
        query_chunk = query[query_start:query_end]
        query_norm = (query_chunk * query_chunk).sum(dim=1, keepdim=True)
        for reference_start in _row_ranges(reference.shape[0], reference_chunk_size):
            reference_end = min(reference.shape[0], reference_start + reference_chunk_size)
            squared = (
                query_norm
                + reference_norm[reference_start:reference_end].unsqueeze(0)
                - 2.0 * (query_chunk @ reference[reference_start:reference_end].T)
            )
            distances[query_start:query_end, reference_start:reference_end] = squared.clamp_min(0.0).sqrt()
    return distances


def knn_angular_distance_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    *,
    query_chunk_size: int | None = DEFAULT_KNN_QUERY_CHUNK_SIZE,
    reference_chunk_size: int | None = DEFAULT_KNN_REFERENCE_CHUNK_SIZE,
) -> torch.Tensor:
    """Return the exact mean angular distance to each query row's k nearest references."""
    _validate_pair(query, reference)
    _validate_chunk_size("query_chunk_size", query_chunk_size)
    _validate_chunk_size("reference_chunk_size", reference_chunk_size)
    if k < 1:
        raise ValueError("k must be positive")
    if k > reference.shape[0]:
        raise ValueError("k must be less than or equal to the number of reference rows")

    query_chunk_size, reference_chunk_size = _resolve_distance_chunks(
        query.shape[0],
        reference.shape[0],
        query_chunk_size,
        reference_chunk_size,
        DEFAULT_MAX_DISTANCE_ELEMENTS,
    )
    means = torch.empty((query.shape[0],), dtype=query.dtype, device=query.device)

    # A custom CUDA kernel is not needed here because the exact algorithm uses
    # standard PyTorch matmul/topk/acos operations with chunked top-k merging.
    for query_start in _row_ranges(query.shape[0], query_chunk_size):
        query_end = min(query.shape[0], query_start + query_chunk_size)
        query_chunk = query[query_start:query_end]
        best_distances: torch.Tensor | None = None

        for reference_start in _row_ranges(reference.shape[0], reference_chunk_size):
            reference_end = min(reference.shape[0], reference_start + reference_chunk_size)
            distances = batch_angular_distance(
                query_chunk,
                reference[reference_start:reference_end],
                query_chunk_size=query_chunk.shape[0],
                reference_chunk_size=reference_end - reference_start,
            )
            local_k = min(k, distances.shape[1])
            local_best = torch.topk(distances, k=local_k, dim=1, largest=False).values
            if best_distances is None:
                best_distances = local_best
            else:
                merged = torch.cat((best_distances, local_best), dim=1)
                merged_k = min(k, merged.shape[1])
                best_distances = torch.topk(merged, k=merged_k, dim=1, largest=False).values

        if best_distances is None or best_distances.shape[1] != k:
            raise RuntimeError("failed to compute complete top-k distances")
        means[query_start:query_end] = best_distances.mean(dim=1)

    return means


def centroid_angular_distance_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return angular distance from each CUDA query row to the CUDA reference centroid."""
    _validate_pair(query, reference)
    centroid = reference.mean(dim=0, keepdim=True)
    return batch_angular_distance(
        query,
        centroid,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=1,
    ).squeeze(1)


def centroid_euclidean_distance_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return Euclidean distance from each CUDA query row to the CUDA reference centroid."""
    _validate_pair(query, reference)
    centroid = reference.mean(dim=0, keepdim=True)
    return batch_euclidean_distance(
        query,
        centroid,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=1,
    ).squeeze(1)
