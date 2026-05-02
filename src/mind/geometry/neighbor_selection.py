"""CUDA neighbor-selection methods for Phase C comparison experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F


METHOD_NAMES = (
    "knn_angular_k30",
    "kernel_knn_k30",
    "radius_ball",
    "knn_cosine_k30",
    "knn_euclidean_k30",
)

DEFAULT_K = 30
DEFAULT_REFERENCE_CHUNK_SIZE = 16_384
DEFAULT_QUERY_CHUNK_SIZE = 512
DEFAULT_ACOS_EPS = 1e-7


@dataclass(frozen=True)
class NeighborScoreResult:
    method: str
    values: torch.Tensor
    neighbor_counts: torch.Tensor | None = None
    radius: torch.Tensor | None = None


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


def _validate_method(method: str) -> None:
    if method not in METHOD_NAMES:
        raise ValueError(f"Unsupported neighbor selection method: {method}")


def _validate_positive_int(name: str, value: int) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be positive")


def _ranges(size: int, chunk_size: int) -> range:
    return range(0, size, int(chunk_size))


def _angular_distance(query: torch.Tensor, reference: torch.Tensor, *, eps: float = DEFAULT_ACOS_EPS) -> torch.Tensor:
    query_normalized = F.normalize(query, dim=1)
    reference_normalized = F.normalize(reference, dim=1)
    cosine = query_normalized @ reference_normalized.T
    return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))


def _cosine_similarity(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.normalize(query, dim=1) @ F.normalize(reference, dim=1).T


def _euclidean_distance(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    query_norm = (query * query).sum(dim=1, keepdim=True)
    reference_norm = (reference * reference).sum(dim=1).unsqueeze(0)
    squared = query_norm + reference_norm - 2.0 * (query @ reference.T)
    return squared.clamp_min(0.0).sqrt()


def _merge_topk(
    best_values: torch.Tensor | None,
    best_indices: torch.Tensor | None,
    local_values: torch.Tensor,
    local_indices: torch.Tensor,
    *,
    k: int,
    largest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if best_values is None or best_indices is None:
        return local_values, local_indices
    merged_values = torch.cat((best_values, local_values), dim=1)
    merged_indices = torch.cat((best_indices, local_indices), dim=1)
    merged = torch.topk(merged_values, k=min(int(k), merged_values.shape[1]), dim=1, largest=largest, sorted=True)
    return merged.values, merged_indices.gather(1, merged.indices)


def _topk_values_and_indices_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    metric: str,
    k: int,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_pair(query, reference)
    _validate_positive_int("k", k)
    _validate_positive_int("query_chunk_size", query_chunk_size)
    _validate_positive_int("reference_chunk_size", reference_chunk_size)
    neighbor_count = min(int(k), int(reference.shape[0]))
    largest = metric == "cosine"
    all_values: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    reference = reference.to(dtype=torch.float32)

    for query_start in _ranges(query.shape[0], query_chunk_size):
        query_stop = min(query_start + int(query_chunk_size), query.shape[0])
        query_chunk = query[query_start:query_stop].to(dtype=torch.float32)
        best_values: torch.Tensor | None = None
        best_indices: torch.Tensor | None = None
        for reference_start in _ranges(reference.shape[0], reference_chunk_size):
            reference_stop = min(reference_start + int(reference_chunk_size), reference.shape[0])
            reference_chunk = reference[reference_start:reference_stop]
            if metric == "angular":
                scores = _angular_distance(query_chunk, reference_chunk)
            elif metric == "cosine":
                scores = _cosine_similarity(query_chunk, reference_chunk)
            elif metric == "euclidean":
                scores = _euclidean_distance(query_chunk, reference_chunk)
            else:
                raise ValueError(f"Unsupported top-k metric: {metric}")
            local_k = min(neighbor_count, scores.shape[1])
            local = torch.topk(scores, k=local_k, dim=1, largest=largest, sorted=True)
            local_indices = local.indices + int(reference_start)
            best_values, best_indices = _merge_topk(
                best_values,
                best_indices,
                local.values,
                local_indices,
                k=neighbor_count,
                largest=largest,
            )
        if best_values is None or best_indices is None:
            raise RuntimeError("failed to compute top-k neighbors")
        all_values.append(best_values)
        all_indices.append(best_indices)
    return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)


def _selected_angular_distances(
    query: torch.Tensor,
    reference: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float = DEFAULT_ACOS_EPS,
) -> torch.Tensor:
    query_normalized = F.normalize(query.to(dtype=torch.float32), dim=1)
    selected = reference.to(dtype=torch.float32)[indices]
    selected_normalized = F.normalize(selected, dim=2)
    cosine = torch.bmm(selected_normalized, query_normalized.unsqueeze(2)).squeeze(2)
    return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))


def _max_angular_distance_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> torch.Tensor:
    maximum = torch.zeros((), device=query.device, dtype=torch.float32)
    for query_start in _ranges(query.shape[0], query_chunk_size):
        query_stop = min(query_start + int(query_chunk_size), query.shape[0])
        query_chunk = query[query_start:query_stop].to(dtype=torch.float32)
        for reference_start in _ranges(reference.shape[0], reference_chunk_size):
            reference_stop = min(reference_start + int(reference_chunk_size), reference.shape[0])
            distances = _angular_distance(query_chunk, reference[reference_start:reference_stop].to(dtype=torch.float32))
            maximum = torch.maximum(maximum, distances.max())
    return maximum


def _radius_counts_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    radius: torch.Tensor,
    *,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> torch.Tensor:
    counts = torch.zeros((query.shape[0],), device=query.device, dtype=torch.float32)
    radius_value = radius.to(device=query.device, dtype=torch.float32)
    for query_start in _ranges(query.shape[0], query_chunk_size):
        query_stop = min(query_start + int(query_chunk_size), query.shape[0])
        query_chunk = query[query_start:query_stop].to(dtype=torch.float32)
        chunk_counts = torch.zeros((query_chunk.shape[0],), device=query.device, dtype=torch.float32)
        for reference_start in _ranges(reference.shape[0], reference_chunk_size):
            reference_stop = min(reference_start + int(reference_chunk_size), reference.shape[0])
            distances = _angular_distance(query_chunk, reference[reference_start:reference_stop].to(dtype=torch.float32))
            chunk_counts += (distances <= radius_value).sum(dim=1).to(dtype=torch.float32)
        counts[query_start:query_stop] = chunk_counts
    return counts


def _round_radius_outward(radius: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(radius, torch.full_like(radius, float("inf")))


def tune_radius_for_target_count_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    target_count: int = DEFAULT_K,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    binary_steps: int = 32,
) -> torch.Tensor:
    """Tune a global angular radius so the mean neighbor count is near target_count."""
    _validate_pair(query, reference)
    _validate_positive_int("target_count", target_count)
    _validate_positive_int("query_chunk_size", query_chunk_size)
    _validate_positive_int("reference_chunk_size", reference_chunk_size)
    nearest, _ = _topk_values_and_indices_gpu(
        query,
        reference,
        metric="angular",
        k=1,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=reference_chunk_size,
    )
    low = nearest.max().detach()
    high = _max_angular_distance_gpu(
        query,
        reference,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=reference_chunk_size,
    ).detach()
    target = float(min(int(target_count), int(reference.shape[0])))
    if target <= 1.0:
        return _round_radius_outward(low)
    for _ in range(int(binary_steps)):
        midpoint = (low + high) / 2.0
        average_count = _radius_counts_gpu(
            query,
            reference,
            midpoint,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        ).mean()
        if float(average_count.detach().cpu()) < target:
            low = midpoint
        else:
            high = midpoint
    return _round_radius_outward(high)


def _radius_scores_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty((query.shape[0],), device=query.device, dtype=torch.float32)
    counts = torch.empty((query.shape[0],), device=query.device, dtype=torch.float32)
    radius_value = radius.to(device=query.device, dtype=torch.float32)
    for query_start in _ranges(query.shape[0], query_chunk_size):
        query_stop = min(query_start + int(query_chunk_size), query.shape[0])
        query_chunk = query[query_start:query_stop].to(dtype=torch.float32)
        sums = torch.zeros((query_chunk.shape[0],), device=query.device, dtype=torch.float32)
        chunk_counts = torch.zeros((query_chunk.shape[0],), device=query.device, dtype=torch.float32)
        for reference_start in _ranges(reference.shape[0], reference_chunk_size):
            reference_stop = min(reference_start + int(reference_chunk_size), reference.shape[0])
            distances = _angular_distance(query_chunk, reference[reference_start:reference_stop].to(dtype=torch.float32))
            mask = distances <= radius_value
            sums += torch.where(mask, distances, torch.zeros_like(distances)).sum(dim=1)
            chunk_counts += mask.sum(dim=1).to(dtype=torch.float32)
        if torch.any(chunk_counts == 0):
            raise RuntimeError("radius tuning produced an empty neighborhood")
        values[query_start:query_stop] = sums / chunk_counts
        counts[query_start:query_stop] = chunk_counts
    return values, counts


def compute_neighbor_scores_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    method: str,
    k: int = DEFAULT_K,
    target_count: int = DEFAULT_K,
    radius: torch.Tensor | float | None = None,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    kernel_eps: float = 1e-6,
) -> NeighborScoreResult:
    """Return one mean angular drift score per CUDA query row for a method."""
    _validate_method(method)
    _validate_pair(query, reference)
    _validate_positive_int("query_chunk_size", query_chunk_size)
    _validate_positive_int("reference_chunk_size", reference_chunk_size)

    if method in {"knn_angular_k30", "kernel_knn_k30"}:
        distances, _indices = _topk_values_and_indices_gpu(
            query,
            reference,
            metric="angular",
            k=k,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        )
        if method == "kernel_knn_k30":
            weights = 1.0 / distances.clamp_min(float(kernel_eps))
            values = (distances * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(float(kernel_eps))
        else:
            values = distances.mean(dim=1)
        return NeighborScoreResult(method=method, values=values)

    if method == "knn_cosine_k30":
        _similarities, indices = _topk_values_and_indices_gpu(
            query,
            reference,
            metric="cosine",
            k=k,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        )
        values = _selected_angular_distances(query, reference, indices).mean(dim=1)
        return NeighborScoreResult(method=method, values=values)

    if method == "knn_euclidean_k30":
        _distances, indices = _topk_values_and_indices_gpu(
            query,
            reference,
            metric="euclidean",
            k=k,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        )
        values = _selected_angular_distances(query, reference, indices).mean(dim=1)
        return NeighborScoreResult(method=method, values=values)

    if method == "radius_ball":
        if radius is None:
            radius_tensor = tune_radius_for_target_count_gpu(
                query,
                reference,
                target_count=target_count,
                query_chunk_size=query_chunk_size,
                reference_chunk_size=reference_chunk_size,
            )
        else:
            radius_tensor = torch.as_tensor(radius, device=query.device, dtype=torch.float32)
        values, counts = _radius_scores_gpu(
            query,
            reference,
            radius=radius_tensor,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        )
        return NeighborScoreResult(method=method, values=values, neighbor_counts=counts, radius=radius_tensor)

    raise ValueError(f"Unsupported neighbor selection method: {method}")


def _slope(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    x = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    return (x_centered * y_centered).sum() / x_centered.square().sum().clamp_min(1e-8)


def compute_neighbor_feature_row_gpu(
    *,
    layer_vectors: torch.Tensor,
    selected_layers: list[int] | tuple[int, ...],
    reference_layers: dict[int, torch.Tensor],
    method: str,
    k: int = DEFAULT_K,
    target_count: int = DEFAULT_K,
    layer_radii: Mapping[int, torch.Tensor | float] | None = None,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
) -> dict[str, float]:
    """Build raw drift and calibrated summary features for one cache entry."""
    _validate_method(method)
    _validate_cuda_matrix("layer_vectors", layer_vectors)
    if len(selected_layers) != int(layer_vectors.shape[0]):
        raise ValueError("selected_layers must align with layer_vectors rows")
    curve_values: list[torch.Tensor] = []
    for offset, layer_index in enumerate(selected_layers):
        layer = int(layer_index)
        if layer not in reference_layers:
            raise KeyError(f"Missing reference layer: {layer}")
        query = layer_vectors[offset : offset + 1].to(dtype=torch.float32)
        reference = reference_layers[layer].to(device=layer_vectors.device, dtype=torch.float32)
        radius = None if layer_radii is None else layer_radii[layer]
        result = compute_neighbor_scores_gpu(
            query,
            reference,
            method=method,
            k=k,
            target_count=target_count,
            radius=radius,
            reference_chunk_size=reference_chunk_size,
        )
        curve_values.append(result.values.squeeze(0))
    curve = torch.stack(curve_values).to(dtype=torch.float32)
    row = {f"raw_drift_{index}": float(value.detach().cpu()) for index, value in enumerate(curve)}
    row.update(
        {
            "cal_mean_drift": float(curve.mean().detach().cpu()),
            "cal_max_drift": float(curve.max().detach().cpu()),
            "cal_final_drift": float(curve[-1].detach().cpu()),
            "cal_drift_slope": float(_slope(curve).detach().cpu()),
            "cal_drift_variance": float(curve.var(unbiased=False).detach().cpu()),
        }
    )
    return row
