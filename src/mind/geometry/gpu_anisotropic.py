"""CUDA local anisotropic scoring for radius-ball neighborhoods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mind.geometry.gpu_distances import batch_angular_distance
from mind.geometry.neighbor_selection import DEFAULT_RADIUS_MARGIN


DEFAULT_EPS = 1e-6
DEFAULT_RANK_CAP = 8
DEFAULT_REFERENCE_CHUNK_SIZE = 16_384

ANISOTROPIC_VARIANT_NAMES = ("diag_maha", "lowrank_maha", "full_maha_shrink")
VARIANT_NAMES = ("radius_ball_isotropic", *ANISOTROPIC_VARIANT_NAMES)


@dataclass(frozen=True)
class AnisotropicScoreResult:
    variant: str
    values: torch.Tensor
    alpha: torch.Tensor | None = None
    neighbor_counts: torch.Tensor | None = None


def _validate_cuda_matrix(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def _validate_cuda_neighbors(query: torch.Tensor, neighbors: torch.Tensor, *, min_neighbors: int = 2) -> None:
    _validate_cuda_matrix("query", query)
    if not isinstance(neighbors, torch.Tensor):
        raise TypeError("neighbors must be a torch.Tensor")
    if neighbors.device.type != "cuda":
        raise ValueError("neighbors must be a CUDA tensor")
    if neighbors.ndim != 3:
        raise ValueError("neighbors must be a rank-3 tensor")
    if not neighbors.is_floating_point():
        raise TypeError("neighbors must be a floating point tensor")
    if query.device != neighbors.device:
        raise ValueError("query and neighbors must be on the same CUDA device")
    if query.shape[0] != neighbors.shape[0]:
        raise ValueError("query and neighbors must have the same batch size")
    if query.shape[1] != neighbors.shape[2]:
        raise ValueError("query and neighbors must have the same feature dimension")
    if query.shape[0] == 0:
        raise ValueError("query must contain at least one row")
    if neighbors.shape[1] < int(min_neighbors):
        expected = "at least two rows" if int(min_neighbors) == 2 else f"at least {int(min_neighbors)} rows"
        raise ValueError(
            f"neighbors must contain {expected} per query; got {neighbors.shape[1]}"
        )


def _validate_variant(variant: str) -> None:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unsupported anisotropic variant: {variant}")


def _as_unit_float32(tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
    return F.normalize(tensor.to(dtype=torch.float32), dim=dim)


def _radius_values(radius: torch.Tensor | float, *, batch_size: int, device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(radius, device=device, dtype=torch.float32)
    if values.ndim == 0:
        return values.expand(batch_size)
    if values.shape != (batch_size,):
        raise ValueError("radius must be scalar or have shape (batch_size,)")
    return values


def _positive_eigen_tolerance(raw_eigenvalues: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    scale = raw_eigenvalues.abs().max().clamp_min(1.0)
    return torch.finfo(torch.float32).eps * float(max(rows, cols)) * scale


def _local_gram_eigh(centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive Gram eigenvalues, covariance eigenvalues, and eigenvectors."""
    num_neighbors, dimension = centered.shape
    gram = centered @ centered.T
    raw_eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(raw_eigenvalues, descending=True)
    raw_eigenvalues = raw_eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = raw_eigenvalues > _positive_eigen_tolerance(
        raw_eigenvalues,
        rows=int(num_neighbors),
        cols=int(dimension),
    )
    raw_eigenvalues = raw_eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    covariance_eigenvalues = raw_eigenvalues / float(num_neighbors)
    return raw_eigenvalues, covariance_eigenvalues, eigenvectors


def _projections_from_gram(
    centered: torch.Tensor,
    delta: torch.Tensor,
    raw_eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    rank: int,
    eps: float,
) -> torch.Tensor:
    if rank == 0:
        return torch.empty((0,), device=centered.device, dtype=torch.float32)
    scaled = eigenvectors[:, :rank] / raw_eigenvalues[:rank].clamp_min(float(eps)).sqrt().unsqueeze(0)
    return scaled.T @ (centered @ delta)


def _analytic_ledoit_wolf_alpha(centered: torch.Tensor, *, eps: float = DEFAULT_EPS) -> torch.Tensor:
    num_neighbors, dimension = centered.shape
    gram = centered @ centered.T
    trace_covariance = torch.trace(gram) / float(num_neighbors)
    tau = trace_covariance / float(dimension)
    fro_covariance_squared = gram.square().sum() / float(num_neighbors * num_neighbors)
    target_distance_squared = (fro_covariance_squared - float(dimension) * tau.square()).clamp_min(0.0)
    if bool(target_distance_squared <= float(eps)):
        return torch.ones((), device=centered.device, dtype=torch.float32)
    row_norms_squared = centered.square().sum(dim=1)
    numerator = row_norms_squared.square().sum() / float(num_neighbors * num_neighbors)
    numerator = numerator - fro_covariance_squared / float(num_neighbors)
    return (numerator / target_distance_squared).clamp(0.0, 1.0)


def _resolve_alpha(
    shrinkage_alpha: float | torch.Tensor | None,
    *,
    row_index: int,
    centered: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if shrinkage_alpha is None:
        return _analytic_ledoit_wolf_alpha(centered, eps=eps)
    alpha = torch.as_tensor(shrinkage_alpha, device=centered.device, dtype=torch.float32)
    if alpha.ndim == 0:
        resolved = alpha
    elif alpha.ndim == 1:
        resolved = alpha[int(row_index)]
    else:
        raise ValueError("shrinkage_alpha must be scalar or rank-1")
    if bool((resolved < 0.0) | (resolved > 1.0)):
        raise ValueError("shrinkage_alpha must be in [0, 1]")
    return resolved


def diag_maha_gpu(
    query: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> AnisotropicScoreResult:
    """Score query rows with diagonal local Mahalanobis distance on unit vectors."""
    _validate_cuda_neighbors(query, neighbors, min_neighbors=2)
    query_unit = _as_unit_float32(query, dim=1)
    neighbors_unit = _as_unit_float32(neighbors, dim=2)
    mean = neighbors_unit.mean(dim=1)
    centered = neighbors_unit - mean.unsqueeze(1)
    variance = centered.square().mean(dim=1)
    values = (((query_unit - mean).square() / (variance + float(eps))).sum(dim=1)).clamp_min(0.0).sqrt()
    return AnisotropicScoreResult(variant="diag_maha", values=values)


def lowrank_maha_gpu(
    query: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    rank_cap: int = DEFAULT_RANK_CAP,
    eps: float = DEFAULT_EPS,
) -> AnisotropicScoreResult:
    """Score query rows with Gram low-rank Mahalanobis plus residual distance."""
    _validate_cuda_neighbors(query, neighbors, min_neighbors=2)
    if int(rank_cap) < 0:
        raise ValueError("rank_cap must be non-negative")
    query_unit = _as_unit_float32(query, dim=1)
    neighbors_unit = _as_unit_float32(neighbors, dim=2)
    batch_size, num_neighbors, dimension = neighbors_unit.shape
    values = torch.empty((batch_size,), device=query.device, dtype=torch.float32)

    for row_index in range(batch_size):
        local_neighbors = neighbors_unit[row_index]
        mean = local_neighbors.mean(dim=0)
        centered = local_neighbors - mean
        delta = query_unit[row_index] - mean
        raw_eigenvalues, covariance_eigenvalues, eigenvectors = _local_gram_eigh(centered)
        rank = min(
            int(rank_cap),
            int(num_neighbors) - 1,
            int(dimension),
            int(covariance_eigenvalues.numel()),
        )
        projections = _projections_from_gram(
            centered,
            delta,
            raw_eigenvalues,
            eigenvectors,
            rank=rank,
            eps=eps,
        )
        if rank > 0:
            eigenvalues = covariance_eigenvalues[:rank]
            in_subspace = (projections.square() / (eigenvalues + float(eps))).sum()
            projected_norm_squared = projections.square().sum()
            used_trace = eigenvalues.sum()
        else:
            in_subspace = torch.zeros((), device=query.device, dtype=torch.float32)
            projected_norm_squared = torch.zeros((), device=query.device, dtype=torch.float32)
            used_trace = torch.zeros((), device=query.device, dtype=torch.float32)
        residual_norm_squared = (delta.square().sum() - projected_norm_squared).clamp_min(0.0)
        total_trace = centered.square().sum() / float(num_neighbors)
        residual_variance = (total_trace - used_trace).clamp_min(0.0) / float(max(int(dimension) - rank, 1))
        residual_variance = residual_variance.clamp_min(float(eps))
        values[row_index] = (in_subspace + residual_norm_squared / residual_variance).clamp_min(0.0).sqrt()

    return AnisotropicScoreResult(variant="lowrank_maha", values=values)


def full_maha_shrink_gpu(
    query: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    shrinkage_alpha: float | torch.Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> AnisotropicScoreResult:
    """Score query rows with exact low-rank shrinkage Mahalanobis distance."""
    _validate_cuda_neighbors(query, neighbors, min_neighbors=2)
    query_unit = _as_unit_float32(query, dim=1)
    neighbors_unit = _as_unit_float32(neighbors, dim=2)
    batch_size, num_neighbors, dimension = neighbors_unit.shape
    values = torch.empty((batch_size,), device=query.device, dtype=torch.float32)
    alphas = torch.empty((batch_size,), device=query.device, dtype=torch.float32)

    for row_index in range(batch_size):
        local_neighbors = neighbors_unit[row_index]
        mean = local_neighbors.mean(dim=0)
        centered = local_neighbors - mean
        delta = query_unit[row_index] - mean
        raw_eigenvalues, covariance_eigenvalues, eigenvectors = _local_gram_eigh(centered)
        rank = int(covariance_eigenvalues.numel())
        projections = _projections_from_gram(
            centered,
            delta,
            raw_eigenvalues,
            eigenvectors,
            rank=rank,
            eps=eps,
        )
        projected_norm_squared = projections.square().sum() if rank > 0 else torch.zeros((), device=query.device)
        delta_perp_squared = (delta.square().sum() - projected_norm_squared).clamp_min(0.0)
        trace_covariance = centered.square().sum() / float(num_neighbors)
        tau = trace_covariance / float(dimension)
        alpha = _resolve_alpha(
            shrinkage_alpha,
            row_index=row_index,
            centered=centered,
            eps=eps,
        )
        beta = alpha * tau
        eta = 1.0 - alpha
        if rank > 0:
            in_subspace = (projections.square() / (beta + eta * covariance_eigenvalues + float(eps))).sum()
        else:
            in_subspace = torch.zeros((), device=query.device, dtype=torch.float32)
        score_squared = in_subspace + delta_perp_squared / (beta + float(eps))
        values[row_index] = score_squared.clamp_min(0.0).sqrt()
        alphas[row_index] = alpha

    return AnisotropicScoreResult(variant="full_maha_shrink", values=values, alpha=alphas)


def _mean_angular_to_neighbors_gpu(query: torch.Tensor, neighbors: torch.Tensor) -> AnisotropicScoreResult:
    _validate_cuda_neighbors(query, neighbors, min_neighbors=1)
    query_unit = _as_unit_float32(query, dim=1)
    neighbors_unit = _as_unit_float32(neighbors, dim=2)
    cosine = torch.bmm(neighbors_unit, query_unit.unsqueeze(2)).squeeze(2)
    distances = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    return AnisotropicScoreResult(
        variant="radius_ball_isotropic",
        values=distances.mean(dim=1),
        neighbor_counts=torch.full((query.shape[0],), float(neighbors.shape[1]), device=query.device),
    )


def compute_anisotropic_scores_gpu(
    query: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    variant: str,
    eps: float = DEFAULT_EPS,
    rank_cap: int = DEFAULT_RANK_CAP,
    shrinkage_alpha: float | torch.Tensor | None = None,
) -> AnisotropicScoreResult:
    """Dispatch local scoring for already selected radius-ball neighborhoods."""
    _validate_variant(variant)
    if variant == "radius_ball_isotropic":
        return _mean_angular_to_neighbors_gpu(query, neighbors)
    if variant == "diag_maha":
        return diag_maha_gpu(query, neighbors, eps=eps)
    if variant == "lowrank_maha":
        return lowrank_maha_gpu(query, neighbors, rank_cap=rank_cap, eps=eps)
    return full_maha_shrink_gpu(query, neighbors, shrinkage_alpha=shrinkage_alpha, eps=eps)


def _select_radius_ball_neighbors_and_distances_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    min_neighbors: int = 1,
    reference_chunk_size: int,
    radius_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_cuda_matrix("query", query)
    _validate_cuda_matrix("reference", reference)
    if query.shape[0] != 1:
        raise ValueError("radius-ball neighbor selection expects one query row")
    if query.device != reference.device:
        raise ValueError("query and reference must be on the same CUDA device")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same feature dimension")
    if int(reference_chunk_size) < 1:
        raise ValueError("reference_chunk_size must be positive")
    if int(min_neighbors) < 1:
        raise ValueError("min_neighbors must be positive")
    if int(min_neighbors) > int(reference.shape[0]):
        raise ValueError("min_neighbors cannot exceed the number of reference rows")

    radius_tensor = torch.as_tensor(radius, device=query.device, dtype=torch.float32)
    selected_indices: list[torch.Tensor] = []
    selected_distances: list[torch.Tensor] = []
    best_distances: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    query_f = query.to(dtype=torch.float32)
    reference_f = reference.to(dtype=torch.float32)
    for start in range(0, reference_f.shape[0], int(reference_chunk_size)):
        stop = min(start + int(reference_chunk_size), reference_f.shape[0])
        reference_chunk = reference_f[start:stop]
        distances = batch_angular_distance(
            query_f,
            reference_chunk,
            query_chunk_size=1,
            reference_chunk_size=reference_chunk.shape[0],
        ).squeeze(0)
        mask = distances <= radius_tensor + float(radius_margin)
        if torch.any(mask):
            selected_indices.append(torch.arange(start, stop, device=query.device, dtype=torch.long)[mask])
            selected_distances.append(distances[mask])
        local_k = min(int(min_neighbors), int(distances.numel()))
        local_distances, local_indices = torch.topk(distances, k=local_k, largest=False, sorted=True)
        local_indices = local_indices.to(dtype=torch.long) + int(start)
        if best_distances is None or best_indices is None:
            best_distances = local_distances
            best_indices = local_indices
        else:
            merged_distances = torch.cat((best_distances, local_distances), dim=0)
            merged_indices = torch.cat((best_indices, local_indices), dim=0)
            keep_distances, keep_order = torch.topk(
                merged_distances,
                k=min(int(min_neighbors), int(merged_distances.numel())),
                largest=False,
                sorted=True,
            )
            best_distances = keep_distances
            best_indices = merged_indices[keep_order]

    if selected_indices:
        selected_index_tensor = torch.cat(selected_indices, dim=0)
        selected_distance_tensor = torch.cat(selected_distances, dim=0)
    else:
        selected_index_tensor = torch.empty((0,), device=query.device, dtype=torch.long)
        selected_distance_tensor = torch.empty((0,), device=query.device, dtype=torch.float32)

    if selected_index_tensor.numel() >= int(min_neighbors):
        return reference_f[selected_index_tensor], selected_distance_tensor
    if best_distances is None or best_indices is None:
        raise RuntimeError("failed to compute support-floor nearest neighbors")

    combined_indices = torch.cat((selected_index_tensor, best_indices), dim=0)
    combined_distances = torch.cat((selected_distance_tensor, best_distances), dim=0)
    unique_indices, inverse = torch.unique(combined_indices, sorted=False, return_inverse=True)
    unique_distances = torch.full(
        (unique_indices.numel(),),
        float("inf"),
        device=query.device,
        dtype=torch.float32,
    )
    unique_distances.scatter_reduce_(0, inverse, combined_distances, reduce="amin", include_self=True)
    order = torch.argsort(unique_distances)
    unique_indices = unique_indices[order]
    unique_distances = unique_distances[order]
    if unique_indices.numel() < int(min_neighbors):
        raise RuntimeError("support-floor selection produced too few neighbors")
    return reference_f[unique_indices], unique_distances


def _select_radius_ball_neighbors_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    min_neighbors: int = 1,
    reference_chunk_size: int,
    radius_margin: float,
) -> torch.Tensor:
    neighbors, _distances = _select_radius_ball_neighbors_and_distances_gpu(
        query,
        reference,
        radius=radius,
        min_neighbors=min_neighbors,
        reference_chunk_size=reference_chunk_size,
        radius_margin=radius_margin,
    )
    return neighbors


def _variant_sequence(variants: Sequence[str]) -> tuple[str, ...]:
    variant_names = tuple(variants)
    if not variant_names:
        raise ValueError("variants must contain at least one variant")
    for variant in variant_names:
        _validate_variant(variant)
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("variants must not contain duplicates")
    return variant_names


def _row_shrinkage_alpha(
    shrinkage_alpha: float | torch.Tensor | None,
    *,
    row_index: int,
) -> float | torch.Tensor | None:
    if isinstance(shrinkage_alpha, torch.Tensor) and shrinkage_alpha.ndim == 1:
        return shrinkage_alpha[int(row_index)]
    return shrinkage_alpha


def compute_multi_variant_scores_for_radius_ball_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    variants: Sequence[str],
    eps: float = DEFAULT_EPS,
    rank_cap: int = DEFAULT_RANK_CAP,
    shrinkage_alpha: float | torch.Tensor | None = None,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    radius_margin: float = DEFAULT_RADIUS_MARGIN,
    min_neighbors: int = 2,
) -> dict[str, AnisotropicScoreResult]:
    """Select radius-ball neighbors once per query row and score multiple variants."""
    variant_names = _variant_sequence(variants)
    _validate_cuda_matrix("query", query)
    _validate_cuda_matrix("reference", reference)
    if query.device != reference.device:
        raise ValueError("query and reference must be on the same CUDA device")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same feature dimension")
    if int(reference_chunk_size) < 1:
        raise ValueError("reference_chunk_size must be positive")
    if int(min_neighbors) < 1:
        raise ValueError("min_neighbors must be positive")
    if int(min_neighbors) > int(reference.shape[0]):
        raise ValueError("min_neighbors cannot exceed the number of reference rows")

    query_f = query.to(dtype=torch.float32)
    reference_f = reference.to(dtype=torch.float32)
    radii = _radius_values(radius, batch_size=query.shape[0], device=query.device)
    needs_covariance = any(variant in ANISOTROPIC_VARIANT_NAMES for variant in variant_names)
    if needs_covariance and int(min_neighbors) < 2:
        raise ValueError("anisotropic variants require min_neighbors >= 2")

    values: dict[str, list[torch.Tensor]] = {variant: [] for variant in variant_names}
    alphas: dict[str, list[torch.Tensor]] = {variant: [] for variant in variant_names}
    counts: list[torch.Tensor] = []

    for row_index in range(query_f.shape[0]):
        query_row = query_f[row_index : row_index + 1]
        neighbors, distances = _select_radius_ball_neighbors_and_distances_gpu(
            query_row,
            reference_f,
            radius=radii[row_index],
            min_neighbors=min_neighbors,
            reference_chunk_size=reference_chunk_size,
            radius_margin=radius_margin,
        )
        if needs_covariance and neighbors.shape[0] < 2:
            raise RuntimeError(
                f"radius-ball anisotropic neighborhood has fewer than two neighbors: {neighbors.shape[0]}"
            )
        counts.append(torch.tensor(float(neighbors.shape[0]), device=query.device, dtype=torch.float32))

        for variant in variant_names:
            if variant == "radius_ball_isotropic":
                values[variant].append(distances.mean())
                continue
            result = compute_anisotropic_scores_gpu(
                query_row,
                neighbors.unsqueeze(0),
                variant=variant,
                eps=eps,
                rank_cap=rank_cap,
                shrinkage_alpha=_row_shrinkage_alpha(shrinkage_alpha, row_index=row_index),
            )
            values[variant].append(result.values.squeeze(0))
            if result.alpha is not None:
                alphas[variant].append(result.alpha.squeeze(0))

    neighbor_counts = torch.stack(counts)
    return {
        variant: AnisotropicScoreResult(
            variant=variant,
            values=torch.stack(values[variant]),
            alpha=torch.stack(alphas[variant]) if alphas[variant] else None,
            neighbor_counts=neighbor_counts,
        )
        for variant in variant_names
    }


def radius_ball_isotropic_scores_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    radius_margin: float = DEFAULT_RADIUS_MARGIN,
    min_neighbors: int = 2,
) -> AnisotropicScoreResult:
    """Score each query by mean angular distance to radius-ball neighbors."""
    return compute_multi_variant_scores_for_radius_ball_gpu(
        query,
        reference,
        radius=radius,
        variants=("radius_ball_isotropic",),
        reference_chunk_size=reference_chunk_size,
        radius_margin=radius_margin,
        min_neighbors=min_neighbors,
    )["radius_ball_isotropic"]


def compute_anisotropic_scores_for_radius_ball_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    variant: str,
    eps: float = DEFAULT_EPS,
    rank_cap: int = DEFAULT_RANK_CAP,
    shrinkage_alpha: float | torch.Tensor | None = None,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    radius_margin: float = DEFAULT_RADIUS_MARGIN,
    min_neighbors: int = 2,
) -> AnisotropicScoreResult:
    """Select radius-ball neighborhoods and score each query row."""
    return compute_multi_variant_scores_for_radius_ball_gpu(
        query,
        reference,
        radius=radius,
        variants=(variant,),
        eps=eps,
        rank_cap=rank_cap,
        shrinkage_alpha=shrinkage_alpha,
        reference_chunk_size=reference_chunk_size,
        radius_margin=radius_margin,
        min_neighbors=min_neighbors,
    )[variant]


def _slope(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    x = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    return (x_centered * y_centered).sum() / x_centered.square().sum().clamp_min(1e-8)


def compute_anisotropic_feature_row_gpu(
    *,
    layer_vectors: torch.Tensor,
    selected_layers: list[int] | tuple[int, ...],
    reference_layers: Mapping[int, torch.Tensor],
    layer_radii: Mapping[int, torch.Tensor | float],
    variant: str,
    eps: float = DEFAULT_EPS,
    rank_cap: int = DEFAULT_RANK_CAP,
    shrinkage_alpha: float | torch.Tensor | None = None,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    radius_margin: float = DEFAULT_RADIUS_MARGIN,
    expected_curve_length: int | None = None,
    min_neighbors: int = 2,
) -> dict[str, float]:
    """Build the raw-plus-full-curve feature row for one cache entry."""
    _validate_variant(variant)
    _validate_cuda_matrix("layer_vectors", layer_vectors)
    if len(selected_layers) != int(layer_vectors.shape[0]):
        raise ValueError("selected_layers must align with layer_vectors rows")
    if expected_curve_length is not None and len(selected_layers) != int(expected_curve_length):
        raise ValueError(f"raw_plus_full_curve expected {expected_curve_length} raw values")

    curve_values: list[torch.Tensor] = []
    for offset, layer_index in enumerate(selected_layers):
        layer = int(layer_index)
        if layer not in reference_layers:
            raise KeyError(f"Missing reference layer: {layer}")
        if layer not in layer_radii:
            raise KeyError(f"Missing radius for layer: {layer}")
        reference = reference_layers[layer]
        _validate_cuda_matrix(f"reference_layers[{layer}]", reference)
        query = layer_vectors[offset : offset + 1].to(dtype=torch.float32)
        result = compute_anisotropic_scores_for_radius_ball_gpu(
            query,
            reference.to(dtype=torch.float32),
            radius=layer_radii[layer],
            variant=variant,
            eps=eps,
            rank_cap=rank_cap,
            shrinkage_alpha=shrinkage_alpha,
            reference_chunk_size=reference_chunk_size,
            radius_margin=radius_margin,
            min_neighbors=min_neighbors,
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


__all__ = [
    "ANISOTROPIC_VARIANT_NAMES",
    "DEFAULT_EPS",
    "DEFAULT_RANK_CAP",
    "DEFAULT_REFERENCE_CHUNK_SIZE",
    "VARIANT_NAMES",
    "AnisotropicScoreResult",
    "compute_anisotropic_feature_row_gpu",
    "compute_multi_variant_scores_for_radius_ball_gpu",
    "compute_anisotropic_scores_for_radius_ball_gpu",
    "compute_anisotropic_scores_gpu",
    "diag_maha_gpu",
    "full_maha_shrink_gpu",
    "lowrank_maha_gpu",
    "radius_ball_isotropic_scores_gpu",
]
