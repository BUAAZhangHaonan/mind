"""Local PCA manifolds for MIND."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class LocalPCAManifold:
    mean: torch.Tensor
    components: torch.Tensor
    neighborhood_radius: torch.Tensor


def _topk_neighbor_indices(
    reference_vectors: torch.Tensor,
    query_vector: torch.Tensor,
    *,
    k_neighbors: int,
) -> torch.Tensor:
    distances = torch.norm(reference_vectors - query_vector.unsqueeze(0), dim=1)
    return torch.topk(distances, k=min(k_neighbors, reference_vectors.shape[0]), largest=False).indices


def fit_local_pca_manifold(
    reference_vectors: torch.Tensor,
    query_vector: torch.Tensor,
    *,
    k_neighbors: int = 32,
    variance_threshold: float = 0.9,
    max_components: int = 32,
) -> LocalPCAManifold:
    if reference_vectors.ndim != 2:
        raise ValueError("reference_vectors must be rank 2")
    if query_vector.ndim != 1:
        raise ValueError("query_vector must be rank 1")

    reference_vectors = reference_vectors.to(dtype=torch.float32)
    query_vector = query_vector.to(dtype=torch.float32)

    neighbor_indices = _topk_neighbor_indices(
        reference_vectors,
        query_vector,
        k_neighbors=k_neighbors,
    )
    neighbors = reference_vectors[neighbor_indices]
    mean = neighbors.mean(dim=0)
    centered = neighbors - mean

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    variances = singular_values.square()
    variance_ratio = variances / variances.sum().clamp_min(1e-8)
    cumulative = torch.cumsum(variance_ratio, dim=0)
    component_count = int(torch.searchsorted(cumulative, torch.tensor(variance_threshold)).item() + 1)
    component_count = max(1, min(component_count, max_components, vh.shape[0]))

    components = vh[:component_count]
    neighborhood_radius = torch.norm(neighbors - mean.unsqueeze(0), dim=1).mean().clamp_min(1e-8)
    return LocalPCAManifold(
        mean=mean,
        components=components,
        neighborhood_radius=neighborhood_radius,
    )


def normalized_normal_residual(query_vector: torch.Tensor, manifold: LocalPCAManifold) -> float:
    query_vector = query_vector.to(dtype=torch.float32)
    centered = query_vector - manifold.mean
    projection = manifold.components.T @ (manifold.components @ centered)
    residual = centered - projection
    return float(torch.norm(residual) / manifold.neighborhood_radius)


def _normalized_neighbor_residual(
    query_vector: torch.Tensor,
    reference_vectors: torch.Tensor,
    *,
    k_neighbors: int,
) -> float:
    reference_vectors = reference_vectors.to(dtype=torch.float32)
    query_vector = query_vector.to(dtype=torch.float32)
    distances = torch.norm(reference_vectors - query_vector.unsqueeze(0), dim=1)
    topk = torch.topk(distances, k=min(k_neighbors, reference_vectors.shape[0]), largest=False)
    neighbor_distances = topk.values
    radius = neighbor_distances.mean().clamp_min(1e-8)
    return float(neighbor_distances.min() / radius), float(radius)


def clean_reference_entries(entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [entry for entry in entries if entry.get("parsed_answer") == 1]


def build_reference_bank(
    entries: Sequence[dict[str, object]],
    *,
    min_points: int = 0,
) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, list[torch.Tensor]]] = {}
    for entry in entries:
        object_name = str(entry["object_name"])
        layer_vectors = entry["layer_vectors"]
        selected_layers = entry["selected_layers"]
        bank.setdefault(object_name, {})
        for offset, layer_index in enumerate(selected_layers):
            bank[object_name].setdefault(int(layer_index), []).append(layer_vectors[offset].detach().cpu())

    return {
        object_name: {
            layer_index: torch.stack(vectors, dim=0)
            for layer_index, vectors in layer_map.items()
            if len(vectors) >= min_points
        }
        for object_name, layer_map in bank.items()
        if any(len(vectors) >= min_points for vectors in layer_map.values())
    }


def compute_reference_bank_stats(
    entries: Sequence[dict[str, object]],
    *,
    k_neighbors: int = 32,
) -> dict[str, dict[int, dict[str, float]]]:
    bank = build_reference_bank(entries)
    stats_map: dict[str, dict[int, dict[str, float]]] = {}
    for object_name, layer_map in bank.items():
        stats_map[object_name] = {}
        for layer_index, vectors in layer_map.items():
            count = int(vectors.shape[0])
            residuals: list[float] = []
            neighbor_residuals: list[float] = []
            neighbor_radii: list[float] = []
            if count > 1:
                for index in range(count):
                    mask = [offset for offset in range(count) if offset != index]
                    leave_one_out = vectors[mask]
                    query = vectors[index]
                    manifold = fit_local_pca_manifold(
                        leave_one_out,
                        query,
                        k_neighbors=min(k_neighbors, leave_one_out.shape[0]),
                    )
                    residuals.append(normalized_normal_residual(query, manifold))
                    neighbor_residual, neighbor_radius = _normalized_neighbor_residual(
                        query,
                        leave_one_out,
                        k_neighbors=min(k_neighbors, leave_one_out.shape[0]),
                    )
                    neighbor_residuals.append(neighbor_residual)
                    neighbor_radii.append(neighbor_radius)
            radius_array = np.asarray(neighbor_radii if neighbor_radii else [0.0], dtype=np.float32)
            residual_array = np.asarray(residuals if residuals else [0.0], dtype=np.float32)
            neighbor_residual_array = np.asarray(
                neighbor_residuals if neighbor_residuals else [0.0],
                dtype=np.float32,
            )
            stats_map[object_name][int(layer_index)] = {
                "count": float(count),
                "residual_mean": float(residual_array.mean()),
                "residual_std": float(residual_array.std()),
                "neighbor_residual_mean": float(neighbor_residual_array.mean()),
                "neighbor_residual_std": float(neighbor_residual_array.std()),
                "neighbor_radius_mean": float(radius_array.mean()),
                "neighbor_radius_std": float(radius_array.std()),
                "neighbor_radius_q10": float(np.quantile(radius_array, 0.10)),
                "neighbor_radius_q50": float(np.quantile(radius_array, 0.50)),
                "neighbor_radius_q90": float(np.quantile(radius_array, 0.90)),
                "supports_manifold": float(count >= k_neighbors),
            }
    return stats_map
