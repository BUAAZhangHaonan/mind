"""Local PCA manifolds for MIND."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

import numpy as np
import torch


SHARED_BANK_KEY = "__shared__"
SHUFFLED_OBJECT_MAP_FILENAME = "shuffled_object_map.json"


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


def _compute_layer_statistics(
    vectors: torch.Tensor,
    *,
    k_neighbors: int,
    variance_threshold: float = 0.9,
    max_components: int = 32,
    batch_size: int = 64,
) -> dict[str, float]:
    compute_device = torch.device("cuda" if torch.cuda.is_available() and vectors.shape[0] >= 256 else "cpu")
    vectors = vectors.to(device=compute_device, dtype=torch.float32)
    count = int(vectors.shape[0])
    if count <= 1:
        return {
            "count": float(count),
            "residual_mean": 0.0,
            "residual_std": 0.0,
            "neighbor_residual_mean": 0.0,
            "neighbor_residual_std": 0.0,
            "neighbor_radius_mean": 0.0,
            "neighbor_radius_std": 0.0,
            "neighbor_radius_q10": 0.0,
            "neighbor_radius_q50": 0.0,
            "neighbor_radius_q90": 0.0,
            "supports_manifold": float(count >= k_neighbors),
        }

    neighbor_count = min(k_neighbors, count - 1)
    distance_matrix = torch.cdist(vectors, vectors)
    diagonal = torch.arange(count, device=distance_matrix.device)
    distance_matrix[diagonal, diagonal] = torch.inf
    neighbor_distances, neighbor_indices = torch.topk(
        distance_matrix,
        k=neighbor_count,
        largest=False,
        dim=1,
    )
    neighbor_residuals = neighbor_distances[:, 0] / neighbor_distances.mean(dim=1).clamp_min(1e-8)
    residual_batches: list[torch.Tensor] = []
    neighbor_radii: list[torch.Tensor] = []

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch_indices = neighbor_indices[start:stop]
        queries = vectors[start:stop]
        neighbors = vectors[batch_indices]
        means = neighbors.mean(dim=1)
        centered_neighbors = neighbors - means.unsqueeze(1)
        gram = centered_neighbors @ centered_neighbors.transpose(1, 2)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        variances = torch.flip(eigenvalues.clamp_min(0.0), dims=[1])
        left_vectors = torch.flip(eigenvectors, dims=[2])
        variance_ratio = variances / variances.sum(dim=1, keepdim=True).clamp_min(1e-8)
        cumulative = torch.cumsum(variance_ratio, dim=1)
        component_count = (cumulative < variance_threshold).sum(dim=1) + 1
        component_count = component_count.clamp(max=min(max_components, left_vectors.shape[2]))
        centered_queries = queries - means
        manifold_radii = torch.norm(centered_neighbors, dim=2).mean(dim=1).clamp_min(1e-8)
        squared_query_norm = centered_queries.square().sum(dim=1)
        batch_residuals = []
        for batch_offset in range(stop - start):
            component_limit = int(component_count[batch_offset].item())
            basis_left = left_vectors[batch_offset, :, :component_limit]
            singular_values = variances[batch_offset, :component_limit].sqrt().clamp_min(1e-8)
            projected_coefficients = (
                basis_left.transpose(0, 1)
                @ (centered_neighbors[batch_offset] @ centered_queries[batch_offset])
            ) / singular_values
            projection_norm = projected_coefficients.square().sum()
            residual_norm = torch.sqrt(
                (squared_query_norm[batch_offset] - projection_norm).clamp_min(0.0)
            )
            batch_residuals.append(residual_norm / manifold_radii[batch_offset])
        residual_batches.append(torch.stack(batch_residuals))
        neighbor_radii.append(neighbor_distances[start:stop].mean(dim=1))

    residual_array = torch.cat(residual_batches).detach().cpu().numpy().astype(np.float32)
    neighbor_residual_array = neighbor_residuals.detach().cpu().numpy().astype(np.float32)
    radius_array = torch.cat(neighbor_radii).detach().cpu().numpy().astype(np.float32)
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()

    return {
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


def clean_reference_entries(entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [entry for entry in entries if entry.get("parsed_answer") == 1]


def build_shuffled_object_mapping(
    object_names: Sequence[str],
    *,
    shuffle_seed: int = 13,
) -> dict[str, str]:
    unique_names = sorted({str(name) for name in object_names})
    if len(unique_names) < 2:
        raise ValueError("Shuffled-object banks require at least two distinct objects.")
    shuffled = list(unique_names)
    rng = np.random.default_rng(shuffle_seed)
    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    return {
        destination: source
        for destination, source in zip(unique_names, shuffled)
    }


def resolve_reference_scope_key(object_name: str, bank_scope: str) -> str:
    if bank_scope in {"object", "shuffled_object"}:
        return object_name
    if bank_scope == "shared":
        return SHARED_BANK_KEY
    raise ValueError(f"Unsupported bank scope: {bank_scope}")


def build_reference_bank(
    entries: Sequence[dict[str, object]],
    *,
    min_points: int = 0,
    bank_scope: str = "object",
    shuffle_seed: int = 13,
    shuffled_object_mapping: dict[str, str] | None = None,
) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, list[torch.Tensor]]] = {}
    source_to_destination: dict[str, str] = {}
    if bank_scope == "shuffled_object":
        mapping = shuffled_object_mapping or build_shuffled_object_mapping(
            [str(entry["object_name"]) for entry in entries],
            shuffle_seed=shuffle_seed,
        )
        source_to_destination = {source: destination for destination, source in mapping.items()}
    for entry in entries:
        entry_object_name = str(entry["object_name"])
        if bank_scope == "shuffled_object":
            object_name = source_to_destination[entry_object_name]
        else:
            object_name = resolve_reference_scope_key(entry_object_name, bank_scope)
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
    bank_scope: str = "object",
    shuffle_seed: int = 13,
    shuffled_object_mapping: dict[str, str] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    bank = build_reference_bank(
        entries,
        bank_scope=bank_scope,
        shuffle_seed=shuffle_seed,
        shuffled_object_mapping=shuffled_object_mapping,
    )
    stats_map: dict[str, dict[int, dict[str, float]]] = {}
    for object_name, layer_map in bank.items():
        stats_map[object_name] = {}
        for layer_index, vectors in layer_map.items():
            stats_map[object_name][int(layer_index)] = _compute_layer_statistics(
                vectors,
                k_neighbors=k_neighbors,
            )
    return stats_map


def compute_reference_bank_stats_from_bank(
    bank: dict[str, dict[int, torch.Tensor]],
    *,
    k_neighbors: int = 32,
) -> dict[str, dict[int, dict[str, float]]]:
    stats_map: dict[str, dict[int, dict[str, float]]] = {}
    for object_name, layer_map in bank.items():
        stats_map[object_name] = {}
        for layer_index, vectors in layer_map.items():
            stats_map[object_name][int(layer_index)] = _compute_layer_statistics(
                vectors,
                k_neighbors=k_neighbors,
            )
    return stats_map
