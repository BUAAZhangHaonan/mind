"""Drift curve construction."""

from __future__ import annotations

import numpy as np
import torch

from mind.manifolds import (
    fit_local_pca_manifold,
    normalized_normal_residual,
    resolve_reference_scope_key,
)
from mind.wavelets import extract_wavelet_features


def compute_drift_curve(
    *,
    layer_vectors: torch.Tensor,
    selected_layers: list[int],
    object_name: str,
    reference_bank: dict[str, dict[int, torch.Tensor]],
    bank_scope: str = "object",
    k_neighbors: int = 32,
) -> np.ndarray:
    bank_key = resolve_reference_scope_key(object_name, bank_scope)
    if bank_key not in reference_bank:
        raise KeyError(f"Missing reference bank for object {object_name}")
    if layer_vectors.shape[0] != len(selected_layers):
        raise ValueError("layer_vectors and selected_layers must align")

    scores = []
    object_bank = reference_bank[bank_key]
    for offset, layer_index in enumerate(selected_layers):
        reference_vectors = object_bank[int(layer_index)]
        manifold = fit_local_pca_manifold(
            reference_vectors,
            layer_vectors[offset],
            k_neighbors=k_neighbors,
        )
        scores.append(normalized_normal_residual(layer_vectors[offset], manifold))
    return np.asarray(scores, dtype=np.float32)


def _select_batched_drift_device(
    reference_vectors: torch.Tensor,
    *,
    query_count: int,
) -> torch.device:
    pair_elements = int(reference_vectors.shape[0]) * int(reference_vectors.shape[1]) * int(query_count)
    if torch.cuda.is_available() and pair_elements >= 1_000_000:
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_layer_drift_batch(
    *,
    reference_vectors: torch.Tensor,
    query_vectors: torch.Tensor,
    k_neighbors: int,
    variance_threshold: float,
    max_components: int,
    batch_size: int,
) -> np.ndarray:
    if reference_vectors.ndim != 2:
        raise ValueError("reference_vectors must be rank 2")
    if query_vectors.ndim != 2:
        raise ValueError("query_vectors must be rank 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = _select_batched_drift_device(reference_vectors, query_count=query_vectors.shape[0])
    reference_vectors = reference_vectors.to(device=device, dtype=torch.float32)
    query_vectors = query_vectors.to(device=device, dtype=torch.float32)
    neighbor_count = min(k_neighbors, reference_vectors.shape[0])
    layer_scores: list[torch.Tensor] = []

    with torch.no_grad():
        reference_norms = reference_vectors.square().sum(dim=1).unsqueeze(0)
        for start in range(0, int(query_vectors.shape[0]), batch_size):
            queries = query_vectors[start : start + batch_size]
            query_norms = queries.square().sum(dim=1, keepdim=True)
            distance_squares = (
                query_norms
                + reference_norms
                - 2.0 * (queries @ reference_vectors.T)
            ).clamp_min(0.0)
            neighbor_indices = torch.topk(
                distance_squares,
                k=neighbor_count,
                largest=False,
                dim=1,
            ).indices
            neighbors = reference_vectors[neighbor_indices]
            means = neighbors.mean(dim=1)
            centered_neighbors = neighbors - means.unsqueeze(1)

            _, singular_values, vh = torch.linalg.svd(centered_neighbors, full_matrices=False)
            variances = singular_values.square()
            variance_ratio = variances / variances.sum(dim=1, keepdim=True).clamp_min(1e-8)
            cumulative = torch.cumsum(variance_ratio, dim=1)
            component_counts = (cumulative < variance_threshold).sum(dim=1) + 1
            component_counts = component_counts.clamp(max=min(max_components, vh.shape[1]))

            centered_queries = queries - means
            coefficients = torch.bmm(vh, centered_queries.unsqueeze(2)).squeeze(2)
            component_mask = torch.arange(vh.shape[1], device=device).unsqueeze(0) < component_counts.unsqueeze(1)
            coefficients = coefficients * component_mask.to(dtype=coefficients.dtype)
            projections = torch.bmm(vh.transpose(1, 2), coefficients.unsqueeze(2)).squeeze(2)
            residuals = centered_queries - projections
            radii = torch.norm(centered_neighbors, dim=2).mean(dim=1).clamp_min(1e-8)
            layer_scores.append(torch.norm(residuals, dim=1) / radii)

    scores = torch.cat(layer_scores).detach().cpu().numpy().astype(np.float32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def compute_drift_curves_batched(
    *,
    layer_vectors_batch: torch.Tensor,
    selected_layers: list[int],
    object_name: str,
    reference_bank: dict[str, dict[int, torch.Tensor]],
    bank_scope: str = "object",
    bank_key: str | None = None,
    k_neighbors: int = 32,
    batch_size: int = 32,
    variance_threshold: float = 0.9,
    max_components: int = 32,
) -> np.ndarray:
    resolved_bank_key = bank_key or resolve_reference_scope_key(object_name, bank_scope)
    if resolved_bank_key not in reference_bank:
        raise KeyError(f"Missing reference bank for object {object_name}")
    if layer_vectors_batch.ndim != 3:
        raise ValueError("layer_vectors_batch must be rank 3")
    if layer_vectors_batch.shape[1] != len(selected_layers):
        raise ValueError("layer_vectors_batch and selected_layers must align")

    object_bank = reference_bank[resolved_bank_key]
    curves = np.empty((int(layer_vectors_batch.shape[0]), len(selected_layers)), dtype=np.float32)
    for offset, layer_index in enumerate(selected_layers):
        reference_vectors = object_bank[int(layer_index)]
        curves[:, offset] = _compute_layer_drift_batch(
            reference_vectors=reference_vectors,
            query_vectors=layer_vectors_batch[:, offset, :],
            k_neighbors=k_neighbors,
            variance_threshold=variance_threshold,
            max_components=max_components,
            batch_size=batch_size,
        )
    return curves


def calibrate_drift_curve(
    curve: np.ndarray,
    *,
    selected_layers: list[int],
    layer_stats: dict[int, dict[str, float]],
    mean_key: str = "residual_mean",
    std_key: str = "residual_std",
) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    if curve.shape[0] != len(selected_layers):
        raise ValueError("curve and selected_layers must align")
    calibrated = []
    for value, layer_index in zip(curve.tolist(), selected_layers):
        if int(layer_index) not in layer_stats:
            raise KeyError(f"Missing calibration stats for layer {layer_index}")
        stats = layer_stats[int(layer_index)]
        std = max(float(stats[std_key]), 1e-8)
        calibrated.append((float(value) - float(stats[mean_key])) / std)
    return np.asarray(calibrated, dtype=np.float32)


def build_drift_features(
    *,
    raw_curve: np.ndarray,
    calibrated_curve: np.ndarray,
) -> dict[str, float]:
    raw_curve = np.asarray(raw_curve, dtype=np.float32)
    calibrated_curve = np.asarray(calibrated_curve, dtype=np.float32)
    features = {
        f"raw_drift_{index}": float(value)
        for index, value in enumerate(raw_curve.tolist())
    }
    features["raw_max_drift"] = float(raw_curve.max())
    features["raw_mean_drift"] = float(raw_curve.mean())
    features["raw_peak_layer_index"] = float(raw_curve.argmax())
    features.update(extract_wavelet_features(calibrated_curve, prefix="cal_"))
    return features
