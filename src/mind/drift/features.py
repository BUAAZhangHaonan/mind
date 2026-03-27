"""Drift curve construction."""

from __future__ import annotations

import numpy as np
import torch

from mind.manifolds import fit_local_pca_manifold, normalized_normal_residual


def compute_drift_curve(
    *,
    layer_vectors: torch.Tensor,
    selected_layers: list[int],
    object_name: str,
    reference_bank: dict[str, dict[int, torch.Tensor]],
    k_neighbors: int = 32,
) -> np.ndarray:
    if object_name not in reference_bank:
        raise KeyError(f"Missing reference bank for object {object_name}")
    if layer_vectors.shape[0] != len(selected_layers):
        raise ValueError("layer_vectors and selected_layers must align")

    scores = []
    object_bank = reference_bank[object_name]
    for offset, layer_index in enumerate(selected_layers):
        reference_vectors = object_bank[int(layer_index)]
        manifold = fit_local_pca_manifold(
            reference_vectors,
            layer_vectors[offset],
            k_neighbors=k_neighbors,
        )
        scores.append(normalized_normal_residual(layer_vectors[offset], manifold))
    return np.asarray(scores, dtype=np.float32)


def standardize_drift_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    std = curve.std()
    if std < 1e-8:
        return np.zeros_like(curve)
    return (curve - curve.mean()) / std
