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


def standardize_drift_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    std = curve.std()
    if std < 1e-8:
        return np.zeros_like(curve)
    return (curve - curve.mean()) / std


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
