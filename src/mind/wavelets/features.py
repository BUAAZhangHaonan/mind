"""Haar wavelet feature extraction for drift curves."""

from __future__ import annotations

import numpy as np
import pywt


def extract_wavelet_features(curve: np.ndarray) -> dict[str, float]:
    curve = np.asarray(curve, dtype=np.float32)
    coeffs = pywt.wavedec(curve, "haar")
    features = {f"drift_{index}": float(value) for index, value in enumerate(curve.tolist())}
    features["approx_energy"] = float(np.square(coeffs[0]).sum())
    for level, detail in enumerate(coeffs[1:], start=1):
        features[f"detail_energy_l{level}"] = float(np.square(detail).sum())
    features["max_drift"] = float(curve.max())
    features["mean_drift"] = float(curve.mean())
    features["peak_layer_index"] = float(curve.argmax())
    return features
