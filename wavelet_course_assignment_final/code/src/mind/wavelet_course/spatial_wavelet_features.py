"""Spatial hidden-dimension DWT features for wavelet-course experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common_feature_protocols import EPSILON, STAT28_NAMES, stat28


DEFAULT_EXPECTED_NUM_LAYERS = 36
DEFAULT_EXPECTED_HIDDEN_DIM = 4096


@dataclass(frozen=True)
class SpatialWaveletConfig:
    wavelet: str = "db2"
    level: int = 2
    threshold: str = "universal_soft"
    mode: str = "symmetric"
    expected_num_layers: int = DEFAULT_EXPECTED_NUM_LAYERS
    expected_hidden_dim: int = DEFAULT_EXPECTED_HIDDEN_DIM
    epsilon: float = EPSILON


def spatial_dwt_stat28_sequence(
    layer_vectors: Any,
    config: SpatialWaveletConfig | None = None,
) -> np.ndarray:
    """Denoise each layer across hidden dimension, then emit per-layer stat28."""

    cfg = config or SpatialWaveletConfig()
    values = _as_layer_vectors(layer_vectors, config=cfg)
    denoised = spatial_dwt_denoise(values, cfg)
    features = stat28(denoised, epsilon=float(cfg.epsilon)).astype(np.float32, copy=False)
    expected_shape = (int(cfg.expected_num_layers), len(STAT28_NAMES))
    if features.shape != expected_shape:
        raise ValueError(f"spatial stat28 sequence shape {features.shape} != {expected_shape}")
    _raise_if_non_finite(features, name="spatial stat28 sequence")
    return features


def spatial_dwt_stat28_sequence_batch(
    layer_vector_batch: Any,
    config: SpatialWaveletConfig | None = None,
) -> np.ndarray:
    """Batch version of spatial_dwt_stat28_sequence for ``(N, layers, hidden)``."""

    cfg = config or SpatialWaveletConfig()
    values = _as_layer_vector_batch(layer_vector_batch, config=cfg)
    flat = values.reshape((-1, values.shape[-1]))
    denoised = _dwt_denoise_2d(flat, cfg).reshape(values.shape)
    features = stat28(denoised, epsilon=float(cfg.epsilon)).astype(np.float32, copy=False)
    expected_shape = (values.shape[0], int(cfg.expected_num_layers), len(STAT28_NAMES))
    if features.shape != expected_shape:
        raise ValueError(f"spatial stat28 batch shape {features.shape} != {expected_shape}")
    _raise_if_non_finite(features, name="spatial stat28 batch")
    return features


def spatial_dwt_denoise(
    layer_vectors: Any,
    config: SpatialWaveletConfig | None = None,
) -> np.ndarray:
    """Apply 1D DWT denoising along hidden dimension for all layers at once."""

    cfg = config or SpatialWaveletConfig()
    values = _as_layer_vectors(layer_vectors, config=cfg)
    return _dwt_denoise_2d(values, cfg)


def _dwt_denoise_2d(values: np.ndarray, config: SpatialWaveletConfig) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != int(config.expected_hidden_dim):
        raise ValueError(
            f"spatial hidden matrix shape {values.shape} does not have hidden_dim "
            f"{int(config.expected_hidden_dim)}"
        )
    _raise_if_non_finite(values, name="spatial hidden matrix")
    pywt = _import_pywt()
    if int(config.level) < 0:
        raise ValueError("wavelet level must be non-negative")
    if str(config.threshold) not in {"universal_soft", "none", "not_applicable"}:
        raise ValueError(f"unsupported spatial threshold: {config.threshold!r}")
    try:
        wavelet = pywt.Wavelet(str(config.wavelet))
        max_level = int(pywt.dwt_max_level(values.shape[1], wavelet.dec_len))
    except Exception as error:
        raise RuntimeError("spatial hidden wavelet validation failed") from error
    if int(config.level) > max_level:
        raise ValueError(
            f"wavelet level {int(config.level)} is not feasible for hidden_dim "
            f"{values.shape[1]} and wavelet {config.wavelet}"
        )
    if int(config.level) == 0:
        return values.astype(np.float32, copy=True)
    try:
        coeffs = pywt.wavedec(
            values.astype(np.float64, copy=False),
            str(config.wavelet),
            mode=str(config.mode),
            level=int(config.level),
            axis=1,
        )
        if str(config.threshold) == "universal_soft":
            threshold = _universal_soft_threshold(coeffs[-1], signal_length=values.shape[1])
            filtered = [coeffs[0]]
            filtered.extend(pywt.threshold(coeff, threshold, mode="soft") for coeff in coeffs[1:])
        else:
            filtered = coeffs
        restored = pywt.waverec(filtered, str(config.wavelet), mode=str(config.mode), axis=1)
    except Exception as error:
        raise RuntimeError("spatial hidden pywt denoising failed") from error
    restored = np.asarray(restored, dtype=np.float64)[:, : values.shape[1]]
    if restored.shape != values.shape:
        raise RuntimeError(f"spatial hidden reconstruction shape {restored.shape} != {values.shape}")
    output = restored.astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="spatial hidden reconstruction")
    return output


def _universal_soft_threshold(detail_coeffs: np.ndarray, *, signal_length: int) -> np.ndarray:
    if signal_length <= 1:
        raise ValueError("signal_length must be greater than 1")
    sigma = np.median(np.abs(np.asarray(detail_coeffs, dtype=np.float64)), axis=1, keepdims=True) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(float(signal_length)))
    _raise_if_non_finite(threshold, name="spatial hidden threshold")
    return threshold


def _as_layer_vectors(values: Any, *, config: SpatialWaveletConfig) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    expected_shape = (int(config.expected_num_layers), int(config.expected_hidden_dim))
    if array.shape != expected_shape:
        raise ValueError(f"layer_vectors shape {array.shape} != {expected_shape}")
    _raise_if_non_finite(array, name="layer_vectors")
    if float(config.epsilon) <= 0.0 or not np.isfinite(float(config.epsilon)):
        raise ValueError("epsilon must be finite and positive")
    return array


def _as_layer_vector_batch(values: Any, *, config: SpatialWaveletConfig) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    expected_tail = (int(config.expected_num_layers), int(config.expected_hidden_dim))
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
        raise ValueError(f"layer_vectors batch shape {array.shape} does not end with {expected_tail}")
    if array.shape[0] == 0:
        raise ValueError("layer_vectors batch must not be empty")
    _raise_if_non_finite(array, name="layer_vectors batch")
    if float(config.epsilon) <= 0.0 or not np.isfinite(float(config.epsilon)):
        raise ValueError("epsilon must be finite and positive")
    return array


def _raise_if_non_finite(values: np.ndarray, *, name: str) -> None:
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _import_pywt() -> Any:
    try:
        import pywt
    except ImportError as error:
        raise ImportError("spatial hidden wavelet feature extraction requires pywt") from error
    return pywt
