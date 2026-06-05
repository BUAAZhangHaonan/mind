"""Teacher-Bagua wavelet feature extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_LAYER_SHAPE = (36, 4096)
WINDOW_SIZE = 4
WINDOW_STRIDE = 4
NUM_WINDOWS = 9
TIME_FEATURES_PER_DIM = 16
FREQ_FEATURES_PER_DIM = 12
FEATURES_PER_DIM = TIME_FEATURES_PER_DIM + FREQ_FEATURES_PER_DIM
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True)
class TeacherBaguaConfig:
    wavelet: str = "db2"
    level: int = 2
    threshold: str = "universal_soft"
    epsilon: float = DEFAULT_EPSILON
    feature_batch_size: int = 32


@dataclass(frozen=True)
class TeacherBaguaFeatureResult:
    features: np.ndarray
    epsilon_usage: dict[str, int]


def extract_teacher_bagua_features(
    layer_vectors: Any,
    config: TeacherBaguaConfig | Mapping[str, Any] | None = None,
) -> TeacherBaguaFeatureResult:
    """Return Teacher-Bagua features with shape ``(9, 4096 * 28)``."""

    cfg = _coerce_config(config)
    vectors = _as_layer_vectors(layer_vectors)
    result = _extract_teacher_bagua_batch(vectors[np.newaxis, :, :], cfg)
    features = result.features[0]
    expected_shape = (NUM_WINDOWS, EXPECTED_LAYER_SHAPE[1] * FEATURES_PER_DIM)
    if features.shape != expected_shape:
        raise ValueError(f"Teacher-Bagua features shape {features.shape} != {expected_shape}")
    _raise_if_non_finite(features, name="teacher_bagua_features")
    return TeacherBaguaFeatureResult(features=features, epsilon_usage=result.epsilon_usage)


def write_teacher_memmap(
    entries: Iterable[Mapping[str, Any]],
    labels: Iterable[int],
    output_path: str | Path,
    config: TeacherBaguaConfig | Mapping[str, Any] | None = None,
    *,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write Teacher-Bagua features to a float32 memmap and metadata JSON."""

    cfg = _coerce_config(config)
    rows = list(entries)
    label_array = np.asarray(list(labels), dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError("labels must be a 1D iterable")
    if label_array.shape[0] != len(rows):
        raise ValueError("labels length must match entries length")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(rows), NUM_WINDOWS, EXPECTED_LAYER_SHAPE[1] * FEATURES_PER_DIM)
    mmap = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    total_usage: Counter[str] = Counter()
    for start in range(0, len(rows), cfg.feature_batch_size):
        stop = min(start + cfg.feature_batch_size, len(rows))
        layer_batch = []
        for entry in rows[start:stop]:
            if "layer_vectors" not in entry:
                raise ValueError("each entry must contain layer_vectors")
            layer_batch.append(_as_layer_vectors(entry["layer_vectors"]))
        result = _extract_teacher_bagua_batch(np.stack(layer_batch, axis=0), cfg)
        if result.features.shape != (stop - start, NUM_WINDOWS, EXPECTED_LAYER_SHAPE[1] * FEATURES_PER_DIM):
            raise ValueError("Teacher-Bagua batch features have unexpected shape")
        _raise_if_non_finite(result.features, name="teacher_bagua_features")
        mmap[start:stop] = result.features
        total_usage.update(result.epsilon_usage)
    mmap.flush()

    metadata = {
        "shape": list(shape),
        "dtype": "float32",
        "sequence_len": NUM_WINDOWS,
        "feature_dim": EXPECTED_LAYER_SHAPE[1] * FEATURES_PER_DIM,
        "labels_shape": list(label_array.shape),
        "config": _config_payload(cfg),
        "epsilon_usage": dict(total_usage),
    }
    meta_path = Path(metadata_path) if metadata_path is not None else path.with_suffix(path.suffix + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _coerce_config(config: TeacherBaguaConfig | Mapping[str, Any] | None) -> TeacherBaguaConfig:
    if config is None:
        return TeacherBaguaConfig()
    if isinstance(config, TeacherBaguaConfig):
        cfg = config
    else:
        cfg = TeacherBaguaConfig(
            wavelet=str(config.get("wavelet", TeacherBaguaConfig.wavelet)),
            level=int(config.get("level", TeacherBaguaConfig.level)),
            threshold=str(config.get("threshold", TeacherBaguaConfig.threshold)),
            epsilon=float(config.get("epsilon", DEFAULT_EPSILON)),
            feature_batch_size=int(config.get("feature_batch_size", TeacherBaguaConfig.feature_batch_size)),
        )
    if cfg.level < 0:
        raise ValueError("level must be non-negative")
    if cfg.threshold != "universal_soft":
        raise ValueError("Teacher-Bagua only supports threshold='universal_soft'")
    if cfg.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if cfg.feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    return cfg


def _config_payload(config: TeacherBaguaConfig) -> dict[str, Any]:
    return {
        "wavelet": config.wavelet,
        "level": int(config.level),
        "threshold": config.threshold,
        "epsilon": float(config.epsilon),
    }


def _as_layer_vectors(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.shape != EXPECTED_LAYER_SHAPE:
        raise ValueError(f"layer_vectors must have shape {EXPECTED_LAYER_SHAPE}")
    _raise_if_non_finite(array, name="layer_vectors")
    return array


def _extract_teacher_bagua_batch(
    layer_batch: Any,
    config: TeacherBaguaConfig | Mapping[str, Any] | None = None,
) -> TeacherBaguaFeatureResult:
    cfg = _coerce_config(config)
    vectors = _as_layer_batch(layer_batch)
    denoised = _wavelet_denoise_layer_batch(vectors, cfg)
    epsilon_usage: Counter[str] = Counter()

    rows: list[np.ndarray] = []
    for start in range(0, EXPECTED_LAYER_SHAPE[0], WINDOW_STRIDE):
        stop = start + WINDOW_SIZE
        window = denoised[:, :, start:stop]
        if window.shape[2] != WINDOW_SIZE:
            raise ValueError("layer count must produce exactly 9 windows of length 4")
        rows.append(_window_features_batch(window, epsilon=float(cfg.epsilon), usage=epsilon_usage))

    features = np.stack(rows, axis=1).astype(np.float32, copy=False)
    expected_shape = (vectors.shape[0], NUM_WINDOWS, EXPECTED_LAYER_SHAPE[1] * FEATURES_PER_DIM)
    if features.shape != expected_shape:
        raise ValueError(f"Teacher-Bagua batch features shape {features.shape} != {expected_shape}")
    _raise_if_non_finite(features, name="teacher_bagua_features")
    return TeacherBaguaFeatureResult(features=features, epsilon_usage=dict(epsilon_usage))


def _as_layer_batch(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    expected_shape = (EXPECTED_LAYER_SHAPE[0], EXPECTED_LAYER_SHAPE[1])
    if array.ndim != 3 or array.shape[1:] != expected_shape:
        raise ValueError(f"layer batch must have shape (batch, {expected_shape[0]}, {expected_shape[1]})")
    _raise_if_non_finite(array, name="layer_batch")
    return array


def _wavelet_denoise_matrix(traces: np.ndarray, config: TeacherBaguaConfig) -> np.ndarray:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("Teacher-Bagua feature extraction requires pywt") from exc

    try:
        wavelet = pywt.Wavelet(config.wavelet)
        max_level = pywt.dwt_max_level(traces.shape[1], wavelet.dec_len)
        if config.level > max_level:
            raise RuntimeError(
                f"wavelet level {config.level} is not feasible for length {traces.shape[1]} "
                f"and wavelet {config.wavelet}"
            )
        coeffs = pywt.wavedec(traces.astype(np.float64), config.wavelet, level=int(config.level), axis=1)
        if len(coeffs) <= 1:
            denoised = traces.astype(np.float32, copy=True)
        else:
            detail = coeffs[-1]
            if detail.size:
                detail_center = np.median(detail, axis=1, keepdims=True)
                sigma = np.median(np.abs(detail - detail_center), axis=1, keepdims=True) / 0.6745
            else:
                sigma = np.zeros((traces.shape[0], 1), dtype=np.float64)
            threshold = sigma * np.sqrt(2.0 * np.log(float(traces.shape[1])))
            filtered = [coeffs[0]]
            filtered.extend(pywt.threshold(coeff, threshold, mode="soft") for coeff in coeffs[1:])
            restored = pywt.waverec(filtered, config.wavelet, axis=1)[:, : traces.shape[1]]
            denoised = np.asarray(restored, dtype=np.float32)
    except Exception as exc:
        if isinstance(exc, (ImportError, RuntimeError)):
            raise
        raise RuntimeError("Teacher-Bagua pywt denoising failed") from exc
    _raise_if_non_finite(denoised, name="denoised_layer_traces")
    return denoised


def _wavelet_denoise_layer_batch(layer_batch: np.ndarray, config: TeacherBaguaConfig) -> np.ndarray:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("Teacher-Bagua feature extraction requires pywt") from exc

    traces = np.swapaxes(layer_batch, 1, 2).astype(np.float32, copy=False)
    try:
        wavelet = pywt.Wavelet(config.wavelet)
        max_level = pywt.dwt_max_level(traces.shape[2], wavelet.dec_len)
        if config.level > max_level:
            raise RuntimeError(
                f"wavelet level {config.level} is not feasible for length {traces.shape[2]} "
                f"and wavelet {config.wavelet}"
            )
        coeffs = pywt.wavedec(traces.astype(np.float64), config.wavelet, level=int(config.level), axis=2)
        if len(coeffs) <= 1:
            denoised = traces.astype(np.float32, copy=True)
        else:
            detail = coeffs[-1]
            if detail.size:
                detail_center = np.median(detail, axis=2, keepdims=True)
                sigma = np.median(np.abs(detail - detail_center), axis=2, keepdims=True) / 0.6745
            else:
                sigma = np.zeros((traces.shape[0], traces.shape[1], 1), dtype=np.float64)
            threshold = sigma * np.sqrt(2.0 * np.log(float(traces.shape[2])))
            filtered = [coeffs[0]]
            filtered.extend(pywt.threshold(coeff, threshold, mode="soft") for coeff in coeffs[1:])
            restored = pywt.waverec(filtered, config.wavelet, axis=2)[:, :, : traces.shape[2]]
            denoised = np.asarray(restored, dtype=np.float32)
    except Exception as exc:
        if isinstance(exc, (ImportError, RuntimeError)):
            raise
        raise RuntimeError("Teacher-Bagua pywt denoising failed") from exc
    _raise_if_non_finite(denoised, name="denoised_layer_traces")
    return denoised


def _wavelet_denoise_trace(trace: np.ndarray, *, pywt: Any, config: TeacherBaguaConfig) -> np.ndarray:
    coeffs = pywt.wavedec(trace.astype(np.float64), config.wavelet, level=int(config.level))
    if len(coeffs) <= 1:
        return trace.astype(np.float32, copy=True)
    detail = coeffs[-1]
    sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.6745) if detail.size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(float(trace.size)))
    filtered = [coeffs[0]]
    filtered.extend(pywt.threshold(coeff, threshold, mode="soft") for coeff in coeffs[1:])
    restored = pywt.waverec(filtered, config.wavelet)[: trace.size]
    return np.asarray(restored, dtype=np.float32)


def _window_features(
    window: np.ndarray,
    *,
    epsilon: float,
    usage: Counter[str],
) -> np.ndarray:
    values = window.astype(np.float64, copy=False)
    diffs = np.diff(values, axis=1)
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    value_range = maximum - minimum
    slope = (values[:, -1] - values[:, 0]) / float(values.shape[1] - 1)
    centered = values - mean[:, np.newaxis]
    centered_second = np.mean(np.square(centered), axis=1)
    skew = _safe_div_vector(
        np.mean(centered**3, axis=1),
        centered_second**1.5,
        epsilon,
        usage,
        "time_skew",
    )
    kurtosis = _safe_div_vector(
        np.mean(centered**4, axis=1),
        centered_second**2,
        epsilon,
        usage,
        "time_kurtosis",
    )
    time_values = np.column_stack(
        [
            mean,
            std,
            minimum,
            maximum,
            values[:, 0],
            values[:, -1],
            value_range,
            slope,
            np.mean(np.abs(values), axis=1),
            np.mean(np.abs(diffs), axis=1) if diffs.shape[1] else np.zeros(values.shape[0], dtype=np.float64),
            np.median(values, axis=1),
            np.percentile(values, 25.0, axis=1),
            np.percentile(values, 75.0, axis=1),
            _safe_div_vector(std, np.abs(mean), epsilon, usage, "time_coeff_var"),
            skew,
            kurtosis,
        ]
    )

    spectrum = np.fft.rfft(values, axis=1)
    magnitude = np.abs(spectrum)
    power = np.square(magnitude)
    freqs = np.fft.rfftfreq(values.shape[1])
    total_power = np.sum(power, axis=1)
    total_magnitude = np.sum(magnitude, axis=1)
    probability = power / np.maximum(total_power[:, np.newaxis], epsilon)
    usage["freq_power_probability"] += int(np.sum(total_power <= epsilon))
    centroid = _safe_div_vector(np.sum(freqs[np.newaxis, :] * power, axis=1), total_power, epsilon, usage, "freq_centroid")
    spread = np.sqrt(
        np.maximum(
            _safe_div_vector(
                np.sum(np.square(freqs[np.newaxis, :] - centroid[:, np.newaxis]) * power, axis=1),
                total_power,
                epsilon,
                usage,
                "freq_spread",
            ),
            0.0,
        )
    )
    usage["freq_flatness"] += int(np.sum(magnitude <= epsilon))
    positive_magnitude = np.maximum(magnitude, epsilon)
    geometric_mean = np.exp(np.mean(np.log(positive_magnitude), axis=1))
    high_power = np.sum(power[:, 1:], axis=1)
    freq_values = np.column_stack(
        [
            magnitude[:, 0],
            magnitude[:, 1] if magnitude.shape[1] > 1 else np.zeros(values.shape[0], dtype=np.float64),
            magnitude[:, 2] if magnitude.shape[1] > 2 else np.zeros(values.shape[0], dtype=np.float64),
            total_power,
            power[:, 0],
            high_power,
            _safe_div_vector(high_power, total_power, epsilon, usage, "freq_high_ratio"),
            centroid,
            spread,
            _entropy_matrix(probability, epsilon=epsilon, usage=usage, name="freq_entropy"),
            np.argmax(power, axis=1).astype(np.float64),
            _safe_div_vector(
                geometric_mean,
                total_magnitude / float(magnitude.shape[1]),
                epsilon,
                usage,
                "freq_flatness",
            ),
        ]
    )
    return np.concatenate([time_values, freq_values], axis=1).reshape(-1).astype(np.float32, copy=False)


def _window_features_batch(
    window: np.ndarray,
    *,
    epsilon: float,
    usage: Counter[str],
) -> np.ndarray:
    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("window batch must have shape (batch, hidden_dim, window_size)")
    batch_size, hidden_dim, window_size = values.shape
    flat = values.reshape(batch_size * hidden_dim, window_size)
    return _window_features(flat, epsilon=epsilon, usage=usage).reshape(
        batch_size,
        hidden_dim * FEATURES_PER_DIM,
    )


def _time_features(trace: np.ndarray, *, epsilon: float, usage: Counter[str]) -> np.ndarray:
    values = trace.astype(np.float64, copy=False)
    diffs = np.diff(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    value_range = maximum - minimum
    slope = (float(values[-1]) - float(values[0])) / float(values.size - 1)
    centered = values - mean
    centered_second = float(np.mean(np.square(centered)))
    skew = _safe_div(float(np.mean(centered**3)), centered_second ** 1.5, epsilon, usage, "time_skew")
    kurtosis = _safe_div(float(np.mean(centered**4)), centered_second**2, epsilon, usage, "time_kurtosis")
    return np.asarray(
        [
            mean,
            std,
            minimum,
            maximum,
            float(values[0]),
            float(values[-1]),
            value_range,
            slope,
            float(np.mean(np.abs(values))),
            float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
            float(np.median(values)),
            float(np.percentile(values, 25.0)),
            float(np.percentile(values, 75.0)),
            _safe_div(std, abs(mean), epsilon, usage, "time_coeff_var"),
            skew,
            kurtosis,
        ],
        dtype=np.float32,
    )


def _frequency_features(trace: np.ndarray, *, epsilon: float, usage: Counter[str]) -> np.ndarray:
    values = trace.astype(np.float64, copy=False)
    spectrum = np.fft.rfft(values)
    magnitude = np.abs(spectrum)
    power = np.square(magnitude)
    freqs = np.fft.rfftfreq(values.size)
    total_power = float(np.sum(power))
    total_magnitude = float(np.sum(magnitude))
    probability = power / max(total_power, epsilon)
    if total_power <= epsilon:
        usage["freq_power_probability"] += 1
    centroid = _safe_div(float(np.sum(freqs * power)), total_power, epsilon, usage, "freq_centroid")
    spread = np.sqrt(
        max(_safe_div(float(np.sum(((freqs - centroid) ** 2) * power)), total_power, epsilon, usage, "freq_spread"), 0.0)
    )
    positive_magnitude = np.maximum(magnitude, epsilon)
    if np.any(magnitude <= epsilon):
        usage["freq_flatness"] += int(np.sum(magnitude <= epsilon))
    geometric_mean = float(np.exp(np.mean(np.log(positive_magnitude))))
    return np.asarray(
        [
            float(magnitude[0]),
            float(magnitude[1]) if magnitude.size > 1 else 0.0,
            float(magnitude[2]) if magnitude.size > 2 else 0.0,
            total_power,
            float(power[0]),
            float(np.sum(power[1:])),
            _safe_div(float(np.sum(power[1:])), total_power, epsilon, usage, "freq_high_ratio"),
            centroid,
            float(spread),
            _entropy(probability, epsilon=epsilon, usage=usage, name="freq_entropy"),
            float(np.argmax(power)),
            _safe_div(geometric_mean, total_magnitude / float(magnitude.size), epsilon, usage, "freq_flatness"),
        ],
        dtype=np.float32,
    )


def _safe_div(
    numerator: float,
    denominator: float,
    epsilon: float,
    usage: Counter[str],
    name: str,
) -> float:
    denom = float(denominator)
    if abs(denom) <= epsilon:
        usage[name] += 1
        denom = epsilon if denom >= 0.0 else -epsilon
    return float(numerator) / denom


def _safe_div_vector(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float,
    usage: Counter[str],
    name: str,
) -> np.ndarray:
    numer = np.asarray(numerator, dtype=np.float64)
    denom = np.asarray(denominator, dtype=np.float64)
    adjusted = denom.copy()
    mask = np.abs(adjusted) <= epsilon
    usage[name] += int(np.sum(mask))
    adjusted[mask & (adjusted >= 0.0)] = epsilon
    adjusted[mask & (adjusted < 0.0)] = -epsilon
    return numer / adjusted


def _entropy(probability: np.ndarray, *, epsilon: float, usage: Counter[str], name: str) -> float:
    values = np.asarray(probability, dtype=np.float64)
    if np.any(values <= epsilon):
        usage[name] += int(np.sum(values <= epsilon))
    safe = np.maximum(values, epsilon)
    return float(-np.sum(safe * np.log(safe)))


def _entropy_matrix(probability: np.ndarray, *, epsilon: float, usage: Counter[str], name: str) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    usage[name] += int(np.sum(values <= epsilon))
    safe = np.maximum(values, epsilon)
    return -np.sum(safe * np.log(safe), axis=1)


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
