"""Common feature protocols for paired-wavelet v2."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common_wavelet import (
    WaveletConfigError,
    _import_pywt,
    _required_level,
    _required_wavelet,
    _spec_scales,
    _swt_max_level,
    _threshold_detail_coeffs,
    _threshold_dwt_coeffs,
    _threshold_for_transform,
    apply_wavelet,
)

EPSILON = 1e-12
STAT28_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "first",
    "last",
    "range",
    "slope",
    "mean_abs",
    "mean_abs_diff",
    "median",
    "p25",
    "p75",
    "coeff_var",
    "skew",
    "kurtosis",
    "fft_mag0",
    "fft_mag1",
    "fft_mag2",
    "total_power",
    "dc_power",
    "high_power",
    "high_power_ratio",
    "spectral_centroid",
    "spectral_spread",
    "spectral_entropy",
    "argmax_power",
    "spectral_flatness",
)
SUMMARY_POOL_NAMES = ("mean", "std", "max", "min", "median", "q90")
WAVELET_SUMMARY_BASE_NAMES = (
    "approximation_energy",
    "total_detail_energy",
    "detail_approx_ratio",
    "high_frequency_ratio",
    "wavelet_entropy",
    "max_abs_coefficient",
    "energy_center",
    "energy_spread",
)


def raw_sequence(signal: Any) -> np.ndarray:
    """Return a finite sequence with the final axis as time."""

    array = _as_signal(signal)
    output = _last_axis_as_sequence(array).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="raw_sequence")
    return output


def stat28(signal: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    """Return the shared 28 statistical features along the final axis."""

    if epsilon <= 0.0 or not np.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    array = _as_signal(signal).astype(np.float64, copy=False)
    length = array.shape[-1]
    flat = array.reshape((-1, length))

    diffs = np.diff(flat, axis=1)
    mean = np.mean(flat, axis=1)
    std = np.std(flat, axis=1)
    minimum = np.min(flat, axis=1)
    maximum = np.max(flat, axis=1)
    value_range = maximum - minimum
    slope = np.zeros(flat.shape[0], dtype=np.float64)
    if length > 1:
        slope = (flat[:, -1] - flat[:, 0]) / float(length - 1)
    centered = flat - mean[:, np.newaxis]
    centered_second = np.mean(np.square(centered), axis=1)
    skew = _safe_div_vector(np.mean(centered**3, axis=1), centered_second**1.5, float(epsilon))
    kurtosis = _safe_div_vector(np.mean(centered**4, axis=1), centered_second**2, float(epsilon))

    spectrum = np.fft.rfft(flat, axis=1)
    magnitude = np.abs(spectrum)
    power = np.square(magnitude)
    freqs = np.fft.rfftfreq(length)
    total_power = np.sum(power, axis=1)
    total_magnitude = np.sum(magnitude, axis=1)
    probability = power / np.maximum(total_power[:, np.newaxis], float(epsilon))
    centroid = _safe_div_vector(np.sum(freqs[np.newaxis, :] * power, axis=1), total_power, float(epsilon))
    spread = np.sqrt(
        np.maximum(
            _safe_div_vector(
                np.sum(np.square(freqs[np.newaxis, :] - centroid[:, np.newaxis]) * power, axis=1),
                total_power,
                float(epsilon),
            ),
            0.0,
        )
    )
    positive_magnitude = np.maximum(magnitude, float(epsilon))
    geometric_mean = np.exp(np.mean(np.log(positive_magnitude), axis=1))
    arithmetic_mean = total_magnitude / float(magnitude.shape[1])
    high_power = np.sum(power[:, 1:], axis=1)
    mean_abs_diff = (
        np.mean(np.abs(diffs), axis=1)
        if diffs.shape[1]
        else np.zeros(flat.shape[0], dtype=np.float64)
    )

    features = np.column_stack(
        [
            mean,
            std,
            minimum,
            maximum,
            flat[:, 0],
            flat[:, -1],
            value_range,
            slope,
            np.mean(np.abs(flat), axis=1),
            mean_abs_diff,
            np.median(flat, axis=1),
            np.percentile(flat, 25.0, axis=1),
            np.percentile(flat, 75.0, axis=1),
            _safe_div_vector(std, np.abs(mean), float(epsilon)),
            skew,
            kurtosis,
            magnitude[:, 0],
            magnitude[:, 1] if magnitude.shape[1] > 1 else np.zeros(flat.shape[0], dtype=np.float64),
            magnitude[:, 2] if magnitude.shape[1] > 2 else np.zeros(flat.shape[0], dtype=np.float64),
            total_power,
            power[:, 0],
            high_power,
            _safe_div_vector(high_power, total_power, float(epsilon)),
            centroid,
            spread,
            _entropy_matrix(probability, epsilon=float(epsilon)),
            np.argmax(power, axis=1).astype(np.float64),
            _safe_div_vector(geometric_mean, arithmetic_mean, float(epsilon)),
        ]
    )
    output = features.reshape((*array.shape[:-1], len(STAT28_NAMES))).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="stat28")
    return output


def stat28_features(signal: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    return stat28(signal, epsilon=epsilon)


def wavelet_summary_static_pooled(signal: Any, spec: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    """Return pooled wavelet energy/morphology summaries across channels."""

    if epsilon <= 0.0 or not np.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    per_channel = _wavelet_summary_per_channel(signal, spec, epsilon=float(epsilon))
    pooled = _pool_channel_features(per_channel)
    output = pooled.astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="wavelet_summary_static_pooled")
    return output


def wavelet_summary_static_pooled_names(spec: Any | None = None) -> tuple[str, ...]:
    return tuple(
        f"{pool}_{name}"
        for pool in SUMMARY_POOL_NAMES
        for name in _wavelet_summary_names_for_spec(spec)
    )


def window_stat28_sequence(signal: Any, spec: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    """Return one stat28 vector per configured window."""

    windowed = _windowed_transformed_signal_for_spec(signal, spec)
    per_window = stat28(windowed, epsilon=epsilon).astype(np.float32, copy=False)
    output = per_window.reshape((per_window.shape[0], -1)).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="window_stat28_sequence")
    return output


def window_stat28_static_flat(signal: Any, spec: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    """Flatten per-window stat28 vectors into one static feature vector."""

    output = window_stat28_sequence(signal, spec, epsilon=epsilon).reshape(-1).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="window_stat28_static_flat")
    return output


def window_stat28_static_pooled(signal: Any, spec: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    """Pool per-window, per-channel stat28 vectors into one static vector."""

    windowed = _windowed_transformed_signal_for_spec(signal, spec)
    per_window = stat28(windowed, epsilon=epsilon).astype(np.float64, copy=False)
    per_window = per_window.reshape((per_window.shape[0], -1, len(STAT28_NAMES)))
    pooled = [_pool_channel_features(window_features) for window_features in per_window]
    output = np.concatenate(pooled, axis=0).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="window_stat28_static_pooled")
    return output


def features_for_protocol(signal: Any, spec: Any, *, epsilon: float = EPSILON) -> np.ndarray:
    protocol = _spec_text(spec, "feature_protocol", "")
    values = _windowed_signal_for_spec(signal, spec)
    if protocol == "raw_sequence":
        return raw_sequence_for_spec(signal, spec)
    if protocol == "stat28":
        return stat28(values, epsilon=epsilon).reshape(-1).astype(np.float32, copy=False)
    if protocol == "wavelet_summary_static_pooled":
        return wavelet_summary_static_pooled(values, spec, epsilon=epsilon)
    if protocol == "window_stat28_sequence":
        return window_stat28_sequence(signal, spec, epsilon=epsilon)
    if protocol == "window_stat28_static_flat":
        return window_stat28_static_flat(signal, spec, epsilon=epsilon)
    if protocol == "window_stat28_static_pooled":
        return window_stat28_static_pooled(signal, spec, epsilon=epsilon)
    raise ValueError(f"unsupported feature_protocol: {protocol!r}")


def _windowed_signal_for_spec(signal: Any, spec: Any) -> np.ndarray:
    strategy = _spec_text(spec, "window_strategy", "full")
    if strategy == "full":
        return _as_signal(signal)
    from .common_windowing import coerce_window_spec, window_signal

    return window_signal(signal, coerce_window_spec(spec))


def _windowed_transformed_signal_for_spec(signal: Any, spec: Any) -> np.ndarray:
    from .common_windowing import coerce_window_spec, window_signal

    windowed = window_signal(signal, coerce_window_spec(spec))
    transformed = apply_wavelet(windowed, spec)
    _raise_if_non_finite(transformed, name="windowed_transformed_signal")
    return transformed.astype(np.float32, copy=False)


def raw_sequence_for_spec(signal: Any, spec: Any) -> np.ndarray:
    """Return a raw sequence while preserving the layer-depth time axis."""

    array = _as_signal(signal)
    transform = _spec_text(spec, "transform", "none").lower()
    if transform == "none":
        return raw_sequence(array)
    if transform == "swt":
        return _swt_time_preserving_sequence(array, spec)
    if transform == "cwt":
        return _cwt_time_preserving_sequence(array, spec)
    raise WaveletConfigError(
        f"raw_sequence with transform={transform!r} cannot preserve the original length axis; "
        "use transform='none', 'swt', or 'cwt'"
    )


def _swt_time_preserving_sequence(signal: np.ndarray, spec: Any) -> np.ndarray:
    pywt = _import_pywt()
    array = _as_signal(signal).astype(np.float64, copy=False)
    flat = array.reshape((-1, array.shape[-1]))
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    threshold = _threshold_for_transform(spec, "swt")
    max_level = _swt_max_level(flat.shape[-1])
    if level > max_level:
        from .common_wavelet import SWTLevelInfeasibleError

        raise SWTLevelInfeasibleError(
            f"SWT level {level} is not feasible for length {flat.shape[-1]}; max level is {max_level}"
        )
    coeff_pairs = pywt.swt(flat, wavelet_name, level=level, axis=-1, trim_approx=False)
    raw_details = [np.asarray(detail, dtype=np.float64) for _approx, detail in coeff_pairs]
    details = _threshold_detail_coeffs(raw_details, threshold, sample_length=flat.shape[-1])
    components: list[np.ndarray] = []
    for (approx, _detail), detail in zip(coeff_pairs, details, strict=True):
        components.append(np.asarray(approx, dtype=np.float64))
        components.append(np.asarray(detail, dtype=np.float64))
    stacked = np.stack(components, axis=1)
    output = np.moveaxis(stacked, -1, 0).reshape((flat.shape[-1], -1)).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="swt_raw_sequence")
    return output


def _cwt_time_preserving_sequence(signal: np.ndarray, spec: Any) -> np.ndarray:
    _threshold_for_transform(spec, "cwt")
    pywt = _import_pywt()
    array = _as_signal(signal).astype(np.float64, copy=False)
    flat = array.reshape((-1, array.shape[-1]))
    wavelet_name = _required_wavelet(spec)
    scales = _spec_scales(spec)
    coeffs, _freqs = pywt.cwt(flat, scales, wavelet_name, axis=-1)
    stacked = np.moveaxis(np.asarray(coeffs, dtype=np.float64), 0, 1)
    output = np.moveaxis(stacked, -1, 0).reshape((flat.shape[-1], -1)).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="cwt_raw_sequence")
    return output


def _last_axis_as_sequence(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape((-1, 1))
    moved = np.moveaxis(array, -1, 0)
    return moved.reshape((moved.shape[0], -1))


def _as_signal(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError("signal must have at least one dimension")
    if array.shape[-1] <= 0:
        raise ValueError("signal length axis must be non-empty")
    _raise_if_non_finite(array, name="signal")
    return array


def _wavelet_summary_per_channel(signal: Any, spec: Any, *, epsilon: float) -> np.ndarray:
    array = _as_signal(signal).astype(np.float64, copy=False)
    channels = array.reshape((-1, array.shape[-1]))
    transform = _spec_text(spec, "transform", "none").lower()
    if transform == "none":
        return _summaries_from_coeff_sets(
            approximation=channels,
            details=[],
            all_coefficients=[channels],
            epsilon=epsilon,
        )
    if transform == "dwt":
        return _dwt_summary_per_channel(channels, spec, epsilon=epsilon)
    if transform == "swt":
        return _swt_summary_per_channel(channels, spec, epsilon=epsilon)
    if transform == "wpt":
        return _wpt_summary_per_channel(channels, spec, epsilon=epsilon)
    if transform == "cwt":
        return _cwt_summary_per_channel(channels, spec, epsilon=epsilon)
    raise WaveletConfigError(f"unsupported wavelet transform: {transform!r}")


def _dwt_summary_per_channel(channels: np.ndarray, spec: Any, *, epsilon: float) -> np.ndarray:
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    threshold = _threshold_for_transform(spec, "dwt")
    mode = _spec_text(spec, "mode", "symmetric")
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(channels.shape[-1], wavelet.dec_len)
    if level > max_level:
        raise WaveletConfigError(
            f"DWT level {level} is not feasible for length {channels.shape[-1]} "
            f"and wavelet {wavelet_name!r}; max level is {max_level}"
        )
    coeffs = pywt.wavedec(channels, wavelet_name, mode=mode, level=level, axis=-1)
    coeffs = _threshold_dwt_coeffs(coeffs, threshold, sample_length=channels.shape[-1])
    approximation = np.asarray(coeffs[0], dtype=np.float64)
    details = [np.asarray(coeff, dtype=np.float64) for coeff in coeffs[1:]]
    return _summaries_from_coeff_sets(
        approximation=approximation,
        details=details,
        all_coefficients=[approximation, *details],
        epsilon=epsilon,
    )


def _swt_summary_per_channel(channels: np.ndarray, spec: Any, *, epsilon: float) -> np.ndarray:
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    threshold = _threshold_for_transform(spec, "swt")
    max_level = _swt_max_level(channels.shape[-1])
    if level > max_level:
        from .common_wavelet import SWTLevelInfeasibleError

        raise SWTLevelInfeasibleError(
            f"SWT level {level} is not feasible for length {channels.shape[-1]}; max level is {max_level}"
        )
    coeff_pairs = pywt.swt(channels, wavelet_name, level=level, axis=-1, trim_approx=False)
    raw_details = [np.asarray(detail, dtype=np.float64) for _approx, detail in coeff_pairs]
    details = _threshold_detail_coeffs(raw_details, threshold, sample_length=channels.shape[-1])
    approximation = np.asarray(coeff_pairs[0][0], dtype=np.float64)
    return _summaries_from_coeff_sets(
        approximation=approximation,
        details=details,
        all_coefficients=[approximation, *details],
        epsilon=epsilon,
    )


def _wpt_summary_per_channel(channels: np.ndarray, spec: Any, *, epsilon: float) -> np.ndarray:
    _threshold_for_transform(spec, "wpt")
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    mode = _spec_text(spec, "mode", "symmetric")
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(channels.shape[-1], wavelet.dec_len)
    if level > max_level:
        raise WaveletConfigError(
            f"WPT level {level} is not feasible for length {channels.shape[-1]} "
            f"and wavelet {wavelet_name!r}; max level is {max_level}"
        )
    if level not in {1, 2}:
        raise WaveletConfigError(
            f"batched WPT summary supports level 1 or 2 for this experiment; got level {level}"
        )

    approx, detail = pywt.dwt(channels, wavelet_name, mode=mode, axis=-1)
    if level == 1:
        packets = [
            np.asarray(approx, dtype=np.float64),
            np.asarray(detail, dtype=np.float64),
        ]
    else:
        aa, ad = pywt.dwt(approx, wavelet_name, mode=mode, axis=-1)
        da, dd = pywt.dwt(detail, wavelet_name, mode=mode, axis=-1)
        packets = [
            np.asarray(aa, dtype=np.float64),
            np.asarray(ad, dtype=np.float64),
            np.asarray(dd, dtype=np.float64),
            np.asarray(da, dtype=np.float64),
        ]
    return _summaries_from_coeff_sets(
        approximation=packets[0],
        details=packets[1:],
        all_coefficients=packets,
        epsilon=epsilon,
    )


def _cwt_summary_per_channel(channels: np.ndarray, spec: Any, *, epsilon: float) -> np.ndarray:
    _threshold_for_transform(spec, "cwt")
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    scales = _spec_scales(spec)
    coeffs, _freqs = pywt.cwt(channels, scales, wavelet_name, axis=-1)
    coeff_cube = np.moveaxis(np.asarray(coeffs, dtype=np.float64), 0, 1)
    approximation = coeff_cube[:, -1, :]
    details = [coeff_cube[:, index, :] for index in range(max(coeff_cube.shape[1] - 1, 0))]
    return _summaries_from_coeff_sets(
        approximation=approximation,
        details=details,
        all_coefficients=[coeff_cube[:, index, :] for index in range(coeff_cube.shape[1])],
        epsilon=epsilon,
    )


def _summaries_from_coeff_sets(
    *,
    approximation: np.ndarray,
    details: list[np.ndarray],
    all_coefficients: list[np.ndarray],
    epsilon: float,
) -> np.ndarray:
    approximation = np.asarray(approximation, dtype=np.float64)
    if approximation.ndim != 2:
        raise ValueError("approximation coefficients must have shape (channels, coeffs)")
    detail_arrays = [np.asarray(detail, dtype=np.float64) for detail in details]
    all_arrays = [np.asarray(coeff, dtype=np.float64) for coeff in all_coefficients]
    for coeff in [approximation, *detail_arrays, *all_arrays]:
        if coeff.ndim != 2 or coeff.shape[0] != approximation.shape[0]:
            raise ValueError("coefficient arrays must share shape (channels, coeffs)")

    approximation_energy = _energy_by_row(approximation)
    detail_energies = [_energy_by_row(detail) for detail in detail_arrays]
    total_detail_energy = (
        np.sum(np.stack(detail_energies, axis=0), axis=0)
        if detail_energies
        else np.zeros(approximation.shape[0], dtype=np.float64)
    )
    total_energy = approximation_energy + total_detail_energy
    high_frequency_energy = detail_energies[-1] if detail_energies else np.zeros_like(total_energy)
    morphology_coeffs = detail_arrays if detail_arrays else all_arrays
    energy_center, energy_spread = _energy_center_and_spread(morphology_coeffs, epsilon=epsilon)
    features = np.column_stack(
        [
            approximation_energy,
            *detail_energies,
            total_detail_energy,
            _safe_div_vector(total_detail_energy, approximation_energy, epsilon),
            _safe_div_vector(high_frequency_energy, total_energy, epsilon),
            _coefficient_entropy(all_arrays, epsilon=epsilon),
            _max_abs_by_row(all_arrays),
            energy_center,
            energy_spread,
        ]
    )
    _raise_if_non_finite(features, name="wavelet_channel_summary")
    return features.astype(np.float64, copy=False)


def _energy_by_row(values: np.ndarray) -> np.ndarray:
    return np.sum(np.square(np.asarray(values, dtype=np.float64)), axis=1)


def _coefficient_entropy(coefficients: list[np.ndarray], *, epsilon: float) -> np.ndarray:
    energy = _concatenate_rowwise([np.square(coeff) for coeff in coefficients])
    total = np.sum(energy, axis=1, keepdims=True)
    probability = energy / np.maximum(total, epsilon)
    return _entropy_matrix(probability, epsilon=epsilon)


def _max_abs_by_row(coefficients: list[np.ndarray]) -> np.ndarray:
    absolute = _concatenate_rowwise([np.abs(coeff) for coeff in coefficients])
    return np.max(absolute, axis=1)


def _energy_center_and_spread(
    coefficients: list[np.ndarray],
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    energy = _concatenate_rowwise([np.square(coeff) for coeff in coefficients])
    total = np.sum(energy, axis=1)
    positions = np.arange(energy.shape[1], dtype=np.float64)
    center = _safe_div_vector(np.sum(energy * positions[np.newaxis, :], axis=1), total, epsilon)
    spread = np.sqrt(
        np.maximum(
            _safe_div_vector(
                np.sum(energy * np.square(positions[np.newaxis, :] - center[:, np.newaxis]), axis=1),
                total,
                epsilon,
            ),
            0.0,
        )
    )
    zero_mask = total <= epsilon
    center[zero_mask] = 0.0
    spread[zero_mask] = 0.0
    return center, spread


def _concatenate_rowwise(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("at least one coefficient array is required")
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    row_count = arrays[0].shape[0]
    if any(array.ndim != 2 or array.shape[0] != row_count for array in arrays):
        raise ValueError("coefficient arrays must share shape (channels, coeffs)")
    return np.concatenate(arrays, axis=1)


def _pool_channel_features(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("channel features must have shape (channels, features)")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("channel features must be non-empty")
    _raise_if_non_finite(matrix, name="channel_features")
    return np.concatenate(
        [
            np.mean(matrix, axis=0),
            np.std(matrix, axis=0),
            np.max(matrix, axis=0),
            np.min(matrix, axis=0),
            np.median(matrix, axis=0),
            np.percentile(matrix, 90.0, axis=0),
        ],
        axis=0,
    )


def _wavelet_summary_names_for_spec(spec: Any | None) -> tuple[str, ...]:
    detail_count = _detail_count_for_spec(spec)
    return (
        "approximation_energy",
        *(f"detail_energy_{index + 1}" for index in range(detail_count)),
        *WAVELET_SUMMARY_BASE_NAMES[1:],
    )


def _detail_count_for_spec(spec: Any | None) -> int:
    if spec is None:
        return 0
    transform = _spec_text(spec, "transform", "none").lower()
    if transform == "none":
        return 0
    if transform in {"dwt", "swt"}:
        return _required_level(spec)
    if transform == "wpt":
        return (2 ** _required_level(spec)) - 1
    if transform == "cwt":
        return max(int(_spec_scales(spec).shape[0]) - 1, 0)
    raise WaveletConfigError(f"unsupported wavelet transform: {transform!r}")


def _safe_div_vector(numerator: np.ndarray, denominator: np.ndarray, epsilon: float) -> np.ndarray:
    numer = np.asarray(numerator, dtype=np.float64)
    denom = np.asarray(denominator, dtype=np.float64)
    adjusted = denom.copy()
    mask = np.abs(adjusted) <= epsilon
    adjusted[mask & (adjusted >= 0.0)] = epsilon
    adjusted[mask & (adjusted < 0.0)] = -epsilon
    return numer / adjusted


def _entropy_matrix(probability: np.ndarray, *, epsilon: float) -> np.ndarray:
    safe = np.maximum(np.asarray(probability, dtype=np.float64), epsilon)
    return -np.sum(safe * np.log(safe), axis=1)


def _spec_text(spec: Any, key: str, default: str) -> str:
    if isinstance(spec, dict):
        value = spec.get(key, default)
    else:
        value = getattr(spec, key, default)
    return str(value).strip() if value is not None else default


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


__all__ = [
    "EPSILON",
    "STAT28_NAMES",
    "features_for_protocol",
    "raw_sequence",
    "raw_sequence_for_spec",
    "stat28",
    "stat28_features",
    "wavelet_summary_static_pooled",
    "wavelet_summary_static_pooled_names",
    "window_stat28_sequence",
    "window_stat28_static_flat",
    "window_stat28_static_pooled",
]
