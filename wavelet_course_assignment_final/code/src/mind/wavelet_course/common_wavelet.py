"""Common wavelet transforms for paired-wavelet v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SUPPORTED_TRANSFORMS = ("none", "dwt", "swt", "wpt", "cwt")
DWT_SWT_THRESHOLDS = ("none", "universal_soft", "universal_hard", "sure_soft")
NOT_APPLICABLE_THRESHOLD = "not_applicable"
SUPPORTED_THRESHOLDS = (*DWT_SWT_THRESHOLDS, NOT_APPLICABLE_THRESHOLD)
DEFAULT_CWT_SCALES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0)
MAD_NORMAL_SCALE = 0.6744897501960817


class WaveletConfigError(ValueError):
    """Raised when a wavelet configuration is invalid or unsupported."""


class SWTLevelInfeasibleError(WaveletConfigError):
    """Raised when SWT level cannot be applied to the signal length."""


def apply_wavelet(signal: Any, spec: Any) -> np.ndarray:
    """Apply a transform along the final length axis only.

    The output keeps the same leading dimensions as ``signal``. The final axis
    is replaced by the concatenated transform coefficients.
    """

    array = _as_signal(signal)
    transform = _spec_text(spec, "transform", "none").lower()
    if transform not in SUPPORTED_TRANSFORMS:
        raise WaveletConfigError(f"unsupported wavelet transform: {transform!r}")
    if transform == "none":
        _reject_unexpected_none_config(spec)
        return array.astype(np.float32, copy=True)
    threshold = _threshold_for_transform(spec, transform)
    if transform == "dwt":
        output = _apply_dwt(array, spec, threshold=threshold)
    elif transform == "swt":
        output = _apply_swt(array, spec, threshold=threshold)
    elif transform == "wpt":
        output = _apply_wpt(array, spec)
    elif transform == "cwt":
        output = _apply_cwt(array, spec)
    else:
        raise WaveletConfigError(f"unsupported wavelet transform: {transform!r}")
    _raise_if_non_finite(output, name=f"{transform}_coefficients")
    return output.astype(np.float32, copy=False)


def _apply_dwt(array: np.ndarray, spec: Any, *, threshold: str) -> np.ndarray:
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    mode = _spec_text(spec, "mode", "symmetric")
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(array.shape[-1], wavelet.dec_len)
    if level > max_level:
        raise WaveletConfigError(
            f"DWT level {level} is not feasible for length {array.shape[-1]} "
            f"and wavelet {wavelet_name!r}; max level is {max_level}"
        )
    coeffs = pywt.wavedec(array.astype(np.float64), wavelet_name, mode=mode, level=level, axis=-1)
    coeffs = _threshold_dwt_coeffs(coeffs, threshold, sample_length=array.shape[-1])
    return np.concatenate([np.asarray(coeff, dtype=np.float64) for coeff in coeffs], axis=-1)


def _apply_swt(array: np.ndarray, spec: Any, *, threshold: str) -> np.ndarray:
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    try:
        pywt.Wavelet(wavelet_name)
    except Exception as exc:
        raise WaveletConfigError(f"invalid SWT wavelet configuration: {wavelet_name!r}") from exc
    max_level = _swt_max_level(array.shape[-1])
    if level > max_level:
        raise SWTLevelInfeasibleError(
            f"SWT level {level} is not feasible for length {array.shape[-1]}; max level is {max_level}"
        )
    coeff_pairs = pywt.swt(
        array.astype(np.float64),
        wavelet_name,
        level=level,
        axis=-1,
        trim_approx=False,
    )
    coeffs: list[np.ndarray] = []
    detail_coeffs = [np.asarray(detail, dtype=np.float64) for _approx, detail in coeff_pairs]
    thresholded_details = _threshold_detail_coeffs(
        detail_coeffs,
        threshold,
        sample_length=array.shape[-1],
    )
    for (approx, _detail), detail in zip(coeff_pairs, thresholded_details, strict=True):
        coeffs.append(np.asarray(approx, dtype=np.float64))
        coeffs.append(np.asarray(detail, dtype=np.float64))
    return np.concatenate(coeffs, axis=-1)


def _apply_wpt(array: np.ndarray, spec: Any) -> np.ndarray:
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    level = _required_level(spec)
    mode = _spec_text(spec, "mode", "symmetric")
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(array.shape[-1], wavelet.dec_len)
    if level > max_level:
        raise WaveletConfigError(
            f"WPT level {level} is not feasible for length {array.shape[-1]} "
            f"and wavelet {wavelet_name!r}; max level is {max_level}"
        )

    flat, leading = _flatten_leading(array)
    rows: list[np.ndarray] = []
    for row in flat:
        packet = pywt.WaveletPacket(data=row.astype(np.float64), wavelet=wavelet_name, mode=mode, maxlevel=level)
        nodes = packet.get_level(level, order="freq")
        if not nodes:
            raise WaveletConfigError(f"WPT produced no nodes at level {level}")
        rows.append(np.concatenate([np.asarray(node.data, dtype=np.float64) for node in nodes], axis=0))
    return _unflatten_leading(np.stack(rows, axis=0), leading)


def _apply_cwt(array: np.ndarray, spec: Any) -> np.ndarray:
    pywt = _import_pywt()
    wavelet_name = _required_wavelet(spec)
    scales = _spec_scales(spec)
    flat, leading = _flatten_leading(array)
    rows: list[np.ndarray] = []
    for row in flat:
        coeffs, _freqs = pywt.cwt(row.astype(np.float64), scales, wavelet_name)
        rows.append(np.asarray(coeffs, dtype=np.float64).reshape(-1))
    return _unflatten_leading(np.stack(rows, axis=0), leading)


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


def _flatten_leading(array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    leading = tuple(array.shape[:-1])
    return array.reshape((-1, array.shape[-1])), leading


def _unflatten_leading(flat: np.ndarray, leading: tuple[int, ...]) -> np.ndarray:
    if flat.ndim != 2:
        raise ValueError("flat coefficients must have shape (rows, coeffs)")
    return flat.reshape((*leading, flat.shape[-1]))


def _swt_max_level(length: int) -> int:
    value = int(length)
    if value <= 0:
        return 0
    level = 0
    while value % 2 == 0:
        level += 1
        value //= 2
    return level


def _import_pywt() -> Any:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("paired wavelet transforms require pywt") from exc
    return pywt


def _required_wavelet(spec: Any) -> str:
    wavelet = _spec_text(spec, "wavelet", "")
    if not wavelet or wavelet == "none":
        raise WaveletConfigError("wavelet is required for wavelet transforms")
    return wavelet


def _required_level(spec: Any) -> int:
    value = _spec_value(spec, "level", None)
    if value is None:
        raise WaveletConfigError("level is required for this wavelet transform")
    level = int(value)
    if level <= 0:
        raise WaveletConfigError("level must be positive")
    return level


def _spec_scales(spec: Any) -> np.ndarray:
    values = _spec_value(spec, "cwt_scales", None)
    if values is None:
        values = _spec_value(spec, "scales", DEFAULT_CWT_SCALES)
    scales = np.asarray(list(values), dtype=np.float64)
    if scales.ndim != 1 or scales.shape[0] == 0:
        raise WaveletConfigError("cwt_scales must be a non-empty 1D sequence")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise WaveletConfigError("cwt_scales must contain finite positive values")
    return scales


def _threshold_for_transform(spec: Any, transform: str) -> str:
    default = NOT_APPLICABLE_THRESHOLD if transform in {"cwt", "wpt"} else "none"
    threshold = _spec_text(spec, "threshold", default).lower()
    if threshold not in SUPPORTED_THRESHOLDS:
        raise WaveletConfigError(f"unsupported threshold: {threshold!r}")
    if transform in {"dwt", "swt"}:
        if threshold not in DWT_SWT_THRESHOLDS:
            raise WaveletConfigError(
                f"transform={transform!r} only accepts thresholds {DWT_SWT_THRESHOLDS}"
            )
        return threshold
    if transform in {"cwt", "wpt"}:
        if threshold != NOT_APPLICABLE_THRESHOLD:
            raise WaveletConfigError(
                f"transform={transform!r} requires threshold={NOT_APPLICABLE_THRESHOLD!r}"
            )
        return threshold
    raise WaveletConfigError(f"unsupported wavelet transform: {transform!r}")


def _threshold_dwt_coeffs(
    coeffs: Sequence[np.ndarray],
    threshold: str,
    *,
    sample_length: int,
) -> list[np.ndarray]:
    if threshold == "none":
        return [np.asarray(coeff, dtype=np.float64) for coeff in coeffs]
    if len(coeffs) < 2:
        return [np.asarray(coeff, dtype=np.float64) for coeff in coeffs]
    details = [np.asarray(coeff, dtype=np.float64) for coeff in coeffs[1:]]
    return [
        np.asarray(coeffs[0], dtype=np.float64),
        *_threshold_detail_coeffs(details, threshold, sample_length=sample_length),
    ]


def _threshold_detail_coeffs(
    details: Sequence[np.ndarray],
    threshold: str,
    *,
    sample_length: int,
) -> list[np.ndarray]:
    if threshold == "none":
        return [np.asarray(detail, dtype=np.float64) for detail in details]
    if not details:
        return []
    sigma = _detail_sigma(np.asarray(details[-1], dtype=np.float64))
    universal = sigma * np.sqrt(2.0 * np.log(float(max(sample_length, 2))))
    return [
        _threshold_detail(np.asarray(detail, dtype=np.float64), threshold, sigma=sigma, universal=universal)
        for detail in details
    ]


def _threshold_detail(
    detail: np.ndarray,
    threshold: str,
    *,
    sigma: np.ndarray,
    universal: np.ndarray,
) -> np.ndarray:
    if threshold == "universal_soft":
        return _soft_threshold(detail, universal)
    if threshold == "universal_hard":
        return _hard_threshold(detail, universal)
    if threshold == "sure_soft":
        return _soft_threshold(detail, _sure_soft_threshold(detail, sigma=sigma, universal=universal))
    raise WaveletConfigError(f"unsupported threshold: {threshold!r}")


def _detail_sigma(detail: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(detail), axis=-1, keepdims=True)
    return mad / MAD_NORMAL_SCALE


def _soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    return np.sign(values) * np.maximum(magnitude - threshold, 0.0)


def _hard_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values) > threshold, values, 0.0)


def _sure_soft_threshold(detail: np.ndarray, *, sigma: np.ndarray, universal: np.ndarray) -> np.ndarray:
    flat_detail, leading = _flatten_leading(np.asarray(detail, dtype=np.float64))
    sigma_flat = np.asarray(sigma, dtype=np.float64).reshape(-1)
    universal_flat = np.asarray(universal, dtype=np.float64).reshape(-1)
    thresholds = np.zeros((flat_detail.shape[0], 1), dtype=np.float64)
    for row_index, row in enumerate(flat_detail):
        row_sigma = float(sigma_flat[row_index])
        if row_sigma <= 0.0 or not np.isfinite(row_sigma):
            continue
        squared = np.sort(np.square(np.abs(row) / row_sigma))
        count = squared.shape[0]
        ranks = np.arange(1, count + 1, dtype=np.float64)
        risks = count - (2.0 * ranks) + np.cumsum(squared) + (count - ranks) * squared
        sure = np.sqrt(squared[int(np.argmin(risks))]) * row_sigma
        thresholds[row_index, 0] = min(float(universal_flat[row_index]), float(sure))
    return thresholds.reshape((*leading, 1))


def _reject_unexpected_none_config(spec: Any) -> None:
    wavelet = _spec_value(spec, "wavelet", None)
    level = _spec_value(spec, "level", None)
    threshold = _spec_text(spec, "threshold", "none").lower()
    if threshold not in SUPPORTED_THRESHOLDS:
        raise WaveletConfigError(f"unsupported threshold: {threshold!r}")
    if wavelet not in {None, "", "none"}:
        raise WaveletConfigError("transform='none' does not accept wavelet")
    if level is not None:
        raise WaveletConfigError("transform='none' does not accept level")
    if threshold != "none":
        raise WaveletConfigError("transform='none' only accepts threshold='none'")


def _spec_text(spec: Any, key: str, default: str) -> str:
    value = _spec_value(spec, key, default)
    return str(value).strip() if value is not None else default


def _spec_value(spec: Any, key: str, default: Any) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


__all__ = [
    "DEFAULT_CWT_SCALES",
    "DWT_SWT_THRESHOLDS",
    "NOT_APPLICABLE_THRESHOLD",
    "SUPPORTED_THRESHOLDS",
    "SUPPORTED_TRANSFORMS",
    "SWTLevelInfeasibleError",
    "WaveletConfigError",
    "apply_wavelet",
]
