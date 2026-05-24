"""Common last-axis windowing strategies for paired-wavelet v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SUPPORTED_WINDOW_STRATEGIES = ("full", "non_overlapping", "sliding")


@dataclass(frozen=True, slots=True)
class WindowSpec:
    strategy: str = "full"
    size: int | None = None
    stride: int | None = None
    drop_remainder: bool = True

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip()
        if strategy not in SUPPORTED_WINDOW_STRATEGIES:
            raise ValueError(f"unsupported window strategy: {strategy!r}")
        object.__setattr__(self, "strategy", strategy)
        if strategy == "full":
            if self.size is not None or self.stride is not None:
                raise ValueError("full windows do not accept size or stride")
            return
        size = _positive_int(self.size, "size")
        if strategy == "non_overlapping":
            stride = size if self.stride is None else _positive_int(self.stride, "stride")
            if stride != size:
                raise ValueError("non_overlapping windows require stride == size")
        else:
            stride = _positive_int(self.stride, "stride")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "drop_remainder", bool(self.drop_remainder))


def window_signal(signal: Any, spec: WindowSpec | None = None) -> np.ndarray:
    """Stack windows from the final length axis.

    The returned shape is ``(num_windows, *signal.shape[:-1], window_length)``.
    """

    array = _as_signal(signal)
    window_spec = spec or WindowSpec()
    slices = window_slices(array.shape[-1], window_spec)
    windows = [array[..., item] for item in slices]
    output = np.stack(windows, axis=0).astype(np.float32, copy=False)
    _raise_if_non_finite(output, name="windowed_signal")
    return output


def window_slices(length: int, spec: WindowSpec | None = None) -> tuple[slice, ...]:
    window_spec = spec or WindowSpec()
    value = int(length)
    if value <= 0:
        raise ValueError("length must be positive")
    if window_spec.strategy == "full":
        return (slice(0, value),)

    size = int(window_spec.size)
    stride = int(window_spec.stride)
    if size > value:
        raise ValueError(f"window size {size} exceeds length {value}")
    starts = list(range(0, value - size + 1, stride))
    if not starts:
        raise ValueError("window strategy produced no windows")
    if not window_spec.drop_remainder and starts[-1] + size != value:
        raise ValueError("drop_remainder=False is unsupported when the final window would be ragged")
    return tuple(slice(start, start + size) for start in starts)


def coerce_window_spec(value: Any) -> WindowSpec:
    if value is None:
        return WindowSpec()
    if isinstance(value, WindowSpec):
        return value
    strategy = getattr(value, "window_strategy", None)
    if strategy is None and isinstance(value, dict):
        strategy = value.get("window_strategy", value.get("strategy", "full"))
    size = getattr(value, "window_size", None)
    if size is None:
        size = getattr(value, "size", None)
    stride = getattr(value, "stride", None)
    if isinstance(value, dict):
        size = value.get("window_size", value.get("size", size))
        stride = value.get("stride", stride)
    return WindowSpec(strategy=strategy or "full", size=size, stride=stride)


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


def _positive_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


__all__ = [
    "SUPPORTED_WINDOW_STRATEGIES",
    "WindowSpec",
    "coerce_window_spec",
    "window_signal",
    "window_slices",
]
