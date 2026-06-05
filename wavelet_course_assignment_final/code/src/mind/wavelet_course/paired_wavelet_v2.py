"""Compatibility facade for the older ``paired_wavelet_v2.*`` API.

The v2 implementation now lives in the common paired modules. This file keeps
legacy imports working without adding another package directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common_classifiers import (
    XGBOOST_NOT_INSTALLED,
    train_static_classifier,
    xgboost_missing_failure_rows,
)
from .common_wavelet import SWTLevelInfeasibleError, WaveletConfigError
from .common_windowing import WindowSpec, window_signal
from .paired_reporting import write_paired_metrics_csv
from .signal_builders import (
    REQUIRED_OURS_TRACE_NAMES,
    ours_semantic_trace_signal,
    teacher_hidden_dim_signal,
)


PairedWaveletConfigError = WaveletConfigError


@dataclass(frozen=True)
class OursSignalResult:
    signal: np.ndarray
    trace_names: list[str]


@dataclass(frozen=True)
class PairedTrainingGridResult:
    status: str
    rows: list[dict[str, Any]]
    result: Any | None = None


def build_paired_grid(
    *,
    teacher_configs: Sequence[Mapping[str, Any]],
    ours_configs: Sequence[Mapping[str, Any]],
    shared_fields: Mapping[str, Any] | None = None,
    classifier: str,
) -> list[dict[str, Any]]:
    """Build the legacy cross-product paired grid rows."""

    shared = dict(shared_fields or {})
    rows: list[dict[str, Any]] = []
    for teacher in teacher_configs:
        teacher_name = str(teacher["name"])
        for ours in ours_configs:
            ours_name = str(ours["name"])
            rows.append(
                {
                    **shared,
                    "paired_config_name": f"{teacher_name}__{ours_name}__{classifier}",
                    "teacher_config_name": teacher_name,
                    "ours_config_name": ours_name,
                    "method_family": "paired_wavelet_v2",
                    "classifier": classifier,
                    "teacher_wavelet": teacher.get("wavelet", ""),
                    "teacher_level": teacher.get("level", ""),
                    "ours_wavelet": ours.get("wavelet", ""),
                    "ours_level": ours.get("level", ""),
                }
            )
    return rows


def build_teacher_signal(entry: Mapping[str, Any]) -> np.ndarray:
    return teacher_hidden_dim_signal(entry["layer_vectors"])


def build_ours_signal(
    entry: Mapping[str, Any],
    *,
    yes_token_id: int,
    no_token_id: int,
) -> OursSignalResult:
    signal = ours_semantic_trace_signal(
        entry["layer_vectors"],
        final_logits=entry["first_token_logits"],
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
    )
    return OursSignalResult(signal=signal, trace_names=list(REQUIRED_OURS_TRACE_NAMES))


def apply_wavelet_along_layers(
    signal: Any,
    *,
    transform: str,
    wavelet: str,
    level: int,
    mode: str = "symmetric",
) -> list[np.ndarray]:
    pywt = _import_pywt()
    array = _as_signal(signal)
    name = str(transform).lower()
    if name == "dwt":
        coeffs = pywt.wavedec(array.astype(np.float64), wavelet, mode=mode, level=int(level), axis=-1)
        return [np.asarray(coeff, dtype=np.float32) for coeff in coeffs]
    if name == "swt":
        validate_swt_level(signal_length=array.shape[-1], wavelet=wavelet, level=int(level))
        coeff_pairs = pywt.swt(array.astype(np.float64), wavelet, level=int(level), axis=-1, trim_approx=False)
        return [
            np.asarray(coeff, dtype=np.float32)
            for pair in coeff_pairs
            for coeff in pair
        ]
    raise PairedWaveletConfigError(f"unsupported legacy transform: {transform!r}")


def validate_swt_level(*, signal_length: int, wavelet: str, level: int) -> None:
    del wavelet
    max_level = _swt_max_level(int(signal_length))
    if int(level) > max_level:
        raise SWTLevelInfeasibleError(
            f"SWT level {int(level)} is not feasible for signal length {int(signal_length)}; "
            f"max level is {max_level}"
        )


def window_signal_by_layers(signal: Any, *, window_size: int, stride: int) -> np.ndarray:
    return window_signal(
        signal,
        WindowSpec(strategy="sliding", size=int(window_size), stride=int(stride)),
    )


def validate_feature_matrix(features: Any, *, context: str = "features") -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{context}: feature matrix must be 2D")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{context}: feature matrix must have non-empty dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{context}: feature matrix contains non-finite values")
    return array


def train_paired_logreg(
    features: Any,
    labels: Any,
    *,
    random_state: int = 0,
) -> Any:
    return train_static_classifier(
        "logreg",
        validate_feature_matrix(features, context="paired_features"),
        _labels(labels),
        random_state=random_state,
    )


def train_paired_xgboost_grid(
    features: Any,
    labels: Any,
    *,
    paired_configs: Sequence[Mapping[str, Any]],
    allow_no_xgboost: bool = False,
    random_state: int = 0,
) -> PairedTrainingGridResult:
    x = validate_feature_matrix(features, context="paired_features")
    y = _labels(labels)
    result = train_static_classifier(
        "xgboost",
        x,
        y,
        allow_missing_xgboost=allow_no_xgboost,
        random_state=random_state,
    )
    if result.status != "success":
        if result.failure_reason == XGBOOST_NOT_INSTALLED:
            return PairedTrainingGridResult(
                status="failure",
                rows=xgboost_missing_failure_rows(paired_configs),
                result=result,
            )
        return PairedTrainingGridResult(
            status="failure",
            rows=[
                {
                    **dict(config),
                    "status": "failure",
                    "failure_reason": result.failure_reason or "xgboost_training_failed",
                }
                for config in paired_configs
            ],
            result=result,
        )
    return PairedTrainingGridResult(
        status="success",
        rows=[
            {
                **dict(config),
                "status": "success",
                "failure_reason": "",
            }
            for config in paired_configs
        ],
        result=result,
    )


def _labels(labels: Any) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("labels must be a 1D vector")
    if array.shape[0] == 0:
        raise ValueError("labels must not be empty")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError("labels must contain only 0/1 values")
    if np.unique(array).shape[0] < 2:
        raise ValueError("labels must contain at least two classes")
    return array


def _as_signal(signal: Any) -> np.ndarray:
    if hasattr(signal, "detach") and callable(signal.detach):
        signal = signal.detach().cpu().numpy()
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError("signal must have at least one dimension")
    if not np.isfinite(array).all():
        raise ValueError("signal contains non-finite values")
    return array


def _swt_max_level(length: int) -> int:
    value = int(length)
    level = 0
    while value > 0 and value % 2 == 0:
        level += 1
        value //= 2
    return level


def _import_pywt() -> Any:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("paired wavelet transforms require pywt") from exc
    return pywt


def _module(name: str, attrs: Mapping[str, Any]) -> types.ModuleType:
    module = types.ModuleType(f"{__name__}.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module.__name__] = module
    setattr(sys.modules[__name__], name, module)
    return module


__path__ = []  # Allows ``mind.wavelet_course.paired_wavelet_v2.grid`` imports.

_module("grid", {"build_paired_grid": build_paired_grid})
_module(
    "features",
    {
        "OursSignalResult": OursSignalResult,
        "PairedWaveletConfigError": PairedWaveletConfigError,
        "apply_wavelet_along_layers": apply_wavelet_along_layers,
        "build_ours_signal": build_ours_signal,
        "build_teacher_signal": build_teacher_signal,
        "validate_swt_level": validate_swt_level,
        "window_signal_by_layers": window_signal_by_layers,
    },
)
_module("reporting", {"write_paired_metrics_csv": write_paired_metrics_csv})
_module(
    "training",
    {
        "PairedTrainingGridResult": PairedTrainingGridResult,
        "train_paired_logreg": train_paired_logreg,
        "train_paired_xgboost_grid": train_paired_xgboost_grid,
        "validate_feature_matrix": validate_feature_matrix,
    },
)


__all__ = [
    "OursSignalResult",
    "PairedTrainingGridResult",
    "PairedWaveletConfigError",
    "apply_wavelet_along_layers",
    "build_ours_signal",
    "build_paired_grid",
    "build_teacher_signal",
    "train_paired_logreg",
    "train_paired_xgboost_grid",
    "validate_feature_matrix",
    "validate_swt_level",
    "window_signal_by_layers",
    "write_paired_metrics_csv",
]
