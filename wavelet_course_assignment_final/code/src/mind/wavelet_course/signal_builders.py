"""Signal builders for paired-wavelet v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EPSILON = 1e-12
YES_NO_TRACE_SOURCE = "final_broadcast"
REQUIRED_OURS_TRACE_NAMES = (
    "norm_trace",
    "delta_norm_trace",
    "cos_prev_trace",
    "cos_final_trace",
    "second_delta_norm_trace",
    "curvature_trace",
    "middle_late_alignment_trace",
    "yes_no_margin_trace",
    "yes_no_entropy_trace",
    "hidden_variance_trace",
)


def teacher_hidden_dim_signal(
    layer_vectors: Any,
    *,
    expected_num_layers: int | None = None,
    expected_hidden_dim: int | None = None,
) -> np.ndarray:
    """Return hidden-dimension layer traces with shape ``(hidden_dim, layers)``."""

    vectors = _as_layer_vectors(
        layer_vectors,
        expected_num_layers=expected_num_layers,
        expected_hidden_dim=expected_hidden_dim,
    )
    signal = np.swapaxes(vectors, 0, 1).astype(np.float32, copy=False)
    _raise_if_non_finite(signal, name="teacher_hidden_dim_signal")
    return signal


def ours_semantic_trace_signal(
    layer_vectors: Any,
    *,
    final_logits: Any | None = None,
    yes_token_id: int | None = None,
    no_token_id: int | None = None,
    final_yes_logit: float | None = None,
    final_no_logit: float | None = None,
    yes_logit: float | None = None,
    no_logit: float | None = None,
    expected_num_layers: int | None = None,
    expected_hidden_dim: int | None = None,
    trace_names: Sequence[str] = REQUIRED_OURS_TRACE_NAMES,
    epsilon: float = EPSILON,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[str, ...]]:
    """Return Ours semantic traces with shape ``(10, 36)``.

    The yes/no traces are final-layer broadcasts. Callers must provide either
    final logits plus yes/no token ids, or explicit final yes/no logits.
    """

    names = tuple(trace_names)
    if names != REQUIRED_OURS_TRACE_NAMES:
        raise ValueError(f"trace_names must be exactly {REQUIRED_OURS_TRACE_NAMES!r}")
    if epsilon <= 0.0 or not np.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")

    vectors = _as_layer_vectors(
        layer_vectors,
        expected_num_layers=expected_num_layers,
        expected_hidden_dim=expected_hidden_dim,
    )
    yes_value, no_value = _resolve_yes_no_logits(
        final_logits=final_logits,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        final_yes_logit=final_yes_logit if final_yes_logit is not None else yes_logit,
        final_no_logit=final_no_logit if final_no_logit is not None else no_logit,
    )
    traces = ours_semantic_traces(
        vectors,
        final_yes_logit=yes_value,
        final_no_logit=no_value,
        epsilon=float(epsilon),
    )
    signal = np.stack([traces[name] for name in names], axis=0).astype(np.float32, copy=False)
    _raise_if_non_finite(signal, name="ours_semantic_trace_signal")
    if return_names:
        return signal, names
    return signal


def ours_semantic_traces(
    layer_vectors: Any,
    *,
    final_yes_logit: float,
    final_no_logit: float,
    epsilon: float = EPSILON,
) -> dict[str, np.ndarray]:
    """Build the named Ours semantic traces for the 36-layer v2 signal."""

    if epsilon <= 0.0 or not np.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    vectors = _as_layer_vectors(layer_vectors, expected_num_layers=36)
    yes_value = _require_finite_scalar(final_yes_logit, "final_yes_logit")
    no_value = _require_finite_scalar(final_no_logit, "final_no_logit")
    length = vectors.shape[0]

    norms = np.linalg.norm(vectors, axis=1).astype(np.float64)
    log_norms = np.log(norms + float(epsilon))

    layer_deltas = np.zeros_like(vectors, dtype=np.float64)
    layer_deltas[1:] = vectors[1:].astype(np.float64) - vectors[:-1].astype(np.float64)
    delta = np.empty_like(norms)
    delta[0] = 0.0
    delta[1:] = np.linalg.norm(layer_deltas[1:], axis=1)

    second_delta = np.zeros(length, dtype=np.float64)
    second_delta[2:] = np.linalg.norm(layer_deltas[2:] - layer_deltas[1:-1], axis=1)

    cos_prev = np.empty(length, dtype=np.float64)
    cos_prev[0] = 1.0
    for index in range(1, length):
        cos_prev[index] = _cosine(vectors[index], vectors[index - 1], epsilon=float(epsilon))

    final = vectors[-1]
    cos_final = np.asarray(
        [_cosine(vector, final, epsilon=float(epsilon)) for vector in vectors],
        dtype=np.float64,
    )
    curvature = np.zeros(length, dtype=np.float64)
    for index in range(2, length):
        curvature[index] = 1.0 - _cosine(
            layer_deltas[index],
            layer_deltas[index - 1],
            epsilon=float(epsilon),
        )

    middle_mean = np.mean(vectors[9:27], axis=0)
    late_mean = np.mean(vectors[27:36], axis=0)
    middle_late_alignment = np.asarray(
        [
            _cosine(vector, late_mean, epsilon=float(epsilon))
            - _cosine(vector, middle_mean, epsilon=float(epsilon))
            for vector in vectors
        ],
        dtype=np.float64,
    )
    hidden_variance = np.var(vectors.astype(np.float64), axis=1)
    margin, entropy = _yes_no_margin_entropy(yes_value, no_value, epsilon=float(epsilon))
    traces = {
        "norm_trace": log_norms,
        "delta_norm_trace": delta,
        "cos_prev_trace": cos_prev,
        "cos_final_trace": cos_final,
        "second_delta_norm_trace": second_delta,
        "curvature_trace": curvature,
        "middle_late_alignment_trace": middle_late_alignment,
        "yes_no_margin_trace": np.full(length, margin, dtype=np.float64),
        "yes_no_entropy_trace": np.full(length, entropy, dtype=np.float64),
        "hidden_variance_trace": hidden_variance,
    }
    for name, trace in traces.items():
        _raise_if_non_finite(trace, name=name)
    return traces


def _resolve_yes_no_logits(
    *,
    final_logits: Any | None,
    yes_token_id: int | None,
    no_token_id: int | None,
    final_yes_logit: float | None,
    final_no_logit: float | None,
) -> tuple[float, float]:
    explicit = final_yes_logit is not None or final_no_logit is not None
    token_based = final_logits is not None or yes_token_id is not None or no_token_id is not None
    if explicit and token_based:
        raise ValueError("provide either explicit final yes/no logits or final_logits with token ids, not both")
    if explicit:
        if final_yes_logit is None or final_no_logit is None:
            raise ValueError("explicit final yes/no logits require both final_yes_logit and final_no_logit")
        return (
            _require_finite_scalar(final_yes_logit, "final_yes_logit"),
            _require_finite_scalar(final_no_logit, "final_no_logit"),
        )
    if final_logits is None or yes_token_id is None or no_token_id is None:
        raise ValueError("final_logits, yes_token_id, and no_token_id are required for yes/no broadcast traces")
    logits = _final_logits(final_logits)
    yes_id = int(yes_token_id)
    no_id = int(no_token_id)
    if not 0 <= yes_id < logits.shape[0]:
        raise ValueError(f"yes_token_id {yes_id} is outside logits size {logits.shape[0]}")
    if not 0 <= no_id < logits.shape[0]:
        raise ValueError(f"no_token_id {no_id} is outside logits size {logits.shape[0]}")
    return float(logits[yes_id]), float(logits[no_id])


def _as_layer_vectors(
    values: Any,
    *,
    expected_num_layers: int | None = None,
    expected_hidden_dim: int | None = None,
) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("layer_vectors must have shape (layers, hidden_dim)")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("layer_vectors must have non-empty layer and hidden dimensions")
    if expected_num_layers is not None and array.shape[0] != int(expected_num_layers):
        raise ValueError(f"layer_vectors has {array.shape[0]} layers, expected {expected_num_layers}")
    if expected_hidden_dim is not None and array.shape[1] != int(expected_hidden_dim):
        raise ValueError(f"layer_vectors has hidden_dim {array.shape[1]}, expected {expected_hidden_dim}")
    _raise_if_non_finite(array, name="layer_vectors")
    return array


def _final_logits(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[0] == 0:
            raise ValueError("final_logits 2D array must have at least one row")
        array = array[-1]
    if array.ndim != 1:
        raise ValueError("final_logits must be a 1D vector or 2D layer-by-vocab array")
    if array.shape[0] == 0:
        raise ValueError("final_logits must not be empty")
    _raise_if_non_finite(array, name="final_logits")
    return array


def _yes_no_margin_entropy(yes_logit: float, no_logit: float, *, epsilon: float) -> tuple[float, float]:
    pair = np.asarray([yes_logit, no_logit], dtype=np.float64)
    shifted = pair - float(np.max(pair))
    probs = np.exp(shifted)
    probs = probs / max(float(np.sum(probs)), epsilon)
    entropy = -float(np.sum(np.maximum(probs, epsilon) * np.log(np.maximum(probs, epsilon))))
    return float(pair[0] - pair[1]), entropy


def _cosine(left: np.ndarray, right: np.ndarray, *, epsilon: float) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= epsilon:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def _require_finite_scalar(value: object, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


__all__ = [
    "EPSILON",
    "REQUIRED_OURS_TRACE_NAMES",
    "YES_NO_TRACE_SOURCE",
    "ours_semantic_trace_signal",
    "ours_semantic_traces",
    "teacher_hidden_dim_signal",
]
