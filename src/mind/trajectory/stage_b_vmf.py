"""Single-vMF prototype diagnostics for Stage B embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def fit_single_vmf_prototype(embeddings: np.ndarray | Sequence[Sequence[float]]) -> dict[str, object]:
    """Fit a single vMF-style prototype by normalized resultant direction."""

    values = _row_normalize(_as_2d_array(embeddings, name="embeddings"))
    resultant = values.sum(axis=0)
    resultant_norm = float(np.linalg.norm(resultant))
    if resultant_norm <= 0.0 or not np.isfinite(resultant_norm):
        mean_direction = np.zeros(values.shape[1], dtype=np.float64)
        mean_direction[0] = 1.0
        r_bar = 0.0
    else:
        mean_direction = resultant / resultant_norm
        r_bar = resultant_norm / float(values.shape[0])

    concentration = _concentration_proxy(r_bar, values.shape[1])
    return {
        "mean_direction": mean_direction.astype(float).tolist(),
        "resultant_length": float(resultant_norm),
        "mean_resultant_length": float(r_bar),
        "concentration_proxy": float(concentration),
        "num_bank": int(values.shape[0]),
        "embedding_dim": int(values.shape[1]),
    }


def score_single_vmf_prototype(
    prototype: Mapping[str, object],
    query_embeddings: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Score queries by negative vMF alignment with the fitted direction."""

    queries = _row_normalize(_as_2d_array(query_embeddings, name="query_embeddings"))
    direction = np.asarray(prototype["mean_direction"], dtype=np.float64)
    if direction.ndim != 1 or direction.shape[0] != queries.shape[1]:
        raise ValueError("prototype mean_direction does not match query dimensionality")
    norm = np.linalg.norm(direction)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("prototype mean_direction must have positive finite norm")
    direction = direction / norm
    concentration = float(prototype.get("concentration_proxy", 1.0))
    return (-(queries @ direction) * concentration).astype(np.float32)


def compute_vmf_diagnostic(
    *,
    bank_embeddings: np.ndarray | Sequence[Sequence[float]],
    query_embeddings: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> dict[str, object]:
    """Fit a single prototype and score query rows."""

    prototype = fit_single_vmf_prototype(bank_embeddings)
    if query_embeddings is None:
        scores: list[float] = []
    else:
        scores = score_single_vmf_prototype(prototype, query_embeddings).astype(float).tolist()
    result = dict(prototype)
    result["scores"] = scores
    return result


def write_vmf_diagnostic(result: Mapping[str, object], output_path: Path | str) -> dict[str, object]:
    """Write a vMF diagnostic payload as JSON."""

    payload = _jsonable(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _concentration_proxy(r_bar: float, dim: int) -> float:
    clipped = min(max(float(r_bar), 0.0), 0.999999)
    if clipped <= 0.0:
        return 0.0
    value = clipped * (float(dim) - clipped * clipped) / max(1.0 - clipped * clipped, 1e-12)
    if not np.isfinite(value):
        return 0.0
    return max(float(value), 0.0)


def _as_2d_array(value: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _row_normalize(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("embedding rows must have positive finite norm")
    return value / norms


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
