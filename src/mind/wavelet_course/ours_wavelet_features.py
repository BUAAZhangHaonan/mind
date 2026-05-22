"""Compact SWT features for six layer-course traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_LAYER_SHAPE = (36, 4096)
TRACE_LENGTH = 36
MAX_FEATURE_DIM = 512
EPSILON = 1e-12

TRACE_NAMES = (
    "norm_trace",
    "delta_norm_trace",
    "cos_prev_trace",
    "cos_final_trace",
    "yes_no_margin_trace",
    "yes_no_entropy_trace",
)

YES_CANDIDATES = ("yes", "Yes", "YES", " yes", " Yes", " YES")
NO_CANDIDATES = ("no", "No", "NO", " no", " No", " NO")


class WaveletConfigError(ValueError):
    """Raised when an SWT configuration is not feasible for the trace length."""


@dataclass(frozen=True)
class OursWaveletConfig:
    wavelet: str = "db2"
    level: int = 2
    yes_token_id: int | None = None
    no_token_id: int | None = None
    epsilon: float = EPSILON


@dataclass(frozen=True)
class OursWaveletFeatureResult:
    features: np.ndarray
    feature_names: list[str]


def extract_ours_wavelet_features(
    layer_vectors: Any,
    first_token_logits: Any,
    config: OursWaveletConfig | Mapping[str, Any] | None = None,
    *,
    tokenizer: Any | None = None,
    vocab: Mapping[str, int] | None = None,
) -> OursWaveletFeatureResult:
    """Build one compact Ours-Wavelet feature vector."""

    cfg = _coerce_config(config)
    yes_id, no_id = _resolve_ids_for_config(cfg, tokenizer=tokenizer, vocab=vocab, logits=first_token_logits)
    vectors = _as_layer_vectors(layer_vectors)
    logits = _final_logits(first_token_logits)
    _validate_token_id(yes_id, logits_size=logits.shape[0], name="yes_token_id")
    _validate_token_id(no_id, logits_size=logits.shape[0], name="no_token_id")
    traces = build_ours_traces(vectors, logits, yes_token_id=yes_id, no_token_id=no_id, epsilon=cfg.epsilon)
    features, names = _features_from_traces(traces, cfg)
    if features.shape[0] > MAX_FEATURE_DIM:
        raise ValueError(f"Ours-Wavelet feature_dim {features.shape[0]} exceeds {MAX_FEATURE_DIM}")
    _raise_if_non_finite(features, name="ours_wavelet_features")
    return OursWaveletFeatureResult(features=features.astype(np.float32, copy=False), feature_names=names)


def build_ours_traces(
    layer_vectors: Any,
    final_logits: Any,
    *,
    yes_token_id: int,
    no_token_id: int,
    epsilon: float = EPSILON,
) -> dict[str, np.ndarray]:
    """Construct the six 36-point traces used by Ours-Wavelet."""

    vectors = _as_layer_vectors(layer_vectors)
    logits = _final_logits(final_logits)
    norms = np.linalg.norm(vectors, axis=1).astype(np.float64)
    delta = np.empty_like(norms)
    delta[0] = 0.0
    delta[1:] = np.diff(norms)

    cos_prev = np.empty(TRACE_LENGTH, dtype=np.float64)
    cos_prev[0] = 1.0
    for index in range(1, TRACE_LENGTH):
        cos_prev[index] = _cosine(vectors[index], vectors[index - 1], epsilon=epsilon)

    final = vectors[-1]
    cos_final = np.asarray([_cosine(vector, final, epsilon=epsilon) for vector in vectors], dtype=np.float64)
    margin, entropy = _yes_no_margin_entropy(logits, yes_token_id=yes_token_id, no_token_id=no_token_id, epsilon=epsilon)
    margin_trace = np.full(TRACE_LENGTH, margin, dtype=np.float64)
    entropy_trace = np.full(TRACE_LENGTH, entropy, dtype=np.float64)
    traces = {
        "norm_trace": norms,
        "delta_norm_trace": delta,
        "cos_prev_trace": cos_prev,
        "cos_final_trace": cos_final,
        "yes_no_margin_trace": margin_trace,
        "yes_no_entropy_trace": entropy_trace,
    }
    for name, trace in traces.items():
        _raise_if_non_finite(trace, name=name)
    return traces


def resolve_yes_no_token_ids(
    tokenizer: Any | None = None,
    vocab: Mapping[str, int] | None = None,
    logits_size: int | None = None,
) -> tuple[int, int]:
    """Resolve yes/no token ids from a tokenizer or vocabulary."""

    yes_ids = _candidate_token_ids(YES_CANDIDATES, tokenizer=tokenizer, vocab=vocab, logits_size=logits_size)
    no_ids = _candidate_token_ids(NO_CANDIDATES, tokenizer=tokenizer, vocab=vocab, logits_size=logits_size)
    if not yes_ids or not no_ids:
        raise ValueError("could not resolve yes/no token ids from tokenizer or vocab")
    return min(yes_ids), min(no_ids)


def save_ours_features(
    features: np.ndarray,
    feature_names: Sequence[str],
    output_path: str | Path,
    *,
    names_path: str | Path | None = None,
) -> dict[str, Any]:
    """Save Ours-Wavelet features as NPY and feature names as JSON."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError("features must be a 1D or 2D array")
    _raise_if_non_finite(array, name="features")
    names = [str(name) for name in feature_names]
    feature_dim = int(array.shape[-1])
    if len(names) != feature_dim:
        raise ValueError("feature_names length must match feature dimension")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    json_path = Path(names_path) if names_path is not None else path.with_suffix(".feature_names.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"feature_dim": feature_dim, "feature_names": names}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _coerce_config(config: OursWaveletConfig | Mapping[str, Any] | None) -> OursWaveletConfig:
    if config is None:
        cfg = OursWaveletConfig()
    elif isinstance(config, OursWaveletConfig):
        cfg = config
    else:
        cfg = OursWaveletConfig(
            wavelet=str(config.get("wavelet", OursWaveletConfig.wavelet)),
            level=int(config.get("level", OursWaveletConfig.level)),
            yes_token_id=_optional_int(config.get("yes_token_id")),
            no_token_id=_optional_int(config.get("no_token_id")),
            epsilon=float(config.get("epsilon", EPSILON)),
        )
    if cfg.wavelet == "db2" and cfg.level not in {2, 3}:
        raise WaveletConfigError("db2 SWT supports only level 2 or 3 for this module")
    if cfg.wavelet == "sym4" and cfg.level != 2:
        raise WaveletConfigError("sym4 SWT supports only level 2 for this module")
    if cfg.wavelet not in {"db2", "sym4"}:
        raise WaveletConfigError("Ours-Wavelet supports only db2 or sym4")
    if cfg.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return cfg


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_ids_for_config(
    config: OursWaveletConfig,
    *,
    tokenizer: Any | None,
    vocab: Mapping[str, int] | None,
    logits: Any,
) -> tuple[int, int]:
    logits_size = int(_final_logits(logits).shape[0])
    if config.yes_token_id is not None and config.no_token_id is not None:
        return int(config.yes_token_id), int(config.no_token_id)
    if tokenizer is None and vocab is None:
        raise ValueError("yes_token_id/no_token_id are required when tokenizer or vocab is not provided")
    return resolve_yes_no_token_ids(tokenizer=tokenizer, vocab=vocab, logits_size=logits_size)


def _candidate_token_ids(
    candidates: Sequence[str],
    *,
    tokenizer: Any | None,
    vocab: Mapping[str, int] | None,
    logits_size: int | None,
) -> set[int]:
    ids: set[int] = set()
    if vocab is not None:
        for token in candidates:
            value = vocab.get(token)
            if value is not None:
                ids.add(int(value))
    if tokenizer is not None:
        for token in candidates:
            ids.update(_ids_from_tokenizer(tokenizer, token))
    if logits_size is not None:
        ids = {token_id for token_id in ids if 0 <= token_id < int(logits_size)}
    return ids


def _ids_from_tokenizer(tokenizer: Any, text: str) -> set[int]:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(text)
        if isinstance(token_id, int) and token_id >= 0:
            return {token_id}
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(encoded, Sequence) and len(encoded) == 1:
            return {int(encoded[0])}
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if isinstance(input_ids, Sequence) and len(input_ids) == 1:
            return {int(input_ids[0])}
    return set()


def _features_from_traces(
    traces: Mapping[str, np.ndarray],
    config: OursWaveletConfig,
) -> tuple[np.ndarray, list[str]]:
    _check_swt_feasible(trace_length=TRACE_LENGTH, wavelet=config.wavelet, level=config.level)
    values: list[float] = []
    names: list[str] = []
    for trace_name in TRACE_NAMES:
        trace = np.asarray(traces[trace_name], dtype=np.float64)
        trace_values, trace_names = _trace_features(trace, trace_name=trace_name, config=config)
        values.extend(trace_values)
        names.extend(trace_names)
    return np.asarray(values, dtype=np.float32), names


def _check_swt_feasible(*, trace_length: int, wavelet: str, level: int) -> None:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("Ours-Wavelet feature extraction requires pywt") from exc
    try:
        pywt.Wavelet(wavelet)
        max_level = pywt.swt_max_level(trace_length)
    except Exception as exc:
        raise WaveletConfigError(f"invalid SWT wavelet configuration: {wavelet}") from exc
    if int(level) > int(max_level):
        raise WaveletConfigError(
            f"SWT level {level} is not feasible for trace length {trace_length}; max level is {max_level}"
        )


def _trace_features(
    trace: np.ndarray,
    *,
    trace_name: str,
    config: OursWaveletConfig,
) -> tuple[list[float], list[str]]:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("Ours-Wavelet feature extraction requires pywt") from exc

    coeffs = pywt.swt(trace, config.wavelet, level=int(config.level), trim_approx=False)
    values, names = _raw_trace_features(trace, trace_name=trace_name)
    total_trace_energy = float(np.sum(np.square(trace)))
    for level_index, (approx, detail) in enumerate(coeffs, start=1):
        wave_values, wave_names = _wavelet_level_features(
            approx=np.asarray(approx, dtype=np.float64),
            detail=np.asarray(detail, dtype=np.float64),
            trace_name=trace_name,
            level_index=level_index,
            total_trace_energy=total_trace_energy,
            epsilon=config.epsilon,
        )
        values.extend(wave_values)
        names.extend(wave_names)
    return values, names


def _raw_trace_features(trace: np.ndarray, *, trace_name: str) -> tuple[list[float], list[str]]:
    x = np.arange(trace.shape[0], dtype=np.float64)
    slope = float(np.polyfit(x, trace, deg=1)[0])
    names = [
        "mean",
        "std",
        "min",
        "max",
        "slope",
        "last_minus_first",
        "middle_mean",
        "late_mean",
        "mid_late_diff",
    ]
    middle_mean = float(np.mean(trace[9:27]))
    late_mean = float(np.mean(trace[27:36]))
    values = [
        float(np.mean(trace)),
        float(np.std(trace)),
        float(np.min(trace)),
        float(np.max(trace)),
        slope,
        float(trace[-1] - trace[0]),
        middle_mean,
        late_mean,
        float(late_mean - middle_mean),
    ]
    return values, [f"{trace_name}__raw__{name}" for name in names]


def _wavelet_level_features(
    *,
    approx: np.ndarray,
    detail: np.ndarray,
    trace_name: str,
    level_index: int,
    total_trace_energy: float,
    epsilon: float,
) -> tuple[list[float], list[str]]:
    detail_energy = np.square(detail)
    approx_energy = np.square(approx)
    detail_total = float(np.sum(detail_energy))
    approx_total = float(np.sum(approx_energy))
    total_energy = detail_total + approx_total
    probabilities = detail_energy / max(detail_total, epsilon)
    high_frequency = float(np.mean(np.abs(detail)))
    center, spread = _energy_center_spread(detail_energy, epsilon=epsilon)
    prefix = f"{trace_name}__swt_l{level_index}"
    names = [
        "energy",
        "detail_energy",
        "approx_energy",
        "detail_approx_ratio",
        "high_frequency",
        "detail_entropy",
        "max_abs_detail",
        "argmax_abs_detail",
        "detail_energy_center",
        "detail_energy_spread",
    ]
    values = [
        total_energy,
        detail_total,
        approx_total,
        detail_total / max(approx_total, epsilon),
        high_frequency,
        _entropy(probabilities, epsilon=epsilon),
        float(np.max(np.abs(detail))),
        float(np.argmax(np.abs(detail))),
        center,
        spread,
    ]
    if total_trace_energy <= epsilon:
        values[0] = 0.0
    return values, [f"{prefix}__{name}" for name in names]


def _energy_center_spread(energy: np.ndarray, *, epsilon: float) -> tuple[float, float]:
    total = float(np.sum(energy))
    if total <= epsilon:
        return 0.0, 0.0
    positions = np.arange(energy.shape[0], dtype=np.float64)
    center = float(np.sum(positions * energy) / total)
    spread = float(np.sqrt(np.sum(((positions - center) ** 2) * energy) / total))
    return center, spread


def _as_layer_vectors(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.shape != EXPECTED_LAYER_SHAPE:
        raise ValueError(f"layer_vectors must have shape {EXPECTED_LAYER_SHAPE}")
    _raise_if_non_finite(array, name="layer_vectors")
    return array


def _final_logits(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[-1]
    if array.ndim != 1:
        raise ValueError("first_token_logits must be a 1D vector or 2D layer-by-vocab array")
    if array.shape[0] == 0:
        raise ValueError("first_token_logits must not be empty")
    _raise_if_non_finite(array, name="first_token_logits")
    return array


def _validate_token_id(token_id: int, *, logits_size: int, name: str) -> None:
    if not 0 <= int(token_id) < int(logits_size):
        raise ValueError(f"{name} {token_id} is outside logits size {logits_size}")


def _yes_no_margin_entropy(
    logits: np.ndarray,
    *,
    yes_token_id: int,
    no_token_id: int,
    epsilon: float,
) -> tuple[float, float]:
    pair = np.asarray([logits[int(yes_token_id)], logits[int(no_token_id)]], dtype=np.float64)
    shifted = pair - float(np.max(pair))
    probs = np.exp(shifted)
    probs = probs / max(float(np.sum(probs)), epsilon)
    margin = float(pair[0] - pair[1])
    entropy = _entropy(probs, epsilon=epsilon)
    return margin, entropy


def _cosine(left: np.ndarray, right: np.ndarray, *, epsilon: float) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= epsilon:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def _entropy(probabilities: np.ndarray, *, epsilon: float) -> float:
    safe = np.maximum(np.asarray(probabilities, dtype=np.float64), epsilon)
    return float(-np.sum(safe * np.log(safe)))


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
