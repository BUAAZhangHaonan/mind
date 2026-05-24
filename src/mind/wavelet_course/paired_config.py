"""Strict paired-wavelet v2 configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

PAIR_SOURCES = ("Teacher", "Ours")
PAIR_BLOCKS = ("A", "B", "C", "D", "E")
SUPPORTED_TRANSFORMS = ("none", "dwt", "swt", "wpt", "cwt")
SUPPORTED_FEATURE_PROTOCOLS = (
    "raw_sequence",
    "stat28",
    "wavelet_summary_static_pooled",
    "window_stat28_sequence",
    "window_stat28_static_flat",
    "window_stat28_static_pooled",
)
SUPPORTED_WINDOW_STRATEGIES = ("full", "non_overlapping", "sliding")
DWT_SWT_THRESHOLDS = ("none", "universal_soft", "universal_hard", "sure_soft")
NOT_APPLICABLE_THRESHOLD = "not_applicable"
SUPPORTED_THRESHOLDS = (*DWT_SWT_THRESHOLDS, NOT_APPLICABLE_THRESHOLD)
SUPPORTED_CLASSIFIERS = ("logreg", "linear_svm", "rf", "extra_trees", "xgboost")
SUPPORTED_SEQUENCE_MODELS = ("lstm_projected", "gru_projected", "tcn", "cnn1d")
SUPPORTED_WINDOW_MODES = ("global", "win4_s4", "win6_s3", "win9_s9", "win12_s6")

TEACHER_SIGNAL_BUILDER = "teacher_hidden_dim_signal"
OURS_SIGNAL_BUILDER = "ours_semantic_trace_signal"
SOURCE_SIGNAL_BUILDERS = {
    "Teacher": TEACHER_SIGNAL_BUILDER,
    "Ours": OURS_SIGNAL_BUILDER,
}
DEFAULT_CWT_SCALES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0)


@dataclass(frozen=True, slots=True)
class PairSpec:
    """One row in the paired v2 grid.

    A pair is represented by two rows with the same ``pair_id`` and block:
    one ``Teacher`` row and one ``Ours`` row. The rows differ only in source
    and signal builder.
    """

    pair_id: str
    block: str
    source: str
    signal_builder: str
    transform: str
    feature_protocol: str
    wavelet: str | None = None
    level: int | None = None
    window_strategy: str = "full"
    window_size: int | None = None
    stride: int | None = None
    mode: str = "symmetric"
    cwt_scales: tuple[float, ...] = DEFAULT_CWT_SCALES
    threshold: str = "none"
    classifier: str | None = None
    sequence_model: str | None = None
    window_mode: str = "global"

    def __post_init__(self) -> None:
        pair_id = _require_text(self.pair_id, "pair_id")
        block = _require_choice(self.block, PAIR_BLOCKS, "block")
        source = _require_choice(self.source, PAIR_SOURCES, "source")
        signal_builder = _require_text(self.signal_builder, "signal_builder")
        expected_builder = SOURCE_SIGNAL_BUILDERS[source]
        if signal_builder != expected_builder:
            raise ValueError(
                f"{source} rows must use signal_builder={expected_builder!r}, "
                f"got {signal_builder!r}"
            )
        transform = _require_choice(str(self.transform).lower(), SUPPORTED_TRANSFORMS, "transform")
        feature_protocol = _require_choice(
            str(self.feature_protocol),
            SUPPORTED_FEATURE_PROTOCOLS,
            "feature_protocol",
        )
        window_strategy = _require_choice(
            str(self.window_strategy),
            SUPPORTED_WINDOW_STRATEGIES,
            "window_strategy",
        )
        mode = _require_text(self.mode, "mode")
        cwt_scales = _coerce_scales(self.cwt_scales)
        threshold = _require_choice(str(self.threshold).lower(), SUPPORTED_THRESHOLDS, "threshold")
        classifier = _optional_choice(self.classifier, SUPPORTED_CLASSIFIERS, "classifier")
        sequence_model = _optional_choice(self.sequence_model, SUPPORTED_SEQUENCE_MODELS, "sequence_model")
        window_mode = _require_choice(str(self.window_mode), SUPPORTED_WINDOW_MODES, "window_mode")

        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "signal_builder", signal_builder)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "feature_protocol", feature_protocol)
        object.__setattr__(self, "window_strategy", window_strategy)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "cwt_scales", cwt_scales)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "classifier", classifier)
        object.__setattr__(self, "sequence_model", sequence_model)
        object.__setattr__(self, "window_mode", window_mode)

        if transform == "none":
            if self.wavelet not in {None, "", "none"}:
                raise ValueError("transform='none' does not accept a wavelet")
            if self.level is not None:
                raise ValueError("transform='none' does not accept a level")
            if threshold != "none":
                raise ValueError("transform='none' only accepts threshold='none'")
            object.__setattr__(self, "wavelet", None)
        elif transform == "cwt":
            if threshold != NOT_APPLICABLE_THRESHOLD:
                raise ValueError("transform='cwt' requires threshold='not_applicable'")
            if self.level is not None:
                raise ValueError("transform='cwt' uses cwt_scales, not level")
            object.__setattr__(self, "wavelet", _require_text(self.wavelet, "wavelet"))
        elif transform == "wpt":
            if threshold != NOT_APPLICABLE_THRESHOLD:
                raise ValueError("transform='wpt' requires threshold='not_applicable'")
            object.__setattr__(self, "wavelet", _require_text(self.wavelet, "wavelet"))
            object.__setattr__(self, "level", _require_positive_int(self.level, "level"))
        else:
            if threshold not in DWT_SWT_THRESHOLDS:
                raise ValueError(f"transform={transform!r} only accepts thresholds {DWT_SWT_THRESHOLDS}")
            object.__setattr__(self, "wavelet", _require_text(self.wavelet, "wavelet"))
            object.__setattr__(self, "level", _require_positive_int(self.level, "level"))

        _validate_window(window_strategy, self.window_size, self.stride)
        if window_strategy == "full":
            object.__setattr__(self, "window_size", None)
            object.__setattr__(self, "stride", None)
        elif window_strategy == "non_overlapping":
            size = _require_positive_int(self.window_size, "window_size")
            stride = size if self.stride is None else _require_positive_int(self.stride, "stride")
            if stride != size:
                raise ValueError("non_overlapping windows require stride == window_size")
            object.__setattr__(self, "window_size", size)
            object.__setattr__(self, "stride", stride)
        else:
            object.__setattr__(self, "window_size", _require_positive_int(self.window_size, "window_size"))
            object.__setattr__(self, "stride", _require_positive_int(self.stride, "stride"))

    @property
    def row_id(self) -> str:
        return f"{self.pair_id}::{self.source}"

    def paired_key(self) -> tuple[Any, ...]:
        """Return the fields that must match between Teacher and Ours rows."""

        return (
            self.pair_id,
            self.block,
            self.transform,
            self.feature_protocol,
            self.wavelet,
            self.level,
            self.window_strategy,
            self.window_size,
            self.stride,
            self.mode,
            self.cwt_scales,
            self.threshold,
            self.classifier,
            self.sequence_model,
            self.window_mode,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "row_id": self.row_id,
            "block": self.block,
            "source": self.source,
            "signal_builder": self.signal_builder,
            "transform": self.transform,
            "feature_protocol": self.feature_protocol,
            "wavelet": self.wavelet,
            "level": self.level,
            "window_strategy": self.window_strategy,
            "window_size": self.window_size,
            "stride": self.stride,
            "mode": self.mode,
            "cwt_scales": list(self.cwt_scales),
            "threshold": self.threshold,
            "classifier": self.classifier,
            "sequence_model": self.sequence_model,
            "window_mode": self.window_mode,
        }


@dataclass(frozen=True, slots=True)
class PairedRunSpec:
    """Complete paired-wavelet v2 run configuration."""

    run_id: str
    pairs: tuple[PairSpec, ...]
    expected_blocks: tuple[str, ...] = PAIR_BLOCKS
    expected_sources: tuple[str, ...] = PAIR_SOURCES
    expected_num_layers: int | None = 36
    expected_hidden_dim: int | None = 4096
    epsilon: float = 1e-12
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        pairs = tuple(self.pairs)
        if not pairs:
            raise ValueError("pairs must not be empty")
        if not all(isinstance(pair, PairSpec) for pair in pairs):
            raise ValueError("pairs must contain only PairSpec rows")
        object.__setattr__(self, "pairs", pairs)

        blocks = tuple(_require_choice(block, PAIR_BLOCKS, "expected_blocks") for block in self.expected_blocks)
        sources = tuple(_require_choice(source, PAIR_SOURCES, "expected_sources") for source in self.expected_sources)
        if not blocks:
            raise ValueError("expected_blocks must not be empty")
        if not sources:
            raise ValueError("expected_sources must not be empty")
        object.__setattr__(self, "expected_blocks", blocks)
        object.__setattr__(self, "expected_sources", sources)

        if self.expected_num_layers is not None:
            object.__setattr__(
                self,
                "expected_num_layers",
                _require_positive_int(self.expected_num_layers, "expected_num_layers"),
            )
        if self.expected_hidden_dim is not None:
            object.__setattr__(
                self,
                "expected_hidden_dim",
                _require_positive_int(self.expected_hidden_dim, "expected_hidden_dim"),
            )
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "description", str(self.description))

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "expected_blocks": list(self.expected_blocks),
            "expected_sources": list(self.expected_sources),
            "expected_num_layers": self.expected_num_layers,
            "expected_hidden_dim": self.expected_hidden_dim,
            "epsilon": self.epsilon,
            "description": self.description,
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


def pair_spec_from_mapping(row: Mapping[str, Any]) -> PairSpec:
    """Build a strict ``PairSpec`` from a mapping."""

    return PairSpec(
        pair_id=row["pair_id"],
        block=row["block"],
        source=row["source"],
        signal_builder=row["signal_builder"],
        transform=row["transform"],
        feature_protocol=row["feature_protocol"],
        wavelet=row.get("wavelet"),
        level=row.get("level"),
        window_strategy=row.get("window_strategy", "full"),
        window_size=row.get("window_size"),
        stride=row.get("stride"),
        mode=row.get("mode", "symmetric"),
        cwt_scales=tuple(row.get("cwt_scales", DEFAULT_CWT_SCALES)),
        threshold=row.get("threshold", _default_threshold_for_transform(row.get("transform"))),
        classifier=row.get("classifier"),
        sequence_model=row.get("sequence_model"),
        window_mode=row.get("window_mode", "global"),
    )


def _require_text(value: object, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _require_choice(value: object, allowed: Sequence[str], name: str) -> str:
    text = _require_text(value, name)
    if text not in set(allowed):
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {text!r}")
    return text


def _optional_choice(value: object, allowed: Sequence[str], name: str) -> str | None:
    if value in {None, ""}:
        return None
    return _require_choice(str(value).lower(), allowed, name)


def _require_positive_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _coerce_scales(values: object) -> tuple[float, ...]:
    try:
        scales = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("cwt_scales must be a finite non-empty sequence") from exc
    if not scales:
        raise ValueError("cwt_scales must not be empty")
    if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("cwt_scales must contain only finite positive values")
    return scales


def _default_threshold_for_transform(transform: object) -> str:
    return NOT_APPLICABLE_THRESHOLD if str(transform).lower() in {"cwt", "wpt"} else "none"


def _validate_window(strategy: str, size: int | None, stride: int | None) -> None:
    if strategy == "full":
        if size is not None or stride is not None:
            raise ValueError("full windows do not accept window_size or stride")
        return
    if strategy == "non_overlapping":
        if size is None:
            raise ValueError("non_overlapping windows require window_size")
        return
    if strategy == "sliding":
        if size is None or stride is None:
            raise ValueError("sliding windows require window_size and stride")


__all__ = [
    "DEFAULT_CWT_SCALES",
    "DWT_SWT_THRESHOLDS",
    "NOT_APPLICABLE_THRESHOLD",
    "OURS_SIGNAL_BUILDER",
    "PAIR_BLOCKS",
    "PAIR_SOURCES",
    "PairSpec",
    "PairedRunSpec",
    "SOURCE_SIGNAL_BUILDERS",
    "SUPPORTED_CLASSIFIERS",
    "SUPPORTED_FEATURE_PROTOCOLS",
    "SUPPORTED_SEQUENCE_MODELS",
    "SUPPORTED_THRESHOLDS",
    "SUPPORTED_TRANSFORMS",
    "SUPPORTED_WINDOW_MODES",
    "SUPPORTED_WINDOW_STRATEGIES",
    "TEACHER_SIGNAL_BUILDER",
    "pair_spec_from_mapping",
]
