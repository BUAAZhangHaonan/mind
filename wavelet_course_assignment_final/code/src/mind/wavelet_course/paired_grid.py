"""Paired-wavelet v2 grid construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from .paired_config import (
    OURS_SIGNAL_BUILDER,
    PAIR_BLOCKS,
    PAIR_SOURCES,
    TEACHER_SIGNAL_BUILDER,
    PairSpec,
    PairedRunSpec,
)


CWT_SCALES_1_16 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0]


def _window_fields(window_mode: str) -> dict[str, int | str]:
    if window_mode == "global":
        return {"window_strategy": "full"}
    if window_mode == "win4_s4":
        return {"window_strategy": "non_overlapping", "window_size": 4, "stride": 4}
    if window_mode == "win6_s3":
        return {"window_strategy": "sliding", "window_size": 6, "stride": 3}
    if window_mode == "win9_s9":
        return {"window_strategy": "non_overlapping", "window_size": 9, "stride": 9}
    if window_mode == "win12_s6":
        return {"window_strategy": "sliding", "window_size": 12, "stride": 6}
    raise ValueError(f"unsupported window_mode={window_mode!r}")


def _definition(
    *,
    pair_id: str,
    block: str,
    transform: str,
    feature_protocol: str,
    wavelet: str | None = None,
    level: int | None = None,
    threshold: str | None = None,
    classifier: str | None = None,
    sequence_model: str | None = None,
    window_mode: str = "global",
    cwt_scales: list[float] | None = None,
) -> dict[str, Any]:
    if threshold is None:
        threshold = "not_applicable" if transform in {"cwt", "wpt"} else "none"
    row: dict[str, Any] = {
        "pair_id": pair_id,
        "block": block,
        "transform": transform,
        "feature_protocol": feature_protocol,
        "threshold": threshold,
        "classifier": classifier,
        "sequence_model": sequence_model,
        "window_mode": window_mode,
    }
    if wavelet is not None:
        row["wavelet"] = wavelet
    if level is not None:
        row["level"] = level
    if cwt_scales is not None:
        row["cwt_scales"] = list(cwt_scales)
    row.update(_window_fields(window_mode))
    return row


def _block_a_definitions() -> tuple[dict[str, Any], ...]:
    wavelets = (
        ("dwt", "haar", 1),
        ("dwt", "db2", 2),
        ("dwt", "db4", 2),
        ("dwt", "db6", 2),
        ("dwt", "sym2", 2),
        ("dwt", "sym4", 2),
        ("dwt", "sym6", 2),
        ("dwt", "coif1", 2),
        ("dwt", "coif2", 2),
        ("dwt", "bior2.2", 2),
        ("dwt", "bior4.4", 2),
        ("swt", "haar", 2),
        ("swt", "db2", 2),
        ("swt", "sym4", 2),
        ("wpt", "db2", 2),
        ("wpt", "sym4", 2),
    )
    rows = [
        _definition(
            pair_id="A_none",
            block="A",
            transform="none",
            feature_protocol="wavelet_summary_static_pooled",
            classifier="logreg",
        )
    ]
    rows.extend(
        _definition(
            pair_id=f"A_{transform}_{wavelet}_l{level}",
            block="A",
            transform=transform,
            feature_protocol="wavelet_summary_static_pooled",
            wavelet=wavelet,
            level=level,
            threshold="universal_soft" if transform in {"dwt", "swt"} else None,
            classifier="logreg",
        )
        for transform, wavelet, level in wavelets
    )
    rows.extend(
        [
            _definition(
                pair_id="A_cwt_morl_scales_1_16",
                block="A",
                transform="cwt",
                feature_protocol="wavelet_summary_static_pooled",
                wavelet="morl",
                cwt_scales=CWT_SCALES_1_16,
                classifier="logreg",
            ),
            _definition(
                pair_id="A_cwt_mexh_scales_1_16",
                block="A",
                transform="cwt",
                feature_protocol="wavelet_summary_static_pooled",
                wavelet="mexh",
                cwt_scales=CWT_SCALES_1_16,
                classifier="logreg",
            ),
        ]
    )
    return tuple(rows)


def _block_b_definitions() -> tuple[dict[str, Any], ...]:
    rows = [
        _definition(
            pair_id="B_direct_raw_sequence",
            block="B",
            transform="none",
            feature_protocol="raw_sequence",
            sequence_model="lstm_projected",
        )
    ]
    rows.extend(
        _definition(
            pair_id=f"B_{window_mode}_window_stat28_sequence_lstm_projected",
            block="B",
            transform="swt",
            feature_protocol="window_stat28_sequence",
            wavelet="db2",
            level=2,
            threshold="universal_soft",
            sequence_model="lstm_projected",
            window_mode=window_mode,
        )
        for window_mode in ("global", "win4_s4", "win6_s3", "win9_s9", "win12_s6")
    )
    return tuple(rows)


def _block_c_definitions() -> tuple[dict[str, Any], ...]:
    static_rows = [
        _definition(
            pair_id=f"C_wavelet_summary_static_pooled_{classifier}",
            block="C",
            transform="swt",
            feature_protocol="wavelet_summary_static_pooled",
            wavelet="db2",
            level=2,
            threshold="universal_soft",
            classifier=classifier,
        )
        for classifier in ("logreg", "linear_svm", "rf", "extra_trees", "xgboost")
    ]
    sequence_rows = [
        _definition(
            pair_id=f"C_raw_sequence_{sequence_model}",
            block="C",
            transform="swt",
            feature_protocol="raw_sequence",
            wavelet="db2",
            level=2,
            threshold="universal_soft",
            sequence_model=sequence_model,
        )
        for sequence_model in ("lstm_projected", "gru_projected", "tcn", "cnn1d")
    ]
    return (*static_rows, *sequence_rows)


def _block_d_definitions() -> tuple[dict[str, Any], ...]:
    return tuple(
        _definition(
            pair_id=f"D_dwt_db2_l2_threshold_{threshold}",
            block="D",
            transform="dwt",
            feature_protocol="wavelet_summary_static_pooled",
            wavelet="db2",
            level=2,
            threshold=threshold,
            classifier="logreg",
        )
        for threshold in ("none", "universal_soft", "universal_hard", "sure_soft")
    )


def _block_e_definitions() -> tuple[dict[str, Any], ...]:
    return tuple(
        _definition(
            pair_id=f"E_{window_mode}_window_stat28_static_pooled_{classifier}",
            block="E",
            transform="dwt",
            feature_protocol="window_stat28_static_pooled",
            wavelet="db2",
            level=1,
            threshold="universal_soft",
            classifier=classifier,
            window_mode=window_mode,
        )
        for window_mode in ("win4_s4", "win9_s9", "global")
        for classifier in ("logreg", "rf", "xgboost")
    )


PAIR_DEFINITIONS: tuple[dict[str, Any], ...] = (
    *_block_a_definitions(),
    *_block_b_definitions(),
    *_block_c_definitions(),
    *_block_d_definitions(),
    *_block_e_definitions(),
)


def build_paired_grid(*, blocks: Sequence[str] | None = None) -> tuple[PairSpec, ...]:
    """Return exact Teacher/Ours rows for the paired-wavelet v2 A-E grid."""

    block_filter = set(PAIR_BLOCKS if blocks is None else blocks)
    unknown = sorted(block_filter - set(PAIR_BLOCKS))
    if unknown:
        raise ValueError(f"unknown paired grid blocks: {unknown}")
    rows: list[PairSpec] = []
    for definition in PAIR_DEFINITIONS:
        if definition["block"] not in block_filter:
            continue
        rows.extend(_paired_rows(definition))
    return tuple(rows)


def build_paired_run_spec(run_id: str = "paired_wavelet_v2") -> PairedRunSpec:
    """Return a strict run spec for the full paired v2 grid."""

    pairs = build_paired_grid()
    assert_paired_grid_complete(pairs)
    return PairedRunSpec(run_id=run_id, pairs=pairs)


def paired_grid_as_dicts(*, blocks: Sequence[str] | None = None) -> list[dict[str, Any]]:
    return [row.as_dict() for row in build_paired_grid(blocks=blocks)]


def assert_paired_grid_complete(
    rows: Iterable[PairSpec],
    *,
    expected_blocks: Sequence[str] = PAIR_BLOCKS,
    expected_sources: Sequence[str] = PAIR_SOURCES,
) -> tuple[PairSpec, ...]:
    """Fail unless every pair has exact Teacher/Ours rows and all blocks exist."""

    pairs = tuple(rows)
    if not pairs:
        raise ValueError("paired grid must not be empty")
    if not all(isinstance(row, PairSpec) for row in pairs):
        raise ValueError("paired grid rows must be PairSpec instances")

    expected_block_set = set(expected_blocks)
    expected_source_set = set(expected_sources)
    if expected_block_set - set(PAIR_BLOCKS):
        raise ValueError(f"unsupported expected_blocks: {sorted(expected_block_set - set(PAIR_BLOCKS))}")
    if expected_source_set - set(PAIR_SOURCES):
        raise ValueError(f"unsupported expected_sources: {sorted(expected_source_set - set(PAIR_SOURCES))}")

    duplicate_keys: set[tuple[str, str]] = set()
    grouped: dict[str, list[PairSpec]] = defaultdict(list)
    for row in pairs:
        key = (row.pair_id, row.source)
        if key in duplicate_keys:
            raise ValueError(f"duplicate paired grid row for pair_id={row.pair_id!r}, source={row.source!r}")
        duplicate_keys.add(key)
        grouped[row.pair_id].append(row)

    present_blocks = {row.block for row in pairs}
    missing_blocks = sorted(expected_block_set - present_blocks)
    if missing_blocks:
        raise ValueError(f"paired grid missing blocks: {missing_blocks}")

    for pair_id, pair_rows in grouped.items():
        sources = {row.source for row in pair_rows}
        if sources != expected_source_set:
            raise ValueError(
                f"pair_id={pair_id!r} must have exact sources {sorted(expected_source_set)}, "
                f"got {sorted(sources)}"
            )
        if len(pair_rows) != len(expected_source_set):
            raise ValueError(f"pair_id={pair_id!r} has duplicate or extra source rows")
        keys = {row.paired_key() for row in pair_rows}
        if len(keys) != 1:
            raise ValueError(f"pair_id={pair_id!r} Teacher/Ours rows are not config-matched")

    default_pair_ids = {definition["pair_id"] for definition in PAIR_DEFINITIONS if definition["block"] in expected_block_set}
    present_pair_ids = set(grouped)
    missing_pairs = sorted(default_pair_ids - present_pair_ids)
    extra_pairs = sorted(present_pair_ids - default_pair_ids)
    if missing_pairs:
        raise ValueError(f"paired grid missing pair_ids: {missing_pairs}")
    if extra_pairs:
        raise ValueError(f"paired grid contains unknown pair_ids: {extra_pairs}")
    return pairs


def _paired_rows(definition: dict[str, Any]) -> tuple[PairSpec, PairSpec]:
    common = dict(definition)
    teacher = PairSpec(source="Teacher", signal_builder=TEACHER_SIGNAL_BUILDER, **common)
    ours = PairSpec(source="Ours", signal_builder=OURS_SIGNAL_BUILDER, **common)
    return teacher, ours


__all__ = [
    "CWT_SCALES_1_16",
    "PAIR_DEFINITIONS",
    "assert_paired_grid_complete",
    "build_paired_grid",
    "build_paired_run_spec",
    "paired_grid_as_dicts",
]
