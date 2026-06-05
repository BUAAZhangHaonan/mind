from __future__ import annotations

from collections import Counter

import pytest


EXPECTED_PAIR_IDS_BY_BLOCK = {
    "A": [
        "A_none",
        "A_dwt_haar_l1",
        "A_dwt_db2_l2",
        "A_dwt_db4_l2",
        "A_dwt_db6_l2",
        "A_dwt_sym2_l2",
        "A_dwt_sym4_l2",
        "A_dwt_sym6_l2",
        "A_dwt_coif1_l2",
        "A_dwt_coif2_l2",
        "A_dwt_bior2.2_l2",
        "A_dwt_bior4.4_l2",
        "A_swt_haar_l2",
        "A_swt_db2_l2",
        "A_swt_sym4_l2",
        "A_wpt_db2_l2",
        "A_wpt_sym4_l2",
        "A_cwt_morl_scales_1_16",
        "A_cwt_mexh_scales_1_16",
    ],
    "B": [
        "B_direct_raw_sequence",
        "B_global_window_stat28_sequence_lstm_projected",
        "B_win4_s4_window_stat28_sequence_lstm_projected",
        "B_win6_s3_window_stat28_sequence_lstm_projected",
        "B_win9_s9_window_stat28_sequence_lstm_projected",
        "B_win12_s6_window_stat28_sequence_lstm_projected",
    ],
    "C": [
        "C_wavelet_summary_static_pooled_logreg",
        "C_wavelet_summary_static_pooled_linear_svm",
        "C_wavelet_summary_static_pooled_rf",
        "C_wavelet_summary_static_pooled_extra_trees",
        "C_wavelet_summary_static_pooled_xgboost",
        "C_raw_sequence_lstm_projected",
        "C_raw_sequence_gru_projected",
        "C_raw_sequence_tcn",
        "C_raw_sequence_cnn1d",
    ],
    "D": [
        "D_dwt_db2_l2_threshold_none",
        "D_dwt_db2_l2_threshold_universal_soft",
        "D_dwt_db2_l2_threshold_universal_hard",
        "D_dwt_db2_l2_threshold_sure_soft",
    ],
    "E": [
        "E_win4_s4_window_stat28_static_pooled_logreg",
        "E_win4_s4_window_stat28_static_pooled_rf",
        "E_win4_s4_window_stat28_static_pooled_xgboost",
        "E_win9_s9_window_stat28_static_pooled_logreg",
        "E_win9_s9_window_stat28_static_pooled_rf",
        "E_win9_s9_window_stat28_static_pooled_xgboost",
        "E_global_window_stat28_static_pooled_logreg",
        "E_global_window_stat28_static_pooled_rf",
        "E_global_window_stat28_static_pooled_xgboost",
    ],
}

EXPECTED_PAIR_IDS = [
    pair_id
    for block_pair_ids in EXPECTED_PAIR_IDS_BY_BLOCK.values()
    for pair_id in block_pair_ids
]

DEFAULT_CWT_SCALES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0]
CWT_SCALES_1_16 = DEFAULT_CWT_SCALES

WINDOW_FIELDS_BY_MODE = {
    "global": {"window_strategy": "full", "window_size": None, "stride": None},
    "win4_s4": {"window_strategy": "non_overlapping", "window_size": 4, "stride": 4},
    "win6_s3": {"window_strategy": "sliding", "window_size": 6, "stride": 3},
    "win9_s9": {"window_strategy": "non_overlapping", "window_size": 9, "stride": 9},
    "win12_s6": {"window_strategy": "sliding", "window_size": 12, "stride": 6},
}


def _expected_pair_field_specs() -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}

    def add(
        pair_id: str,
        *,
        block: str,
        transform: str,
        feature_protocol: str,
        wavelet: str | None = None,
        level: int | None = None,
        threshold: str,
        classifier: str | None = None,
        sequence_model: str | None = None,
        window_mode: str = "global",
        cwt_scales: list[float] | None = None,
    ) -> None:
        specs[pair_id] = {
            "pair_id": pair_id,
            "block": block,
            "transform": transform,
            "feature_protocol": feature_protocol,
            "wavelet": wavelet,
            "level": level,
            **WINDOW_FIELDS_BY_MODE[window_mode],
            "mode": "symmetric",
            "cwt_scales": cwt_scales if cwt_scales is not None else DEFAULT_CWT_SCALES,
            "threshold": threshold,
            "classifier": classifier,
            "sequence_model": sequence_model,
            "window_mode": window_mode,
        }

    add(
        "A_none",
        block="A",
        transform="none",
        feature_protocol="wavelet_summary_static_pooled",
        threshold="none",
        classifier="logreg",
    )
    for transform, wavelet, level in (
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
    ):
        add(
            f"A_{transform}_{wavelet}_l{level}",
            block="A",
            transform=transform,
            feature_protocol="wavelet_summary_static_pooled",
            wavelet=wavelet,
            level=level,
            threshold="universal_soft",
            classifier="logreg",
        )
    for transform, wavelet in (("wpt", "db2"), ("wpt", "sym4")):
        add(
            f"A_{transform}_{wavelet}_l2",
            block="A",
            transform=transform,
            feature_protocol="wavelet_summary_static_pooled",
            wavelet=wavelet,
            level=2,
            threshold="not_applicable",
            classifier="logreg",
        )
    for wavelet in ("morl", "mexh"):
        add(
            f"A_cwt_{wavelet}_scales_1_16",
            block="A",
            transform="cwt",
            feature_protocol="wavelet_summary_static_pooled",
            wavelet=wavelet,
            threshold="not_applicable",
            classifier="logreg",
            cwt_scales=CWT_SCALES_1_16,
        )

    add(
        "B_direct_raw_sequence",
        block="B",
        transform="none",
        feature_protocol="raw_sequence",
        threshold="none",
        sequence_model="lstm_projected",
    )
    for window_mode in ("global", "win4_s4", "win6_s3", "win9_s9", "win12_s6"):
        add(
            f"B_{window_mode}_window_stat28_sequence_lstm_projected",
            block="B",
            transform="swt",
            feature_protocol="window_stat28_sequence",
            wavelet="db2",
            level=2,
            threshold="universal_soft",
            sequence_model="lstm_projected",
            window_mode=window_mode,
        )

    for classifier in ("logreg", "linear_svm", "rf", "extra_trees", "xgboost"):
        add(
            f"C_wavelet_summary_static_pooled_{classifier}",
            block="C",
            transform="swt",
            feature_protocol="wavelet_summary_static_pooled",
            wavelet="db2",
            level=2,
            threshold="universal_soft",
            classifier=classifier,
        )
    for sequence_model in ("lstm_projected", "gru_projected", "tcn", "cnn1d"):
        add(
            f"C_raw_sequence_{sequence_model}",
            block="C",
            transform="swt",
            feature_protocol="raw_sequence",
            wavelet="db2",
            level=2,
            threshold="universal_soft",
            sequence_model=sequence_model,
        )

    for threshold in ("none", "universal_soft", "universal_hard", "sure_soft"):
        add(
            f"D_dwt_db2_l2_threshold_{threshold}",
            block="D",
            transform="dwt",
            feature_protocol="wavelet_summary_static_pooled",
            wavelet="db2",
            level=2,
            threshold=threshold,
            classifier="logreg",
        )

    for window_mode in ("win4_s4", "win9_s9", "global"):
        for classifier in ("logreg", "rf", "xgboost"):
            add(
                f"E_{window_mode}_window_stat28_static_pooled_{classifier}",
                block="E",
                transform="dwt",
                feature_protocol="window_stat28_static_pooled",
                wavelet="db2",
                level=1,
                threshold="universal_soft",
                classifier=classifier,
                window_mode=window_mode,
            )

    return specs


def test_paired_grid_is_complete_and_config_matched() -> None:
    from mind.wavelet_course.paired_config import (
        OURS_SIGNAL_BUILDER,
        PAIR_BLOCKS,
        TEACHER_SIGNAL_BUILDER,
    )
    from mind.wavelet_course.paired_grid import (
        assert_paired_grid_complete,
        build_paired_grid,
        paired_grid_as_dicts,
    )

    rows = assert_paired_grid_complete(build_paired_grid())

    assert len(rows) == 94
    assert len({row.pair_id for row in rows}) == 47
    assert {row.block for row in rows} == set(PAIR_BLOCKS)
    assert {row.source for row in rows} == {"Teacher", "Ours"}
    assert [row.pair_id for row in rows[::2]] == EXPECTED_PAIR_IDS
    assert Counter(row.block for row in rows[::2]) == {
        "A": 19,
        "B": 6,
        "C": 9,
        "D": 4,
        "E": 9,
    }

    for pair_id in EXPECTED_PAIR_IDS:
        pair_rows = [row for row in rows if row.pair_id == pair_id]
        assert [row.source for row in pair_rows] == ["Teacher", "Ours"]
        assert len({row.paired_key() for row in pair_rows}) == 1
        assert pair_rows[0].signal_builder == TEACHER_SIGNAL_BUILDER
        assert pair_rows[1].signal_builder == OURS_SIGNAL_BUILDER
        teacher = pair_rows[0].as_dict()
        ours = pair_rows[1].as_dict()
        for field in {"row_id", "source", "signal_builder"}:
            teacher.pop(field)
            ours.pop(field)
        assert teacher == ours

    as_dicts = paired_grid_as_dicts()
    assert as_dicts[0]["row_id"] == "A_none::Teacher"
    assert as_dicts[-1]["row_id"] == "E_global_window_stat28_static_pooled_xgboost::Ours"


def test_paired_grid_exact_fields_for_every_pair_spec() -> None:
    from mind.wavelet_course.paired_config import (
        OURS_SIGNAL_BUILDER,
        TEACHER_SIGNAL_BUILDER,
    )
    from mind.wavelet_course.paired_grid import paired_grid_as_dicts

    expected_pairs = _expected_pair_field_specs()
    rows = paired_grid_as_dicts()

    assert set(expected_pairs) == set(EXPECTED_PAIR_IDS)
    assert len(rows) == 2 * len(expected_pairs)

    for row in rows:
        source = row["source"]
        signal_builder = {
            "Teacher": TEACHER_SIGNAL_BUILDER,
            "Ours": OURS_SIGNAL_BUILDER,
        }[source]
        expected = {
            **expected_pairs[row["pair_id"]],
            "row_id": f"{row['pair_id']}::{source}",
            "source": source,
            "signal_builder": signal_builder,
        }
        assert row == expected


def test_paired_grid_representative_block_fields() -> None:
    from mind.wavelet_course.paired_grid import paired_grid_as_dicts

    rows = {
        row["pair_id"]: row
        for row in paired_grid_as_dicts()
        if row["source"] == "Teacher"
    }

    assert {
        "block": "A",
        "transform": "none",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": None,
        "level": None,
        "threshold": "none",
        "classifier": "logreg",
    }.items() <= rows["A_none"].items()
    assert rows["A_cwt_morl_scales_1_16"]["cwt_scales"] == CWT_SCALES_1_16
    assert rows["A_cwt_mexh_scales_1_16"]["cwt_scales"] == CWT_SCALES_1_16
    assert {
        "block": "A",
        "transform": "dwt",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "db6",
        "level": 2,
        "threshold": "universal_soft",
        "window_mode": "global",
        "window_strategy": "full",
        "classifier": "logreg",
        "sequence_model": None,
    }.items() <= rows["A_dwt_db6_l2"].items()
    assert {
        "block": "A",
        "transform": "wpt",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "db2",
        "level": 2,
        "threshold": "not_applicable",
        "classifier": "logreg",
    }.items() <= rows["A_wpt_db2_l2"].items()
    assert {
        "block": "A",
        "transform": "cwt",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "morl",
        "level": None,
        "threshold": "not_applicable",
        "classifier": "logreg",
    }.items() <= rows["A_cwt_morl_scales_1_16"].items()
    assert {
        "block": "B",
        "transform": "swt",
        "feature_protocol": "window_stat28_sequence",
        "wavelet": "db2",
        "level": 2,
        "threshold": "universal_soft",
        "window_mode": "win6_s3",
        "window_strategy": "sliding",
        "window_size": 6,
        "stride": 3,
        "classifier": None,
        "sequence_model": "lstm_projected",
    }.items() <= rows["B_win6_s3_window_stat28_sequence_lstm_projected"].items()
    assert {
        "block": "B",
        "transform": "none",
        "feature_protocol": "raw_sequence",
        "threshold": "none",
        "window_mode": "global",
        "sequence_model": "lstm_projected",
    }.items() <= rows["B_direct_raw_sequence"].items()
    assert {
        "block": "C",
        "transform": "swt",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "db2",
        "level": 2,
        "threshold": "universal_soft",
        "classifier": "extra_trees",
        "sequence_model": None,
        "window_mode": "global",
        "window_strategy": "full",
    }.items() <= rows["C_wavelet_summary_static_pooled_extra_trees"].items()
    assert {
        "block": "C",
        "transform": "swt",
        "feature_protocol": "raw_sequence",
        "wavelet": "db2",
        "level": 2,
        "threshold": "universal_soft",
        "classifier": None,
        "sequence_model": "tcn",
        "window_mode": "global",
        "window_strategy": "full",
    }.items() <= rows["C_raw_sequence_tcn"].items()
    assert {
        "block": "D",
        "transform": "dwt",
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "db2",
        "level": 2,
        "threshold": "universal_hard",
        "classifier": "logreg",
        "window_mode": "global",
    }.items() <= rows["D_dwt_db2_l2_threshold_universal_hard"].items()
    assert {
        "block": "E",
        "transform": "dwt",
        "feature_protocol": "window_stat28_static_pooled",
        "wavelet": "db2",
        "level": 1,
        "threshold": "universal_soft",
        "classifier": "rf",
        "sequence_model": None,
        "window_mode": "win4_s4",
        "window_strategy": "non_overlapping",
        "window_size": 4,
        "stride": 4,
    }.items() <= rows["E_win4_s4_window_stat28_static_pooled_rf"].items()


def test_pair_spec_accepts_required_feature_protocol_aliases() -> None:
    from mind.wavelet_course.paired_config import PairSpec, SUPPORTED_FEATURE_PROTOCOLS

    required_protocols = {
        "raw_sequence",
        "window_stat28_sequence",
        "window_stat28_static_flat",
        "window_stat28_static_pooled",
        "wavelet_summary_static_pooled",
    }

    assert required_protocols <= set(SUPPORTED_FEATURE_PROTOCOLS)
    for protocol in required_protocols:
        pair = PairSpec(
            pair_id=f"alias_{protocol}",
            block="B",
            source="Teacher",
            signal_builder="teacher_hidden_dim_signal",
            transform="dwt",
            feature_protocol=protocol,
            wavelet="db2",
            level=1,
            threshold="none",
        )

        assert pair.feature_protocol == protocol


@pytest.mark.parametrize("transform", ["cwt", "wpt"])
def test_pair_spec_cwt_and_wpt_threshold_must_be_not_applicable(transform: str) -> None:
    from mind.wavelet_course.paired_config import PairSpec

    kwargs = {
        "pair_id": f"{transform}_threshold",
        "block": "A",
        "source": "Teacher",
        "signal_builder": "teacher_hidden_dim_signal",
        "transform": transform,
        "feature_protocol": "wavelet_summary_static_pooled",
        "wavelet": "morl" if transform == "cwt" else "db2",
        "threshold": "not_applicable",
    }
    if transform == "wpt":
        kwargs["level"] = 2
    pair = PairSpec(**kwargs)

    assert pair.threshold == "not_applicable"

    with pytest.raises(ValueError, match="not_applicable"):
        PairSpec(**{**kwargs, "threshold": "universal_soft"})


def test_paired_grid_filters_blocks_and_rejects_unknown_blocks() -> None:
    from mind.wavelet_course.paired_grid import (
        assert_paired_grid_complete,
        build_paired_grid,
    )

    rows = assert_paired_grid_complete(
        build_paired_grid(blocks=("B",)),
        expected_blocks=("B",),
    )

    assert len(rows) == 12
    assert {row.block for row in rows} == {"B"}
    assert [row.pair_id for row in rows[::2]] == EXPECTED_PAIR_IDS_BY_BLOCK["B"]

    with pytest.raises(ValueError, match="unknown paired grid blocks"):
        build_paired_grid(blocks=("Z",))


def test_paired_grid_config_file_matches_code_grid() -> None:
    from pathlib import Path

    import yaml

    from mind.wavelet_course.paired_grid import PAIR_DEFINITIONS, paired_grid_as_dicts

    config = yaml.safe_load(
        Path("configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml").read_text(
            encoding="utf-8"
        )
    )

    yaml_definitions = config["paired_wavelet_v2"]["pair_definitions"]
    assert yaml_definitions == [dict(definition) for definition in PAIR_DEFINITIONS]
    assert config["paired_wavelet_v2"]["num_pair_ids"] == 47
    assert config["paired_wavelet_v2"]["num_pair_rows"] == 94
    assert [row["pair_id"] for row in paired_grid_as_dicts()[::2]] == [
        definition["pair_id"] for definition in yaml_definitions
    ]


def test_paired_grid_quick_mode_limits_pair_ids_after_block_filter() -> None:
    from mind.wavelet_course import paired_runner

    rows = paired_runner._resolve_pairs(
        {
            "quick_run": True,
            "paired_wavelet_v2": {
                "blocks": ["A", "B"],
                "expected_sources": ["Teacher", "Ours"],
            },
            "quick": {
                "blocks": ["B"],
                "max_pair_ids": 2,
            },
        },
        quick_run=True,
    )

    assert len(rows) == 4
    assert [row.pair_id for row in rows[::2]] == EXPECTED_PAIR_IDS_BY_BLOCK["B"][:2]
    assert {row.block for row in rows} == {"B"}
