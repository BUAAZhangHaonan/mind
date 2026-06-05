from __future__ import annotations

from itertools import combinations
import sys
import time

import numpy as np
import pytest


def test_wavelet_features_are_computed_along_final_layer_axis() -> None:
    from mind.wavelet_course.common_wavelet import apply_wavelet

    signal = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)

    coeffs = apply_wavelet(
        signal,
        {"transform": "dwt", "wavelet": "haar", "level": 1},
    )

    assert coeffs.shape == (2, 8)
    assert coeffs.dtype == np.float32
    assert not np.array_equal(coeffs, signal)


def test_swt_level_three_fails_for_length_36() -> None:
    from mind.wavelet_course.common_wavelet import (
        SWTLevelInfeasibleError,
        apply_wavelet,
    )

    with pytest.raises(SWTLevelInfeasibleError, match="SWT level 3.*36"):
        apply_wavelet(
            np.zeros((2, 36), dtype=np.float32),
            {"transform": "swt", "wavelet": "db2", "level": 3},
        )


@pytest.mark.parametrize("transform", ["dwt", "swt"])
def test_dwt_and_swt_thresholds_apply_distinct_outputs(transform: str) -> None:
    from mind.wavelet_course.common_wavelet import apply_wavelet

    rng = np.random.default_rng(7)
    base_signal = (0.01 * rng.normal(size=64)).astype(np.float32)
    base_signal[[7, 23, 41, 52]] += np.asarray([3.0, -2.5, 2.25, -1.75], dtype=np.float32)
    signal = np.vstack([base_signal, 0.75 * base_signal + 0.02 * np.roll(base_signal, 3)]).astype(np.float32)
    base_spec = {
        "transform": transform,
        "wavelet": "db2",
        "level": 2,
    }

    outputs = {
        threshold: apply_wavelet(signal, {**base_spec, "threshold": threshold})
        for threshold in ("none", "universal_soft", "universal_hard", "sure_soft")
    }

    assert {output.shape for output in outputs.values()} == {outputs["none"].shape}
    assert all(np.isfinite(output).all() for output in outputs.values())
    for left, right in combinations(outputs, 2):
        assert not np.allclose(outputs[left], outputs[right]), (transform, left, right)


@pytest.mark.parametrize(
    "spec",
    [
        {"transform": "cwt", "wavelet": "morl", "cwt_scales": [1.0, 2.0, 4.0]},
        {"transform": "wpt", "wavelet": "db2", "level": 2},
    ],
)
def test_cwt_and_wpt_require_not_applicable_threshold(spec: dict[str, object]) -> None:
    from mind.wavelet_course.common_wavelet import WaveletConfigError, apply_wavelet

    signal = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    with pytest.raises(WaveletConfigError, match="not_applicable"):
        apply_wavelet(signal, {**spec, "threshold": "universal_soft"})

    coeffs = apply_wavelet(signal, {**spec, "threshold": "not_applicable"})

    assert coeffs.dtype == np.float32
    assert np.isfinite(coeffs).all()


def test_wavelet_rejects_unsupported_threshold() -> None:
    from mind.wavelet_course.common_wavelet import WaveletConfigError, apply_wavelet

    with pytest.raises(WaveletConfigError, match="unsupported threshold"):
        apply_wavelet(
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            {
                "transform": "dwt",
                "wavelet": "db2",
                "level": 2,
                "threshold": "adaptive_magic",
            },
        )


def test_required_feature_protocol_aliases_map_to_exact_behaviors() -> None:
    from mind.wavelet_course.common_feature_protocols import (
        features_for_protocol,
        wavelet_summary_static_pooled_names,
    )

    signal = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    base_spec = {
        "transform": "dwt",
        "wavelet": "haar",
        "level": 1,
        "threshold": "none",
        "window_strategy": "non_overlapping",
        "window_size": 4,
    }

    raw = features_for_protocol(
        signal,
        {
            "transform": "none",
            "threshold": "none",
            "feature_protocol": "raw_sequence",
        },
    )
    sequence = features_for_protocol(signal, {**base_spec, "feature_protocol": "window_stat28_sequence"})
    flat = features_for_protocol(signal, {**base_spec, "feature_protocol": "window_stat28_static_flat"})
    pooled = features_for_protocol(signal, {**base_spec, "feature_protocol": "window_stat28_static_pooled"})
    summary = features_for_protocol(signal, {**base_spec, "feature_protocol": "wavelet_summary_static_pooled"})

    assert raw.shape == (8, 2)
    assert sequence.shape == (2, 56)
    assert flat.shape == (112,)
    np.testing.assert_array_equal(flat, sequence.reshape(-1))
    assert pooled.shape == (336,)
    assert summary.shape == (54,)
    assert len(wavelet_summary_static_pooled_names({**base_spec, "feature_protocol": "wavelet_summary_static_pooled"})) == 54
    assert all(np.isfinite(values).all() for values in (raw, sequence, flat, pooled, summary))


def test_wavelet_summary_static_pooled_uses_required_energy_protocol() -> None:
    from mind.wavelet_course.common_feature_protocols import (
        features_for_protocol,
        wavelet_summary_static_pooled_names,
    )

    signal = np.arange(4 * 36, dtype=np.float32).reshape(4, 36)
    spec = {
        "transform": "dwt",
        "wavelet": "haar",
        "level": 2,
        "threshold": "none",
        "feature_protocol": "wavelet_summary_static_pooled",
    }

    features = features_for_protocol(signal, spec)
    names = wavelet_summary_static_pooled_names(spec)

    assert features.shape == (60,)
    assert len(names) == features.shape[0]
    assert names[:10] == (
        "mean_approximation_energy",
        "mean_detail_energy_1",
        "mean_detail_energy_2",
        "mean_total_detail_energy",
        "mean_detail_approx_ratio",
        "mean_high_frequency_ratio",
        "mean_wavelet_entropy",
        "mean_max_abs_coefficient",
        "mean_energy_center",
        "mean_energy_spread",
    )
    assert names[-10:] == (
        "q90_approximation_energy",
        "q90_detail_energy_1",
        "q90_detail_energy_2",
        "q90_total_detail_energy",
        "q90_detail_approx_ratio",
        "q90_high_frequency_ratio",
        "q90_wavelet_entropy",
        "q90_max_abs_coefficient",
        "q90_energy_center",
        "q90_energy_spread",
    )
    assert np.isfinite(features).all()


def test_wavelet_summary_static_pooled_has_explicit_none_summary() -> None:
    from mind.wavelet_course.common_feature_protocols import (
        features_for_protocol,
        wavelet_summary_static_pooled_names,
    )

    signal = np.asarray(
        [
            np.linspace(-1.0, 1.0, 36),
            np.zeros(36),
            np.sin(np.linspace(0.0, np.pi, 36)),
        ],
        dtype=np.float32,
    )
    spec = {
        "transform": "none",
        "threshold": "none",
        "feature_protocol": "wavelet_summary_static_pooled",
    }

    features = features_for_protocol(signal, spec)
    names = wavelet_summary_static_pooled_names(spec)

    assert features.shape == (48,)
    assert len(names) == 48
    assert "mean_approximation_energy" in names
    assert "mean_total_detail_energy" in names
    assert "q90_energy_spread" in names
    assert np.isfinite(features).all()


def test_wavelet_summary_static_pooled_supports_wpt_and_cwt_names() -> None:
    from mind.wavelet_course.common_feature_protocols import (
        features_for_protocol,
        wavelet_summary_static_pooled_names,
    )

    signal = np.arange(3 * 36, dtype=np.float32).reshape(3, 36)
    wpt_spec = {
        "transform": "wpt",
        "wavelet": "haar",
        "level": 2,
        "threshold": "not_applicable",
        "feature_protocol": "wavelet_summary_static_pooled",
    }
    cwt_spec = {
        "transform": "cwt",
        "wavelet": "mexh",
        "cwt_scales": [1.0, 2.0, 4.0],
        "threshold": "not_applicable",
        "feature_protocol": "wavelet_summary_static_pooled",
    }

    wpt_features = features_for_protocol(signal, wpt_spec)
    cwt_features = features_for_protocol(signal, cwt_spec)

    assert wpt_features.shape == (66,)
    assert cwt_features.shape == (60,)
    assert len(wavelet_summary_static_pooled_names(wpt_spec)) == 66
    assert len(wavelet_summary_static_pooled_names(cwt_spec)) == 60
    assert np.isfinite(wpt_features).all()
    assert np.isfinite(cwt_features).all()


def test_wavelet_summary_wpt_uses_batched_dwt_not_per_channel_wavelet_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywt
    from mind.wavelet_course.common_feature_protocols import features_for_protocol

    def fail_wavelet_packet(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("WPT summary must not instantiate WaveletPacket per channel")

    monkeypatch.setattr(pywt, "WaveletPacket", fail_wavelet_packet)
    signal = np.arange(4096 * 36, dtype=np.float32).reshape(4096, 36)
    features = features_for_protocol(
        signal,
        {
            "transform": "wpt",
            "wavelet": "db2",
            "level": 2,
            "threshold": "not_applicable",
            "feature_protocol": "wavelet_summary_static_pooled",
        },
    )

    assert features.shape == (66,)
    assert np.isfinite(features).all()


def test_wavelet_summary_cwt_uses_single_axis_batched_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pywt
    from mind.wavelet_course.common_feature_protocols import features_for_protocol

    calls: list[tuple[tuple[int, ...], int]] = []

    def counting_cwt(data: object, scales: object, wavelet: object, *args: object, **kwargs: object):
        array = np.asarray(data)
        scale_array = np.asarray(scales, dtype=np.float64)
        calls.append((array.shape, int(kwargs.get("axis", -999))))
        if array.ndim != 2:
            coeff_shape = (scale_array.shape[0], array.shape[-1])
        else:
            coeff_shape = (scale_array.shape[0], *array.shape)
        return np.ones(coeff_shape, dtype=np.float64), np.ones(scale_array.shape[0], dtype=np.float64)

    monkeypatch.setattr(pywt, "cwt", counting_cwt)
    signal = np.arange(4096 * 36, dtype=np.float32).reshape(4096, 36)
    features = features_for_protocol(
        signal,
        {
            "transform": "cwt",
            "wavelet": "mexh",
            "cwt_scales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0],
            "threshold": "not_applicable",
            "feature_protocol": "wavelet_summary_static_pooled",
        },
    )

    assert calls == [((4096, 36), -1)]
    assert features.shape == (102,)
    assert np.isfinite(features).all()


@pytest.mark.parametrize(
    ("spec", "expected_shape"),
    [
        (
            {
                "transform": "wpt",
                "wavelet": "db2",
                "level": 2,
                "threshold": "not_applicable",
                "feature_protocol": "wavelet_summary_static_pooled",
            },
            (66,),
        ),
        (
            {
                "transform": "cwt",
                "wavelet": "mexh",
                "cwt_scales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0],
                "threshold": "not_applicable",
                "feature_protocol": "wavelet_summary_static_pooled",
            },
            (102,),
        ),
    ],
)
def test_teacher_wpt_and_cwt_single_sample_summary_runtime(
    spec: dict[str, object],
    expected_shape: tuple[int, ...],
) -> None:
    from mind.wavelet_course.common_feature_protocols import features_for_protocol

    rng = np.random.default_rng(20260506)
    signal = rng.normal(size=(4096, 36)).astype(np.float32)

    start = time.perf_counter()
    features = features_for_protocol(signal, spec)
    elapsed = time.perf_counter() - start

    assert features.shape == expected_shape
    assert np.isfinite(features).all()
    assert elapsed < 2.5


def test_feature_protocols_fail_closed_on_non_finite_input() -> None:
    from mind.wavelet_course.common_feature_protocols import features_for_protocol

    signal = np.ones((2, 36), dtype=np.float32)
    signal[0, 3] = np.nan

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        features_for_protocol(
            signal,
            {
                "transform": "none",
                "threshold": "none",
                "feature_protocol": "wavelet_summary_static_pooled",
            },
        )


def test_default_cwt_scales_match_v2_controlled_grid() -> None:
    from mind.wavelet_course.common_wavelet import DEFAULT_CWT_SCALES, apply_wavelet
    from mind.wavelet_course.paired_config import DEFAULT_CWT_SCALES as CONFIG_DEFAULT_CWT_SCALES
    from mind.wavelet_course.paired_grid import CWT_SCALES_1_16

    expected = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0)
    signal = np.arange(72, dtype=np.float32).reshape(2, 36)

    transformed = apply_wavelet(signal, {"transform": "cwt", "wavelet": "mexh"})

    assert DEFAULT_CWT_SCALES == expected
    assert CONFIG_DEFAULT_CWT_SCALES == expected
    assert CWT_SCALES_1_16 == list(expected)
    assert transformed.shape == (2, 36 * len(expected))


def test_windowing_and_stat28_protocol_keep_teacher_and_ours_layer_parity() -> None:
    from mind.wavelet_course.common_feature_protocols import features_for_protocol
    from mind.wavelet_course.common_windowing import WindowSpec, window_signal

    layer_index_signal = np.arange(36, dtype=np.float32)
    teacher_signal = np.vstack([layer_index_signal, layer_index_signal + 100.0])
    ours_signal = np.vstack(
        [layer_index_signal, layer_index_signal + 10.0, layer_index_signal + 20.0]
    )

    teacher_windows = window_signal(
        teacher_signal,
        WindowSpec(strategy="non_overlapping", size=4),
    )
    ours_windows = window_signal(
        ours_signal,
        WindowSpec(strategy="non_overlapping", size=4),
    )
    teacher_features = features_for_protocol(
        teacher_signal,
        {
            "feature_protocol": "stat28",
            "window_strategy": "non_overlapping",
            "window_size": 4,
        },
    )

    assert teacher_windows.shape == (9, 2, 4)
    assert ours_windows.shape == (9, 3, 4)
    np.testing.assert_array_equal(teacher_windows[:, 0, :], ours_windows[:, 0, :])
    assert teacher_features.shape == (9 * 2 * 28,)
    assert np.isfinite(teacher_features).all()


def test_window_stat28_sequence_windows_original_layer_axis_before_wavelet() -> None:
    from mind.wavelet_course.common_feature_protocols import features_for_protocol

    spec = {
        "transform": "swt",
        "wavelet": "db2",
        "level": 2,
        "threshold": "none",
        "feature_protocol": "window_stat28_sequence",
        "window_strategy": "non_overlapping",
        "window_size": 4,
        "stride": 4,
    }
    teacher_signal = np.arange(4096 * 36, dtype=np.float32).reshape(4096, 36)
    ours_signal = np.arange(10 * 36, dtype=np.float32).reshape(10, 36)

    teacher_features = features_for_protocol(teacher_signal, spec)
    ours_features = features_for_protocol(ours_signal, spec)

    assert teacher_features.shape == (9, 4096 * 28)
    assert ours_features.shape == (9, 10 * 28)
    assert np.isfinite(teacher_features).all()
    assert np.isfinite(ours_features).all()


def test_raw_sequence_swt_keeps_layer_depth_as_time_axis() -> None:
    from mind.wavelet_course.paired_config import TEACHER_SIGNAL_BUILDER, PairSpec
    from mind.wavelet_course.paired_runner import _sequence_features_for_signal

    pair = PairSpec(
        pair_id="C_raw_sequence_lstm_projected",
        block="C",
        source="Teacher",
        signal_builder=TEACHER_SIGNAL_BUILDER,
        transform="swt",
        wavelet="db2",
        level=2,
        threshold="none",
        feature_protocol="raw_sequence",
        sequence_model="lstm_projected",
    )
    signal = np.arange(4 * 36, dtype=np.float32).reshape(4, 36)

    sequence = _sequence_features_for_signal(signal, pair)

    assert sequence.shape == (36, 4 * 4)
    assert np.isfinite(sequence).all()


def test_static_classifiers_fail_closed_and_report_missing_xgboost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.common_classifiers import (
        XGBOOST_NOT_INSTALLED,
        train_static_classifier,
        xgboost_missing_failure_rows,
    )

    features = np.asarray([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        train_static_classifier("logreg", features, labels, random_state=7)

    with pytest.raises(ValueError, match="at least two classes"):
        train_static_classifier(
            "logreg",
            np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32),
            np.asarray([1, 1, 1], dtype=np.int64),
            random_state=7,
        )

    monkeypatch.setitem(sys.modules, "xgboost", None)
    result = train_static_classifier(
        "xgb",
        np.asarray([[0.0, 0.1], [1.0, 1.1], [0.2, 0.3], [1.2, 1.3]], dtype=np.float32),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        allow_missing_xgboost=True,
        random_state=7,
    )
    failure_rows = xgboost_missing_failure_rows(
        [
            {"pair_id": "teacher_haar_l1__ours_db2_l2", "classifier": "xgb"},
            {"pair_id": "teacher_db2_l1__ours_db2_l2", "classifier": "xgb"},
        ]
    )

    assert result.status == "failure"
    assert result.failure_reason == XGBOOST_NOT_INSTALLED
    assert [row["pair_id"] for row in failure_rows] == [
        "teacher_haar_l1__ours_db2_l2",
        "teacher_db2_l1__ours_db2_l2",
    ]
    assert all(row["status"] == "failure" for row in failure_rows)
    assert all(row["failure_reason"] == XGBOOST_NOT_INSTALLED for row in failure_rows)


def test_tree_classifiers_select_max_depth_by_validation_pr_auc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mind.wavelet_course.common_classifiers as common_classifiers

    constructed: list[dict[str, object]] = []

    class FakeForest:
        def __init__(self, **params: object) -> None:
            self.params = dict(params)
            constructed.append(self.params)

        def fit(self, x: np.ndarray, y: np.ndarray) -> "FakeForest":
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            depth = self.params.get("max_depth")
            if depth == 16:
                scores = np.asarray([0.05, 0.95, 0.10, 0.90], dtype=np.float32)
            elif depth == 8:
                scores = np.asarray([0.90, 0.20, 0.80, 0.10], dtype=np.float32)
            else:
                scores = np.asarray([0.60, 0.50, 0.40, 0.30], dtype=np.float32)
            scores = np.resize(scores, x.shape[0])
            return np.column_stack([1.0 - scores, scores])

    monkeypatch.setattr(common_classifiers, "RandomForestClassifier", FakeForest)

    result = common_classifiers.train_static_classifier(
        "rf",
        np.arange(16, dtype=np.float32).reshape(8, 2),
        np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        validation_x=np.arange(8, dtype=np.float32).reshape(4, 2),
        validation_y=np.asarray([0, 1, 0, 1], dtype=np.int64),
        test_x=np.arange(8, dtype=np.float32).reshape(4, 2),
        random_state=7,
        n_estimators=1000,
        model_params={"max_depth": [8, 16, None]},
    )

    assert result.status == "success"
    assert result.best_params["max_depth"] == 16
    assert result.best_validation_pr_auc == pytest.approx(1.0)
    assert [params["max_depth"] for params in constructed] == [8, 16, None]
    assert {params["class_weight"] for params in constructed} == {"balanced_subsample"}
    assert {params["n_estimators"] for params in constructed} == {1000}


def test_xgboost_selects_depth_and_learning_rate_grid_by_validation_pr_auc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mind.wavelet_course.common_classifiers as common_classifiers

    constructed: list[dict[str, object]] = []

    class FakeXGBClassifier:
        def __init__(self, **params: object) -> None:
            self.params = dict(params)
            constructed.append(self.params)

        def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object) -> "FakeXGBClassifier":
            self.fit_kwargs = dict(kwargs)
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            depth = self.params.get("max_depth")
            lr = self.params.get("learning_rate")
            if depth == 3 and lr == 0.1:
                scores = np.asarray([0.05, 0.95, 0.10, 0.90], dtype=np.float32)
            else:
                scores = np.asarray([0.90, 0.20, 0.80, 0.10], dtype=np.float32)
            scores = np.resize(scores, x.shape[0])
            return np.column_stack([1.0 - scores, scores])

    monkeypatch.setattr(common_classifiers, "_load_xgboost_classifier", lambda: FakeXGBClassifier)

    result = common_classifiers.train_static_classifier(
        "xgboost",
        np.arange(16, dtype=np.float32).reshape(8, 2),
        np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        validation_x=np.arange(8, dtype=np.float32).reshape(4, 2),
        validation_y=np.asarray([0, 1, 0, 1], dtype=np.int64),
        test_x=np.arange(8, dtype=np.float32).reshape(4, 2),
        random_state=7,
        n_estimators=5000,
        model_params={
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.1],
            "early_stopping_rounds": 100,
        },
    )

    assert result.status == "success"
    assert result.best_params["max_depth"] == 3
    assert result.best_params["learning_rate"] == 0.1
    assert result.best_params["n_estimators"] == 5000
    assert result.best_params["early_stopping_rounds"] == 100
    assert len(constructed) == 6
    assert {params["max_depth"] for params in constructed} == {2, 3, 4}
    assert {params["learning_rate"] for params in constructed} == {0.03, 0.1}
    assert {params["n_estimators"] for params in constructed} == {5000}
