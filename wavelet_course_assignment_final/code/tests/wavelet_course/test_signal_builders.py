from __future__ import annotations

import numpy as np


def test_teacher_signal_uses_hidden_dim_rows_and_layer_columns() -> None:
    from mind.wavelet_course.signal_builders import teacher_hidden_dim_signal

    layer_vectors = np.arange(36 * 4096, dtype=np.float32).reshape(36, 4096)

    signal = teacher_hidden_dim_signal(
        layer_vectors,
        expected_num_layers=36,
        expected_hidden_dim=4096,
    )

    assert signal.shape == (4096, 36)
    assert signal.dtype == np.float32
    np.testing.assert_array_equal(signal[0], layer_vectors[:, 0])
    np.testing.assert_array_equal(signal[-1], layer_vectors[:, -1])


def test_ours_signal_uses_semantic_trace_rows_and_layer_columns() -> None:
    from mind.wavelet_course.signal_builders import (
        REQUIRED_OURS_TRACE_NAMES,
        ours_semantic_trace_signal,
    )

    layer_vectors = np.arange(36 * 4, dtype=np.float32).reshape(36, 4)
    final_logits = np.asarray([0.2, 0.7, -0.1], dtype=np.float32)

    signal, trace_names = ours_semantic_trace_signal(
        layer_vectors,
        final_logits=final_logits,
        yes_token_id=1,
        no_token_id=2,
        expected_num_layers=36,
        expected_hidden_dim=4,
        return_names=True,
    )

    assert signal.shape == (len(REQUIRED_OURS_TRACE_NAMES), 36)
    assert signal.dtype == np.float32
    assert trace_names == REQUIRED_OURS_TRACE_NAMES
    np.testing.assert_allclose(
        signal[0],
        np.log(np.linalg.norm(layer_vectors, axis=1) + 1e-12),
        rtol=1e-6,
    )
    expected_delta = np.zeros(36, dtype=np.float32)
    expected_delta[1:] = np.linalg.norm(np.diff(layer_vectors, axis=0), axis=1)
    np.testing.assert_allclose(signal[1], expected_delta, rtol=1e-6)
    np.testing.assert_allclose(signal[7], np.full(36, 0.8, dtype=np.float32))
    assert np.all(signal[8] > 0.0)
    np.testing.assert_allclose(signal[9], np.var(layer_vectors, axis=1), rtol=1e-6)


def test_ours_signal_builds_required_v2_trace_definitions() -> None:
    from mind.wavelet_course.signal_builders import (
        REQUIRED_OURS_TRACE_NAMES,
        ours_semantic_trace_signal,
    )

    layer_vectors = np.zeros((36, 3), dtype=np.float32)
    for layer in range(36):
        layer_vectors[layer] = np.asarray([float(layer), float(layer * layer), 1.0], dtype=np.float32)

    signal, trace_names = ours_semantic_trace_signal(
        layer_vectors,
        final_yes_logit=3.0,
        final_no_logit=1.0,
        expected_num_layers=36,
        expected_hidden_dim=3,
        return_names=True,
    )

    name_to_index = {name: index for index, name in enumerate(trace_names)}
    assert trace_names == REQUIRED_OURS_TRACE_NAMES
    assert set(trace_names) == {
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
    }
    assert signal.shape == (10, 36)
    assert np.isfinite(signal).all()

    deltas = np.zeros_like(layer_vectors)
    deltas[1:] = layer_vectors[1:] - layer_vectors[:-1]
    expected_second_delta = np.zeros(36, dtype=np.float32)
    expected_second_delta[2:] = np.linalg.norm(deltas[2:] - deltas[1:-1], axis=1)
    np.testing.assert_allclose(
        signal[name_to_index["second_delta_norm_trace"]],
        expected_second_delta,
        rtol=1e-6,
    )
    assert signal[name_to_index["curvature_trace"]][0] == 0.0
    assert signal[name_to_index["curvature_trace"]][1] == 0.0
    np.testing.assert_allclose(
        signal[name_to_index["yes_no_margin_trace"]],
        np.full(36, 2.0, dtype=np.float32),
    )
    np.testing.assert_allclose(
        signal[name_to_index["hidden_variance_trace"]],
        np.var(layer_vectors, axis=1),
        rtol=1e-6,
    )
