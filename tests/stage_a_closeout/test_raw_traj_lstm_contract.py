from __future__ import annotations

import numpy as np
import torch

from mind.trajectory.stage_a_representations import (
    StageATrajectoryLSTM,
    build_lstm_trajectory,
    build_raw_lstm_trajectory,
)


def _entry() -> dict[str, object]:
    return {
        "layer_vectors": torch.tensor(
            [[3.0, 4.0], [0.0, 2.0], [5.0, 12.0]],
            dtype=torch.float32,
        )
    }


def test_raw_traj_lstm_uses_raw_layer_inputs() -> None:
    raw = build_raw_lstm_trajectory(_entry())

    np.testing.assert_allclose(raw, np.array([[3.0, 4.0], [0.0, 2.0], [5.0, 12.0]], dtype=np.float32))


def test_sphere_traj_lstm_uses_layerwise_normalized_inputs() -> None:
    sphere = build_lstm_trajectory(_entry())

    np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), np.ones(3), atol=1e-6)
    assert not np.allclose(sphere, build_raw_lstm_trajectory(_entry()))


def test_raw_and_sphere_lstm_architecture_parity() -> None:
    raw_model = StageATrajectoryLSTM(hidden_dim=2)
    sphere_model = StageATrajectoryLSTM(hidden_dim=2)

    assert type(raw_model.input_projection) is type(sphere_model.input_projection)
    assert raw_model.input_projection.in_features == sphere_model.input_projection.in_features
    assert raw_model.lstm.hidden_size == sphere_model.lstm.hidden_size
    assert raw_model.embedding_projection.out_features == sphere_model.embedding_projection.out_features
