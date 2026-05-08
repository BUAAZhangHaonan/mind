from __future__ import annotations

import numpy as np
import torch

from mind.trajectory.stage_a_readouts import train_lstm_diagnostic
from mind.trajectory.stage_a_representations import (
    StageATrajectoryLSTM,
    shuffled_layer_permutation,
)


def test_synthetic_small_tensor_training_runs_for_one_epoch() -> None:
    rng = np.random.default_rng(20260506)
    x = rng.normal(size=(12, 4, 6)).astype(np.float32)
    y = np.array([0, 1] * 6, dtype=np.int64)

    result = train_lstm_diagnostic(
        x,
        y,
        num_layers=4,
        hidden_dim=6,
        epochs=1,
        batch_size=4,
        device="cpu",
        seed=20260506,
        patience=1,
    )

    embeddings, logits = result.model.embed_and_score(torch.from_numpy(x))
    assert embeddings.shape == (12, 128)
    assert logits.shape == (12,)
    assert len(result.history) == 1


def test_classifier_score_shape_is_correct() -> None:
    model = StageATrajectoryLSTM(hidden_dim=5)
    x = torch.randn(3, 4, 5)

    embeddings, logits = model.embed_and_score(x)

    assert embeddings.shape == (3, 128)
    assert logits.shape == (3,)


def test_deterministic_seed_behavior() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(10, 3, 5)).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    first = train_lstm_diagnostic(
        x,
        y,
        num_layers=3,
        hidden_dim=5,
        epochs=1,
        batch_size=5,
        device="cpu",
        seed=20260506,
        patience=1,
    )
    second = train_lstm_diagnostic(
        x,
        y,
        num_layers=3,
        hidden_dim=5,
        epochs=1,
        batch_size=5,
        device="cpu",
        seed=20260506,
        patience=1,
    )

    _, first_logits = first.model.embed_and_score(torch.from_numpy(x))
    _, second_logits = second.model.embed_and_score(torch.from_numpy(x))
    torch.testing.assert_close(first_logits, second_logits)


def test_shuffled_and_ordered_permutations_are_deterministic() -> None:
    ordered = list(range(4))
    shuffled = shuffled_layer_permutation(4, seed=20260506)

    assert shuffled == shuffled_layer_permutation(4, seed=20260506)
    assert ordered == list(range(4))
    assert shuffled != ordered
