from __future__ import annotations

import numpy as np
import torch

from mind.trajectory.stage_a_representations import (
    build_representation,
    shuffled_layer_permutation,
)


def _entry() -> dict[str, object]:
    return {
        "layer_vectors": torch.tensor(
            [
                [3.0, 4.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=torch.float16,
        )
    }


def test_raw_static_shape() -> None:
    rep = build_representation(_entry(), "Raw-Static")
    assert rep.shape == (3,)
    assert rep.dtype == np.float32
    np.testing.assert_allclose(rep, np.array([0.0, 2.0, 0.0], dtype=np.float32))


def test_sphere_static_unit_norm() -> None:
    rep = build_representation(_entry(), "Sphere-Static")
    assert rep.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(rep), 1.0, atol=1e-6)


def test_norm_static_scalar() -> None:
    rep = build_representation(_entry(), "Norm-Static")
    assert rep.shape == (1,)
    np.testing.assert_allclose(rep[0], np.log(2.0), atol=1e-6)


def test_raw_traj_meanpool_shape() -> None:
    rep = build_representation(_entry(), "Raw-Traj-MeanPool")
    assert rep.shape == (3,)
    np.testing.assert_allclose(rep, np.array([4 / 3, 2.0, 0.0], dtype=np.float32))


def test_sphere_traj_meanpool_unit_norm() -> None:
    rep = build_representation(_entry(), "Sphere-Traj-MeanPool")
    assert rep.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(rep), 1.0, atol=1e-6)


def test_norm_traj_length_equals_num_layers() -> None:
    rep = build_representation(_entry(), "Norm-Traj")
    assert rep.shape == (3,)
    np.testing.assert_allclose(rep, np.log(np.array([5.0, 1.0, 2.0], dtype=np.float32)), atol=1e-6)


def test_deterministic_shuffled_layer_order() -> None:
    first = shuffled_layer_permutation(num_layers=6, seed=20260506)
    second = shuffled_layer_permutation(num_layers=6, seed=20260506)
    different_size = shuffled_layer_permutation(num_layers=5, seed=20260506)

    assert first == second
    assert sorted(first) == list(range(6))
    assert first != list(range(6))
    assert sorted(different_size) == list(range(5))
