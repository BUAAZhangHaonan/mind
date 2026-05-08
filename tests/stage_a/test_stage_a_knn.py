from __future__ import annotations

import numpy as np
import pytest
import torch

from mind.trajectory.stage_a_readouts import compute_knn_scores


def test_knn_score_shape() -> None:
    bank = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    scores = compute_knn_scores(bank, query, k=2, metric="euclidean")

    assert scores.shape == (2,)


def test_higher_distance_means_more_anomalous() -> None:
    bank = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    query = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)

    scores = compute_knn_scores(bank, query, k=2, metric="euclidean")

    assert scores[1] > scores[0]


def test_fail_if_bank_has_fewer_than_k_correct_samples() -> None:
    bank = np.array([[0.0, 0.0]], dtype=np.float32)
    query = np.array([[0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="bank has fewer than k"):
        compute_knn_scores(bank, query, k=2, metric="euclidean")


def test_angular_distance_is_used_for_sphere_embeddings() -> None:
    bank = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)

    scores = compute_knn_scores(bank, query, k=1, metric="angular")

    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.5)


def test_euclidean_distance_is_used_for_raw_and_norm_embeddings() -> None:
    bank = np.array([[0.0], [2.0]], dtype=np.float32)
    query = np.array([[1.0]], dtype=np.float32)

    scores = compute_knn_scores(bank, query, k=2, metric="euclidean")

    assert scores[0] == pytest.approx(1.0)


def test_torch_backend_matches_numpy_euclidean_scores() -> None:
    rng = np.random.default_rng(17)
    bank = rng.normal(size=(11, 6)).astype(np.float32)
    query = rng.normal(size=(7, 6)).astype(np.float32)

    numpy_scores = compute_knn_scores(bank, query, k=4, metric="euclidean", backend="numpy")
    torch_scores = compute_knn_scores(
        bank,
        query,
        k=4,
        metric="euclidean",
        backend="torch",
        device="cpu",
        chunk_size=2,
    )

    np.testing.assert_allclose(torch_scores, numpy_scores, rtol=1e-5, atol=1e-6)


def test_torch_backend_matches_numpy_angular_scores() -> None:
    rng = np.random.default_rng(23)
    bank = rng.normal(size=(13, 5)).astype(np.float32)
    query = rng.normal(size=(8, 5)).astype(np.float32)

    numpy_scores = compute_knn_scores(bank, query, k=3, metric="angular", backend="numpy")
    torch_scores = compute_knn_scores(
        bank,
        query,
        k=3,
        metric="angular",
        backend="torch",
        device="cpu",
        chunk_size=3,
    )

    np.testing.assert_allclose(torch_scores, numpy_scores, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_cuda_backend_matches_numpy_scores() -> None:
    rng = np.random.default_rng(31)
    bank = rng.normal(size=(17, 7)).astype(np.float32)
    query = rng.normal(size=(9, 7)).astype(np.float32)

    for metric in ("euclidean", "angular"):
        numpy_scores = compute_knn_scores(bank, query, k=5, metric=metric, backend="numpy")
        cuda_scores = compute_knn_scores(
            bank,
            query,
            k=5,
            metric=metric,
            backend="torch",
            device="cuda",
            chunk_size=4,
        )

        np.testing.assert_allclose(cuda_scores, numpy_scores, rtol=2e-5, atol=2e-6)
