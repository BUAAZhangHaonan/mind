from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mind.geometry.gpu_anisotropic import (
    DEFAULT_EPS,
    DEFAULT_RANK_CAP,
    AnisotropicScoreResult,
    compute_anisotropic_feature_row_gpu,
    compute_anisotropic_scores_gpu,
    compute_anisotropic_scores_for_radius_ball_gpu,
    compute_multi_variant_scores_for_radius_ball_gpu,
    diag_maha_gpu,
    full_maha_shrink_gpu,
    lowrank_maha_gpu,
    radius_ball_isotropic_scores_gpu,
)


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for anisotropic geometry tests")
    return torch.device("cuda")


def _normalize_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.detach().cpu().to(dtype=torch.float32), dim=-1)


def _cpu_diag_reference(query: torch.Tensor, neighbors: torch.Tensor, *, eps: float = DEFAULT_EPS) -> torch.Tensor:
    query_cpu = _normalize_cpu(query)
    neighbors_cpu = _normalize_cpu(neighbors)
    mean = neighbors_cpu.mean(dim=1)
    centered = neighbors_cpu - mean.unsqueeze(1)
    variance = centered.square().mean(dim=1)
    return (((query_cpu - mean).square() / (variance + eps)).sum(dim=1)).clamp_min(0.0).sqrt()


def _cpu_angular(query: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    query_cpu = _normalize_cpu(query)
    reference_cpu = _normalize_cpu(reference)
    return torch.acos((query_cpu @ reference_cpu.T).clamp(-1.0 + eps, 1.0 - eps))


def _cpu_local_eigh(centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = centered.shape[0]
    gram = centered @ centered.T
    eigenvalues_raw, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues_raw, descending=True)
    eigenvalues_raw = eigenvalues_raw[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues_raw > torch.finfo(centered.dtype).eps * max(centered.shape) * eigenvalues_raw.abs().max().clamp_min(1.0)
    eigenvalues_raw = eigenvalues_raw[positive]
    eigenvectors = eigenvectors[:, positive]
    covariance_eigenvalues = eigenvalues_raw / float(k)
    return eigenvalues_raw, covariance_eigenvalues, eigenvectors


def _cpu_lowrank_reference(
    query: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    rank_cap: int = DEFAULT_RANK_CAP,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    query_cpu = _normalize_cpu(query)
    neighbors_cpu = _normalize_cpu(neighbors)
    rows = []
    for query_row, neighbor_rows in zip(query_cpu, neighbors_cpu, strict=True):
        k, d = neighbor_rows.shape
        mean = neighbor_rows.mean(dim=0)
        centered = neighbor_rows - mean
        delta = query_row - mean
        raw_eigenvalues, covariance_eigenvalues, eigenvectors = _cpu_local_eigh(centered)
        rank = min(int(rank_cap), k - 1, d, int(covariance_eigenvalues.numel()))
        if rank > 0:
            raw = raw_eigenvalues[:rank]
            eigenvalues = covariance_eigenvalues[:rank]
            basis = (centered.T @ eigenvectors[:, :rank]) / raw.clamp_min(eps).sqrt().unsqueeze(0)
            projected = basis.T @ delta
            in_subspace = (projected.square() / (eigenvalues + eps)).sum()
            projected_norm2 = projected.square().sum()
            used_trace = eigenvalues.sum()
        else:
            in_subspace = torch.zeros((), dtype=torch.float32)
            projected_norm2 = torch.zeros((), dtype=torch.float32)
            used_trace = torch.zeros((), dtype=torch.float32)
        residual_norm2 = (delta.square().sum() - projected_norm2).clamp_min(0.0)
        total_trace = centered.square().sum() / float(k)
        residual_variance = ((total_trace - used_trace).clamp_min(0.0) / float(max(d - rank, 1))).clamp_min(eps)
        rows.append((in_subspace + residual_norm2 / residual_variance).clamp_min(0.0).sqrt())
    return torch.stack(rows)


def _cpu_ledoit_wolf_alpha(centered: torch.Tensor, *, eps: float = DEFAULT_EPS) -> torch.Tensor:
    k, d = centered.shape
    gram = centered @ centered.T
    trace_cov = torch.trace(gram) / float(k)
    tau = trace_cov / float(d)
    fro_cov2 = gram.square().sum() / float(k * k)
    target_dist2 = (fro_cov2 - float(d) * tau.square()).clamp_min(0.0)
    if target_dist2 <= eps:
        return torch.tensor(1.0, dtype=torch.float32)
    row_norm2 = centered.square().sum(dim=1)
    b2 = row_norm2.square().sum() / float(k * k) - fro_cov2 / float(k)
    return (b2 / target_dist2).clamp(0.0, 1.0)


def _cpu_full_shrink_reference(
    query: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_cpu = _normalize_cpu(query)
    neighbors_cpu = _normalize_cpu(neighbors)
    values = []
    alphas = []
    for query_row, neighbor_rows in zip(query_cpu, neighbors_cpu, strict=True):
        k, d = neighbor_rows.shape
        mean = neighbor_rows.mean(dim=0)
        centered = neighbor_rows - mean
        delta = query_row - mean
        alpha = _cpu_ledoit_wolf_alpha(centered, eps=eps)
        covariance = centered.T @ centered / float(k)
        tau = torch.trace(covariance) / float(d)
        shrunk = (1.0 - alpha) * covariance + alpha * tau * torch.eye(d, dtype=torch.float32)
        solved = torch.linalg.solve(shrunk + eps * torch.eye(d, dtype=torch.float32), delta.unsqueeze(1)).squeeze(1)
        values.append((delta * solved).sum().clamp_min(0.0).sqrt())
        alphas.append(alpha)
    return torch.stack(values), torch.stack(alphas)


def test_diag_maha_normalizes_inputs_and_matches_independent_cpu_reference() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[4.0, 3.0, 1.0], [0.5, -2.0, 3.0]], device=device)
    neighbors = torch.tensor(
        [
            [[2.0, 0.5, 1.0], [1.0, 3.0, 2.0], [5.0, 4.0, 1.0]],
            [[1.0, -1.0, 1.0], [2.0, -3.0, 0.5], [0.25, -1.0, 2.0]],
        ],
        device=device,
    )

    result = diag_maha_gpu(query, neighbors)

    assert isinstance(result, AnisotropicScoreResult)
    assert result.values.device.type == "cuda"
    torch.testing.assert_close(result.values.cpu(), _cpu_diag_reference(query, neighbors), atol=1e-5, rtol=1e-5)


def test_anisotropic_variants_reject_single_neighbor_neighborhoods() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    neighbors = torch.tensor([[[1.0, 0.1, 0.0]]], device=device)

    for scorer in (diag_maha_gpu, lowrank_maha_gpu, full_maha_shrink_gpu):
        with pytest.raises(ValueError, match="at least two"):
            scorer(query, neighbors)

    with pytest.raises(RuntimeError, match="fewer than two"):
        compute_anisotropic_scores_for_radius_ball_gpu(
            query,
            neighbors.squeeze(0),
            radius=0.2,
            variant="diag_maha",
            reference_chunk_size=8,
        )


def test_lowrank_maha_uses_gram_denominator_k_and_ignores_zero_eigenvalues() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[0.2, 1.0, 0.0, 0.0]], device=device)
    neighbors = torch.tensor(
        [
            [
                [-2.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        ],
        device=device,
    )

    result = lowrank_maha_gpu(query, neighbors, rank_cap=8)

    expected = _cpu_lowrank_reference(query, neighbors, rank_cap=8)
    torch.testing.assert_close(result.values.cpu(), expected, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(result.values).all()


def test_full_maha_shrink_uses_analytic_alpha_and_matches_dense_small_d_solve() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[2.0, 1.0, 0.5], [0.25, 2.0, 1.0]], device=device)
    neighbors = torch.tensor(
        [
            [[1.0, 0.0, 0.5], [2.0, 1.0, 1.0], [3.0, 1.0, 0.25], [1.5, 0.5, 2.0]],
            [[0.0, 0.25, 1.0], [1.0, 0.5, 1.5], [1.5, 1.5, 0.5], [2.0, 1.0, 1.0]],
        ],
        device=device,
    )

    result = full_maha_shrink_gpu(query, neighbors)
    expected_values, expected_alpha = _cpu_full_shrink_reference(query, neighbors)

    assert result.alpha is not None
    assert torch.all((result.alpha >= 0.0) & (result.alpha <= 1.0))
    torch.testing.assert_close(result.alpha.cpu(), expected_alpha, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.values.cpu(), expected_values, atol=2e-4, rtol=2e-4)


def test_radius_ball_isotropic_matches_mean_angular_distance() -> None:
    device = _cuda_or_skip()
    query = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], device=device)
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 1.0, 0.0],
        ],
        device=device,
    )
    radius = 0.2

    result = radius_ball_isotropic_scores_gpu(query, reference, radius=radius, reference_chunk_size=2)

    expected = []
    distances = _cpu_angular(query, reference)
    for row in distances:
        selected = row[row <= radius]
        expected.append(selected.mean())
    torch.testing.assert_close(result.values.cpu(), torch.stack(expected), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.neighbor_counts.cpu(), torch.tensor([2.0, 2.0]), atol=0.0, rtol=0.0)


def test_multi_variant_radius_ball_matches_manual_selection_for_all_variants() -> None:
    device = _cuda_or_skip()
    query = torch.tensor(
        [
            [1.0, 0.02, 0.0, 0.0],
            [0.02, 1.0, 0.0, 0.0],
        ],
        device=device,
    )
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.10, 0.0, 0.0],
            [0.96, -0.10, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.10, 0.99, 0.0, 0.0],
            [-0.10, 0.96, 0.0, 0.0],
        ],
        device=device,
    )
    variants = ("radius_ball_isotropic", "diag_maha", "lowrank_maha", "full_maha_shrink")
    radius = torch.tensor([0.16, 0.16], device=device)

    results = compute_multi_variant_scores_for_radius_ball_gpu(
        query,
        reference,
        radius=radius,
        variants=variants,
        reference_chunk_size=2,
        radius_margin=0.0,
    )

    distances = torch.acos((_normalize_cpu(query) @ _normalize_cpu(reference).T).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    expected_values = {variant: [] for variant in variants}
    expected_counts = []
    expected_alpha = []
    for row_index, row_distances in enumerate(distances):
        mask = row_distances <= radius[row_index].cpu()
        selected = reference[mask.to(device)]
        expected_counts.append(float(selected.shape[0]))
        expected_values["radius_ball_isotropic"].append(row_distances[mask].mean().to(device))
        for variant in ("diag_maha", "lowrank_maha", "full_maha_shrink"):
            manual = compute_anisotropic_scores_gpu(
                query[row_index : row_index + 1],
                selected.unsqueeze(0),
                variant=variant,
            )
            expected_values[variant].append(manual.values.squeeze(0))
            if variant == "full_maha_shrink":
                assert manual.alpha is not None
                expected_alpha.append(manual.alpha.squeeze(0))

    expected_count_tensor = torch.tensor(expected_counts, device=device)
    assert set(results) == set(variants)
    for variant in variants:
        assert results[variant].neighbor_counts is not None
        torch.testing.assert_close(results[variant].neighbor_counts, expected_count_tensor, atol=0.0, rtol=0.0)
        torch.testing.assert_close(results[variant].values, torch.stack(expected_values[variant]), atol=1e-5, rtol=1e-5)
    assert results["full_maha_shrink"].alpha is not None
    torch.testing.assert_close(results["full_maha_shrink"].alpha, torch.stack(expected_alpha), atol=1e-5, rtol=1e-5)


def test_feature_row_enforces_raw_plus_full_curve_contract_when_requested() -> None:
    device = _cuda_or_skip()
    selected_layers = list(range(16))
    layer_vectors = torch.stack(
        [torch.tensor([1.0, 0.01 * index, 0.0], device=device) for index in range(16)]
    )
    reference_layers = {
        layer: torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.02, 0.0],
                [1.0, -0.02, 0.0],
            ],
            device=device,
        )
        for layer in selected_layers
    }
    layer_radii = {layer: 0.3 for layer in selected_layers}

    row = compute_anisotropic_feature_row_gpu(
        layer_vectors=layer_vectors,
        selected_layers=selected_layers,
        reference_layers=reference_layers,
        layer_radii=layer_radii,
        variant="radius_ball_isotropic",
        expected_curve_length=16,
    )

    assert list(row) == [*(f"raw_drift_{index}" for index in range(16)), "cal_mean_drift", "cal_max_drift", "cal_final_drift", "cal_drift_slope", "cal_drift_variance"]

    with pytest.raises(ValueError, match="expected 16"):
        compute_anisotropic_feature_row_gpu(
            layer_vectors=layer_vectors[:15],
            selected_layers=selected_layers[:15],
            reference_layers=reference_layers,
            layer_radii=layer_radii,
            variant="radius_ball_isotropic",
            expected_curve_length=16,
        )


def test_cuda_smoke_full_shrink_high_dimensional_matrix_free() -> None:
    device = _cuda_or_skip()
    generator = torch.Generator(device=device).manual_seed(13)
    query = torch.randn((1, 4096), device=device, generator=generator)
    neighbors = torch.randn((1, 30, 4096), device=device, generator=generator)

    result = full_maha_shrink_gpu(query, neighbors)

    assert result.values.shape == (1,)
    assert result.alpha is not None
    assert result.alpha.shape == (1,)
    assert torch.isfinite(result.values).all()
    assert torch.all((result.alpha >= 0.0) & (result.alpha <= 1.0))


def test_production_scoring_rejects_cpu_tensors() -> None:
    query = torch.ones((1, 2))
    neighbors = torch.ones((1, 3, 2))

    with pytest.raises(ValueError, match="CUDA"):
        diag_maha_gpu(query, neighbors)
