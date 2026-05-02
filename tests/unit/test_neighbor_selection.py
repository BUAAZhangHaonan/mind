from __future__ import annotations

import importlib.util
import importlib.abc
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mind.geometry.neighbor_selection import (
    METHOD_NAMES,
    compute_neighbor_feature_row_gpu,
    compute_neighbor_scores_gpu,
    tune_radius_for_target_count_gpu,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "neighbor_selection_comparison.py"


def _load_script():
    assert SCRIPT_PATH.exists(), "scripts/experiments/neighbor_selection_comparison.py should exist"
    spec = importlib.util.spec_from_file_location("neighbor_selection_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for neighbor selection tests")
    return torch.device("cuda")


def _cpu_angular(query: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    query_n = F.normalize(query.cpu(), dim=1)
    reference_n = F.normalize(reference.cpu(), dim=1)
    return torch.acos((query_n @ reference_n.T).clamp(-1.0 + eps, 1.0 - eps))


def _cpu_cosine(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.normalize(query.cpu(), dim=1) @ F.normalize(reference.cpu(), dim=1).T


def _cpu_euclidean(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.cdist(query.cpu(), reference.cpu())


def test_method_names_are_exact_and_stable() -> None:
    assert METHOD_NAMES == (
        "knn_angular_k30",
        "kernel_knn_k30",
        "radius_ball",
        "knn_cosine_k30",
        "knn_euclidean_k30",
    )


def test_gpu_knn_methods_match_cpu_references() -> None:
    device = _cuda_or_skip()
    query = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.1]],
        device=device,
    )
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        device=device,
    )
    k = 2
    angular = _cpu_angular(query, reference)

    angular_result = compute_neighbor_scores_gpu(query, reference, method="knn_angular_k30", k=k, reference_chunk_size=2)
    expected_angular = torch.topk(angular, k=k, dim=1, largest=False).values.mean(dim=1)

    kernel_result = compute_neighbor_scores_gpu(query, reference, method="kernel_knn_k30", k=k, reference_chunk_size=2)
    kernel_distances = torch.topk(angular, k=k, dim=1, largest=False).values
    kernel_weights = 1.0 / kernel_distances.clamp_min(1e-6)
    expected_kernel = (kernel_distances * kernel_weights).sum(dim=1) / kernel_weights.sum(dim=1)

    cosine_result = compute_neighbor_scores_gpu(query, reference, method="knn_cosine_k30", k=k, reference_chunk_size=2)
    cosine_indices = torch.topk(_cpu_cosine(query, reference), k=k, dim=1, largest=True).indices
    expected_cosine = angular.gather(1, cosine_indices).mean(dim=1)

    euclidean_result = compute_neighbor_scores_gpu(query, reference, method="knn_euclidean_k30", k=k, reference_chunk_size=2)
    euclidean_indices = torch.topk(_cpu_euclidean(query, reference), k=k, dim=1, largest=False).indices
    expected_euclidean = angular.gather(1, euclidean_indices).mean(dim=1)

    for result, expected in [
        (angular_result, expected_angular),
        (kernel_result, expected_kernel),
        (cosine_result, expected_cosine),
        (euclidean_result, expected_euclidean),
    ]:
        assert result.values.device.type == "cuda"
        torch.testing.assert_close(result.values.cpu(), expected, atol=1e-6, rtol=1e-6)


def test_radius_tuning_gets_target_average_count_without_labels() -> None:
    device = _cuda_or_skip()
    query = torch.eye(3, device=device)
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ],
        device=device,
    )

    radius = tune_radius_for_target_count_gpu(query, reference, target_count=2, reference_chunk_size=2)
    result = compute_neighbor_scores_gpu(
        query,
        reference,
        method="radius_ball",
        target_count=2,
        radius=radius,
        reference_chunk_size=2,
    )

    assert result.radius is not None
    assert result.neighbor_counts is not None
    assert float(result.neighbor_counts.float().mean().detach().cpu()) == pytest.approx(2.0, abs=0.05)
    expected = _cpu_angular(query, reference).topk(k=2, dim=1, largest=False).values.mean(dim=1)
    torch.testing.assert_close(result.values.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_runner_radius_ball_uses_one_shared_radius_per_layer_over_eval_batch() -> None:
    device = _cuda_or_skip()
    script = _load_script()
    q1 = torch.tensor([1.0, 0.0], device=device)
    q2 = torch.tensor([0.0, 1.0], device=device)
    reference = torch.stack(
        [
            torch.tensor([1.0, 0.0], device=device),
            torch.tensor([torch.cos(torch.tensor(0.1)), torch.sin(torch.tensor(0.1))], device=device),
            torch.tensor([torch.cos(torch.tensor(0.2)), torch.sin(torch.tensor(0.2))], device=device),
            torch.tensor([0.0, 1.0], device=device),
            torch.tensor([torch.cos(torch.tensor(torch.pi / 2 - 1.0)), torch.sin(torch.tensor(torch.pi / 2 - 1.0))], device=device),
        ],
        dim=0,
    )
    cache_entries = [
        {
            "sample_id": "q1",
            "image_id": 1,
            "label": 1,
            "parsed_answer": 1,
            "selected_layers": [3],
            "layer_vectors": [q1.cpu().tolist()],
        },
        {
            "sample_id": "q2",
            "image_id": 2,
            "label": 1,
            "parsed_answer": 1,
            "selected_layers": [3],
            "layer_vectors": [q2.cpu().tolist()],
        },
    ]

    frame = script.build_method_feature_frame(
        cache_entries=cache_entries,
        reference_layers={3: reference},
        method="radius_ball",
        device=device,
        target_count=2,
        reference_chunk_size=2,
    )

    per_query_values = [
        compute_neighbor_feature_row_gpu(
            layer_vectors=torch.stack([query], dim=0),
            selected_layers=[3],
            reference_layers={3: reference},
            method="radius_ball",
            target_count=2,
            reference_chunk_size=2,
        )["raw_drift_0"]
        for query in (q1, q2)
    ]
    shared_radius = tune_radius_for_target_count_gpu(
        torch.stack([q1, q2], dim=0),
        reference,
        target_count=2,
        reference_chunk_size=2,
    )
    shared_result = compute_neighbor_scores_gpu(
        torch.stack([q1, q2], dim=0),
        reference,
        method="radius_ball",
        target_count=2,
        radius=shared_radius,
        reference_chunk_size=2,
    )

    assert frame["raw_drift_0"].tolist() != pytest.approx(per_query_values)
    assert frame["raw_drift_0"].tolist() == pytest.approx(shared_result.values.cpu().tolist(), abs=1e-5)


def test_feature_row_uses_raw_curve_and_calibrated_summary_recipe() -> None:
    device = _cuda_or_skip()
    layer_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
    )
    reference_layers = {
        3: torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        7: torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device=device),
    }

    row = compute_neighbor_feature_row_gpu(
        layer_vectors=layer_vectors,
        selected_layers=[3, 7],
        reference_layers=reference_layers,
        method="knn_angular_k30",
        k=2,
    )
    curve = torch.tensor([row["raw_drift_0"], row["raw_drift_1"]])

    assert set(row) == {
        "raw_drift_0",
        "raw_drift_1",
        "cal_mean_drift",
        "cal_max_drift",
        "cal_final_drift",
        "cal_drift_slope",
        "cal_drift_variance",
    }
    assert row["cal_mean_drift"] == pytest.approx(float(curve.mean()))
    assert row["cal_max_drift"] == pytest.approx(float(curve.max()))
    assert row["cal_final_drift"] == pytest.approx(float(curve[-1]))
    assert row["cal_drift_slope"] == pytest.approx(float(curve[1] - curve[0]))
    assert row["cal_drift_variance"] == pytest.approx(float(curve.var(unbiased=False)))


def test_production_neighbor_selection_rejects_cpu_tensors() -> None:
    query = torch.ones((2, 3))
    reference = torch.ones((4, 3))

    with pytest.raises(ValueError, match="CUDA"):
        compute_neighbor_scores_gpu(query, reference, method="knn_angular_k30", k=2)


def test_format_command_writes_all_requested_outputs_with_all_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    assert script.parse_methods("all") == METHOD_NAMES
    assert all(method in script.build_parser().format_help() for method in METHOD_NAMES)
    rows = [
        {
            "model": "toy-model",
            "benchmark": "toy-benchmark",
            "method": method,
            "roc_auc": 0.70 + index * 0.01,
            "roc_auc_ci_lower": 0.60 + index * 0.01,
            "roc_auc_ci_upper": 0.80 + index * 0.01,
            "pr_auc": 0.50 + index * 0.02,
            "pr_auc_ci_lower": 0.40 + index * 0.02,
            "pr_auc_ci_upper": 0.60 + index * 0.02,
            "status": "ok",
            "n_rows": 8,
            "split_strategy": "image_grouped",
        }
        for index, method in enumerate(reversed(METHOD_NAMES))
    ]
    metrics_path = tmp_path / "metrics.csv"
    csv_path = tmp_path / "comparison.csv"
    markdown_path = tmp_path / "comparison.md"
    analysis_path = tmp_path / "analysis.md"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    result = script.main(
        [
            "format",
            "--metrics-csv",
            str(metrics_path),
            "--csv-path",
            str(csv_path),
            "--markdown-path",
            str(markdown_path),
            "--analysis-path",
            str(analysis_path),
        ]
    )

    assert result == 0
    csv_rows = pd.read_csv(csv_path)
    assert csv_rows["method"].tolist() == list(METHOD_NAMES)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| model | benchmark | knn_angular_k30 | kernel_knn_k30 | radius_ball | knn_cosine_k30 | knn_euclidean_k30 |" in markdown
    assert "ROC-AUC 0.7400 [0.6400, 0.8400]; PR-AUC 0.5800 [0.4800, 0.6800]" in markdown
    analysis = analysis_path.read_text(encoding="utf-8")
    assert "Best method by PR-AUC" in analysis
    assert "knn_angular_k30 > kernel_knn_k30 > radius_ball > knn_cosine_k30 > knn_euclidean_k30" in analysis


def test_format_command_imports_and_runs_without_model_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_roots = ("mind.evaluation", "mind.models", "transformers")

    class BlockedImportFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == root or fullname.startswith(f"{root}.") for root in blocked_roots):
                raise AssertionError(f"blocked import: {fullname}")
            return None

    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(f"{root}.") for root in blocked_roots):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockedImportFinder(), *sys.meta_path])

    metrics_path = tmp_path / "metrics.csv"
    csv_path = tmp_path / "comparison.csv"
    markdown_path = tmp_path / "comparison.md"
    analysis_path = tmp_path / "analysis.md"
    pd.DataFrame(
        [
            {
                "model": "toy-model",
                "benchmark": "toy-benchmark",
                "method": "radius_ball",
                "roc_auc": 0.7,
                "roc_auc_ci_lower": 0.6,
                "roc_auc_ci_upper": 0.8,
                "pr_auc": 0.5,
                "pr_auc_ci_lower": 0.4,
                "pr_auc_ci_upper": 0.6,
                "status": "ok",
            }
        ]
    ).to_csv(metrics_path, index=False)

    script = _load_script()
    result = script.main(
        [
            "format",
            "--metrics-csv",
            str(metrics_path),
            "--csv-path",
            str(csv_path),
            "--markdown-path",
            str(markdown_path),
            "--analysis-path",
            str(analysis_path),
        ]
    )

    assert result == 0
    assert pd.read_csv(csv_path)["method"].tolist() == ["radius_ball"]


def test_format_command_rejects_malformed_ok_metric_rows_with_clear_message(tmp_path: Path) -> None:
    script = _load_script()
    metrics_path = tmp_path / "bad-metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "toy-model",
                "benchmark": "toy-benchmark",
                "method": "radius_ball",
                "status": "ok",
                "roc_auc": "not-a-float",
                "roc_auc_ci_lower": 0.1,
                "roc_auc_ci_upper": 0.9,
                "pr_auc": 0.5,
                "pr_auc_ci_lower": 0.4,
                "pr_auc_ci_upper": 0.6,
            }
        ]
    ).to_csv(metrics_path, index=False)

    with pytest.raises(SystemExit, match="Invalid metric row.*roc_auc.*not-a-float"):
        script.main(
            [
                "format",
                "--metrics-csv",
                str(metrics_path),
                "--csv-path",
                str(tmp_path / "comparison.csv"),
                "--markdown-path",
                str(tmp_path / "comparison.md"),
                "--analysis-path",
                str(tmp_path / "analysis.md"),
            ]
        )
