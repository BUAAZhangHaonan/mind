from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch


RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "anisotropic_scoring_comparison.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("anisotropic_scoring_comparison", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_accepts_all_and_rejects_unknown() -> None:
    runner = _load_runner()

    assert runner.parse_variants("all") == runner.VARIANT_NAMES
    assert runner.parse_variants("diag_maha,full_maha_shrink") == ("diag_maha", "full_maha_shrink")

    with pytest.raises(ValueError, match="Unsupported variants"):
        runner.parse_variants("diag_maha,not_real")


def test_artifact_paths_are_sanitized_under_output_root(tmp_path: Path) -> None:
    runner = _load_runner()

    paths = runner.build_artifact_paths(
        output_root=tmp_path,
        model_name="Qwen3/VL 8B",
        benchmark="POPE popular",
        split_strategy="image_grouped",
        variant="diag_maha",
    )

    assert paths.feature_path == tmp_path / "features" / "qwen3-vl-8b" / "pope-popular" / "image_grouped" / "diag_maha.parquet"
    assert paths.metrics_path == tmp_path / "metrics" / "qwen3-vl-8b" / "pope-popular" / "image_grouped" / "diag_maha.json"
    assert paths.predictions_path == tmp_path / "predictions" / "qwen3-vl-8b" / "pope-popular" / "image_grouped" / "diag_maha.parquet"


def test_load_baseline_rows_filters_and_normalizes(tmp_path: Path) -> None:
    runner = _load_runner()
    baseline_csv = tmp_path / "baselines.csv"
    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "no_manifold",
                "status": "ok",
                "roc_auc": 0.6,
                "roc_auc_ci_lower": 0.5,
                "roc_auc_ci_upper": 0.7,
                "pr_auc": 0.2,
                "pr_auc_ci_lower": 0.1,
                "pr_auc_ci_upper": 0.3,
            },
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "shared",
                "status": "ok",
                "roc_auc": 0.1,
                "roc_auc_ci_lower": 0.1,
                "roc_auc_ci_upper": 0.1,
                "pr_auc": 0.1,
                "pr_auc_ci_lower": 0.1,
                "pr_auc_ci_upper": 0.1,
            },
        ]
    ).to_csv(baseline_csv, index=False)

    rows = runner.load_baseline_rows(baseline_csv, methods=("no_manifold", "linear_probe"))

    assert len(rows) == 1
    assert rows[0]["model"] == "qwen3-vl-8b"
    assert rows[0]["benchmark_key"] == "popular"
    assert rows[0]["method"] == "no_manifold"


def test_resolve_random_state_uses_auto_heldout_defaults_and_explicit_wins() -> None:
    runner = _load_runner()

    assert runner.resolve_random_state("qwen3-vl-8b", "object_heldout", "auto") == 5
    assert runner.resolve_random_state("internvl3.5-8b", "object_heldout", "auto") == 1
    assert runner.resolve_random_state("llava-onevision-7b", "object_heldout", "auto") == 0
    assert runner.resolve_random_state("molmo-7b-d-0924", "object_heldout", "auto") == 1
    assert runner.resolve_random_state("qwen3-vl-8b", "image_grouped", "auto") == 13
    assert runner.resolve_random_state("qwen3-vl-8b", "object_heldout", "17") == 17

    with pytest.raises(ValueError, match="Unsupported --random-state"):
        runner.resolve_random_state("qwen3-vl-8b", "object_heldout", "not-an-int")


def test_min_neighbor_radius_lower_bound_uses_max_kth_nearest_distance() -> None:
    runner = _load_runner()
    distances = torch.tensor(
        [
            [0.40, 0.10, 0.30, 0.20],
            [0.05, 0.80, 0.70, 0.90],
            [0.11, 0.12, 0.13, 0.14],
        ],
        dtype=torch.float32,
    )

    lower_bound = runner.min_neighbor_radius_lower_bound_from_distances(distances, min_neighbors=2)

    assert lower_bound.item() == pytest.approx(0.70)


def test_min_neighbor_radius_lower_bound_rejects_impossible_request() -> None:
    runner = _load_runner()
    distances = torch.ones((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="min_neighbors cannot exceed"):
        runner.min_neighbor_radius_lower_bound_from_distances(distances, min_neighbors=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_min_neighbor_radius_tuning_does_not_use_subtask1_private_topk(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()

    class NeighborModuleWithoutPrivateTopk:
        @staticmethod
        def _max_angular_distance_gpu(
            query: torch.Tensor,
            reference: torch.Tensor,
            *,
            query_chunk_size: int,
            reference_chunk_size: int,
        ) -> torch.Tensor:
            distances = runner.batch_angular_distance(
                query,
                reference,
                query_chunk_size=query_chunk_size,
                reference_chunk_size=reference_chunk_size,
            )
            return distances.max()

        @staticmethod
        def _radius_counts_gpu(
            query: torch.Tensor,
            reference: torch.Tensor,
            radius: torch.Tensor,
            *,
            query_chunk_size: int,
            reference_chunk_size: int,
        ) -> torch.Tensor:
            distances = runner.batch_angular_distance(
                query,
                reference,
                query_chunk_size=query_chunk_size,
                reference_chunk_size=reference_chunk_size,
            )
            return (distances <= radius).sum(dim=1).to(dtype=torch.float32)

    monkeypatch.setattr(runner, "_NEIGHBOR", NeighborModuleWithoutPrivateTopk)
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda", dtype=torch.float32)
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )

    radius = runner.tune_radius_for_target_count_with_min_neighbors_gpu(
        query,
        reference,
        target_count=2,
        min_neighbors=2,
        query_chunk_size=1,
        reference_chunk_size=2,
        binary_steps=2,
    )

    distances = runner.batch_angular_distance(query, reference, query_chunk_size=1, reference_chunk_size=2)
    counts = (distances <= radius).sum(dim=1)
    assert int(counts.min().item()) >= 2


def test_run_parser_exposes_min_neighbors_default_and_override() -> None:
    runner = _load_runner()
    parser = runner.build_parser()

    default_args = parser.parse_args(
        [
            "run",
            "--cache-path",
            "cache",
            "--pooled-bank-root",
            "bank",
            "--model-name",
            "qwen3-vl-8b",
            "--benchmark",
            "popular",
        ]
    )
    override_args = parser.parse_args(
        [
            "run",
            "--cache-path",
            "cache",
            "--pooled-bank-root",
            "bank",
            "--model-name",
            "qwen3-vl-8b",
            "--benchmark",
            "popular",
            "--min-neighbors",
            "3",
        ]
    )

    assert default_args.min_neighbors == 2
    assert override_args.min_neighbors == 3


def test_save_radii_json_records_min_neighbors(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "radii.json"

    runner.save_radii_json(
        path=path,
        model_name="qwen3-vl-8b",
        benchmark="popular",
        layer_radii={9: torch.tensor(0.25)},
        target_count=30,
        min_neighbors=2,
        reference_chunk_size=128,
        radius_margin=2e-5,
        n_rows=4,
        limit_rows=None,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target_count"] == 30
    assert payload["min_neighbors"] == 2


def test_format_ingests_numeric_heldout_baseline_metrics_csv(tmp_path: Path) -> None:
    runner = _load_runner()
    metrics_csv = tmp_path / "metrics.csv"
    heldout_csv = tmp_path / "heldout.csv"
    baseline_csv = tmp_path / "baseline.csv"
    heldout_baseline_csv = tmp_path / "heldout_baseline.csv"
    heldout_baseline_metrics_csv = tmp_path / "heldout_baseline_metrics.csv"

    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "DASH-B",
                "benchmark_key": "dash-b",
                "method": "diag_maha",
                "status": "ok",
                "split_strategy": "image_grouped",
                "roc_auc": 0.7,
                "roc_auc_ci_lower": 0.6,
                "roc_auc_ci_upper": 0.8,
                "pr_auc": 0.45,
                "pr_auc_ci_lower": 0.40,
                "pr_auc_ci_upper": 0.50,
            }
        ]
    ).to_csv(metrics_csv, index=False)
    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "diag_maha",
                "status": "ok",
                "split_strategy": "object_heldout",
                "roc_auc": 0.65,
                "roc_auc_ci_lower": 0.55,
                "roc_auc_ci_upper": 0.75,
                "pr_auc": 0.30,
                "pr_auc_ci_lower": 0.20,
                "pr_auc_ci_upper": 0.40,
            }
        ]
    ).to_csv(heldout_csv, index=False)
    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "linear_probe",
                "status": "ok",
                "roc_auc": 0.8,
                "roc_auc_ci_lower": 0.7,
                "roc_auc_ci_upper": 0.9,
                "pr_auc": 0.5,
                "pr_auc_ci_lower": 0.4,
                "pr_auc_ci_upper": 0.6,
            }
        ]
    ).to_csv(baseline_csv, index=False)
    heldout_baseline_csv.write_text("model,benchmark,method,image_grouped,object_heldout\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "no_manifold",
                "status": "ok",
                "split_strategy": "object_heldout",
                "roc_auc": 0.61,
                "roc_auc_ci_lower": 0.51,
                "roc_auc_ci_upper": 0.71,
                "pr_auc": 0.21,
                "pr_auc_ci_lower": 0.11,
                "pr_auc_ci_upper": 0.31,
            },
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "query_local_k30",
                "status": "ok",
                "split_strategy": "object_heldout",
                "roc_auc": 0.62,
                "roc_auc_ci_lower": 0.52,
                "roc_auc_ci_upper": 0.72,
                "pr_auc": 0.22,
                "pr_auc_ci_lower": 0.12,
                "pr_auc_ci_upper": 0.32,
            },
        ]
    ).to_csv(heldout_baseline_metrics_csv, index=False)

    output = runner.write_formatted_outputs(
        metrics_csv=metrics_csv,
        heldout_csv=heldout_csv,
        baseline_csv=baseline_csv,
        heldout_baseline_csv=heldout_baseline_csv,
        heldout_baseline_metrics_csvs=(heldout_baseline_metrics_csv,),
        markdown_path=tmp_path / "main.md",
        csv_path=tmp_path / "main.csv",
        heldout_markdown_path=tmp_path / "heldout.md",
        heldout_csv_path=tmp_path / "heldout_out.csv",
        analysis_path=tmp_path / "analysis.md",
    )

    heldout_table = pd.read_csv(tmp_path / "heldout_out.csv")
    assert output["heldout_rows"] == 1
    assert "PR-AUC 0.2100 [0.1100, 0.3100]" in heldout_table.loc[0, "no_manifold"]


def test_format_outputs_writes_main_and_heldout_tables(tmp_path: Path) -> None:
    runner = _load_runner()
    metrics_csv = tmp_path / "metrics.csv"
    heldout_csv = tmp_path / "heldout.csv"
    baseline_csv = tmp_path / "baseline.csv"
    heldout_baseline_csv = tmp_path / "heldout_baseline.csv"

    metric_rows = []
    for method, pr_auc in {
        "radius_ball_isotropic": 0.40,
        "diag_maha": 0.45,
        "lowrank_maha": 0.42,
        "full_maha_shrink": 0.43,
    }.items():
        metric_rows.append(
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": method,
                "status": "ok",
                "split_strategy": "image_grouped",
                "roc_auc": 0.7,
                "roc_auc_ci_lower": 0.6,
                "roc_auc_ci_upper": 0.8,
                "pr_auc": pr_auc,
                "pr_auc_ci_lower": pr_auc - 0.05,
                "pr_auc_ci_upper": pr_auc + 0.05,
            }
        )
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "diag_maha",
                "status": "ok",
                "split_strategy": "object_heldout",
                "roc_auc": 0.65,
                "roc_auc_ci_lower": 0.55,
                "roc_auc_ci_upper": 0.75,
                "pr_auc": 0.30,
                "pr_auc_ci_lower": 0.20,
                "pr_auc_ci_upper": 0.40,
            }
        ]
    ).to_csv(heldout_csv, index=False)
    pd.DataFrame(
        [
            {
                "model": "qwen3-vl-8b",
                "benchmark": "POPE popular",
                "benchmark_key": "popular",
                "method": "linear_probe",
                "status": "ok",
                "roc_auc": 0.8,
                "roc_auc_ci_lower": 0.7,
                "roc_auc_ci_upper": 0.9,
                "pr_auc": 0.5,
                "pr_auc_ci_lower": 0.4,
                "pr_auc_ci_upper": 0.6,
            }
        ]
    ).to_csv(baseline_csv, index=False)
    heldout_baseline_csv.write_text(
        "model,benchmark,method,image_grouped,object_heldout\n"
        "qwen3-vl-8b,POPE popular,linear_probe,"
        "\"ROC 0.8000 [0.7000, 0.9000]; PR 0.5000 [0.4000, 0.6000]\","
        "\"ROC 0.7000 [0.6000, 0.8000]; PR 0.3500 [0.2500, 0.4500]\"\n",
        encoding="utf-8",
    )

    output = runner.write_formatted_outputs(
        metrics_csv=metrics_csv,
        heldout_csv=heldout_csv,
        baseline_csv=baseline_csv,
        heldout_baseline_csv=heldout_baseline_csv,
        markdown_path=tmp_path / "main.md",
        csv_path=tmp_path / "main.csv",
        heldout_markdown_path=tmp_path / "heldout.md",
        heldout_csv_path=tmp_path / "heldout_out.csv",
        analysis_path=tmp_path / "analysis.md",
        heldout_baseline_metrics_csvs=(),
    )

    assert output["main_rows"] == 1
    assert output["heldout_rows"] == 1
    assert "diag_maha" in (tmp_path / "main.md").read_text(encoding="utf-8")
    assert "linear_probe" in (tmp_path / "heldout.md").read_text(encoding="utf-8")
    assert "Decision Gate" in (tmp_path / "analysis.md").read_text(encoding="utf-8")
