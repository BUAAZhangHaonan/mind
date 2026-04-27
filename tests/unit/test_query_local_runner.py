from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_script():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiments"
        / "run_query_local_bank.py"
    )
    assert script_path.exists(), "scripts/experiments/run_query_local_bank.py should exist"
    spec = importlib.util.spec_from_file_location("run_query_local_bank", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_row(
    script,
    *,
    model: str,
    benchmark: str,
    method: str,
    roc_auc: float = 0.71,
    pr_auc: float = 0.42,
) -> dict[str, object]:
    return {
        "model": model,
        "benchmark": script.BENCHMARK_LABELS[benchmark],
        "benchmark_key": benchmark,
        "method": method,
        "roc_auc": roc_auc,
        "roc_auc_ci_lower": roc_auc - 0.01,
        "roc_auc_ci_upper": roc_auc + 0.01,
        "pr_auc": pr_auc,
        "pr_auc_ci_lower": pr_auc - 0.01,
        "pr_auc_ci_upper": pr_auc + 0.01,
        "status": "ok",
        "n_rows": 12,
        "split_strategy": "image_grouped",
        "feature_source": "features.parquet",
        "metrics_path": "metrics.json",
    }


def _complete_rows(script) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in script.DEFAULT_MODELS:
        for benchmark in script.DEFAULT_BENCHMARKS:
            for method in script.METHODS:
                rows.append(
                    _metric_row(
                        script,
                        model=model,
                        benchmark=benchmark,
                        method=method,
                    )
                )
    return rows


def _phase2_row(
    script,
    *,
    model: str,
    benchmark: str,
    bank_type: str,
    variant: str,
    roc_auc: float,
    pr_auc: float,
) -> dict[str, object]:
    return {
        "model": model,
        "benchmark": script.BENCHMARK_LABELS[benchmark],
        "bank_type": bank_type,
        "variant": variant,
        "roc_auc": roc_auc,
        "roc_auc_ci_lower": roc_auc - 0.01,
        "roc_auc_ci_upper": roc_auc + 0.01,
        "pr_auc": pr_auc,
        "pr_auc_ci_lower": pr_auc - 0.01,
        "pr_auc_ci_upper": pr_auc + 0.01,
        "status": "ok",
        "n_rows": 12,
        "split_strategy": "image_grouped",
        "feature_source": "phase2.parquet",
        "metrics_path": "phase2.json",
    }


def test_validate_rows_requires_complete_ok_grid() -> None:
    script = _load_script()
    rows = _complete_rows(script)

    script.validate_final_rows(rows)

    with pytest.raises(ValueError, match="exactly 48"):
        script.validate_final_rows(rows[:-1])

    bad_status = rows.copy()
    bad_status[0] = {**bad_status[0], "status": "missing_cache", "reason": "missing eval cache"}
    with pytest.raises(ValueError, match="non-ok"):
        script.validate_final_rows(bad_status)


def test_table_writer_formats_wide_method_cells(tmp_path: Path) -> None:
    script = _load_script()
    rows = _complete_rows(script)
    rows[0] = {
        **rows[0],
        "roc_auc": 0.81234,
        "roc_auc_ci_lower": 0.71234,
        "roc_auc_ci_upper": 0.91234,
        "pr_auc": 0.52346,
        "pr_auc_ci_lower": 0.42346,
        "pr_auc_ci_upper": 0.62346,
    }
    csv_path = tmp_path / "query_local.csv"
    md_path = tmp_path / "query_local.md"

    script.write_query_local_tables(rows, csv_path=csv_path, markdown_path=md_path)

    csv_rows = pd.read_csv(csv_path)
    assert csv_rows.shape[0] == 48
    markdown = md_path.read_text(encoding="utf-8")
    assert "| model | benchmark | object_cond | shared | shuffled | query_local_k30 | no_manifold | linear_probe |" in markdown
    assert "ROC-AUC 0.8123 [0.7123, 0.9123]; PR-AUC 0.5235 [0.4235, 0.6235]" in markdown
    assert "missing_cache" not in markdown


def test_phase2_rows_map_to_static_comparison_methods() -> None:
    script = _load_script()
    rows: list[dict[str, object]] = []
    for method, bank_type, variant, roc_auc, pr_auc in [
        ("object_cond", "object_conditioned", "full_curve", 0.80, 0.50),
        ("shared", "shared", "full_curve", 0.81, 0.51),
        ("shuffled", "shuffled_object", "full_curve", 0.82, 0.52),
        ("no_manifold", "object_conditioned", "no_manifold", 0.83, 0.53),
    ]:
        rows.append(
            _phase2_row(
                script,
                model="qwen3-vl-8b",
                benchmark="popular",
                bank_type=bank_type,
                variant=variant,
                roc_auc=roc_auc,
                pr_auc=pr_auc,
            )
        )

    mapped = script.phase2_rows_to_query_local_methods(rows, no_manifold_bank_type="object_conditioned")

    assert [row["method"] for row in mapped] == ["object_cond", "shared", "shuffled", "no_manifold"]
    assert [row["roc_auc"] for row in mapped] == [0.80, 0.81, 0.82, 0.83]
    assert all(row["status"] == "ok" for row in mapped)


def test_build_commands_use_gpu_scripts_and_gpu_zero(tmp_path: Path) -> None:
    script = _load_script()
    job = script.ExperimentJob(
        model="qwen3-vl-8b",
        benchmark="popular",
        method="query_local_k30",
        feature_path=tmp_path / "features.parquet",
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.parquet",
    )

    feature_command = script.build_query_local_feature_command(
        job,
        python_executable="python",
        feature_builder=Path("scripts/experiments/build_query_local_features.py"),
        cache_path=Path("cache"),
        pooled_bank_root=Path("pooled"),
        device="cuda",
        batch_size=32,
        reference_chunk_size=4096,
        k_neighbors=30,
        label_overrides=Path("labels.parquet"),
    )
    linear_job = script.ExperimentJob(
        model="qwen3-vl-8b",
        benchmark="popular",
        method="linear_probe",
        feature_path=tmp_path / "linear.parquet",
        metrics_path=tmp_path / "linear.json",
        predictions_path=tmp_path / "linear_predictions.parquet",
    )
    linear_command = script.build_linear_feature_command(
        linear_job,
        python_executable="python",
        feature_builder=Path("scripts/experiments/build_gpu_linear_probe_features.py"),
        cache_path=Path("cache"),
        device="cuda",
        batch_size=32,
        label_overrides=Path("labels.parquet"),
    )
    detector_command = script.build_detector_command(
        job,
        python_executable="python",
        detector=Path("scripts/experiments/train_gpu_detector.py"),
        device="cuda",
        bootstrap_resamples=100,
        num_folds=5,
        random_state=13,
        max_iter=50,
        columns="full_curve",
    )
    env = script.gpu_env()

    assert "scripts/experiments/build_query_local_features.py" in feature_command
    assert "--pooled-bank-root" in feature_command
    assert "--k-neighbors" in feature_command
    assert "30" in feature_command
    assert feature_command[feature_command.index("--label-overrides") + 1] == "labels.parquet"
    assert "scripts/experiments/build_gpu_linear_probe_features.py" in linear_command
    assert linear_command[linear_command.index("--label-overrides") + 1] == "labels.parquet"
    assert "scripts/experiments/train_gpu_detector.py" in detector_command
    assert detector_command[detector_command.index("--columns") + 1] == "full_curve"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_query_local_method_uses_all_feature_columns() -> None:
    script = _load_script()

    assert script.detector_columns_for_method("query_local_k30", linear_columns="all_features") == "all_features"
    assert script.detector_columns_for_method("linear_probe", linear_columns="all_features") == "all_features"
