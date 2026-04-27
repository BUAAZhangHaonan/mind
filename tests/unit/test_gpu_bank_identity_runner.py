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
        / "run_gpu_bank_identity.py"
    )
    assert script_path.exists(), "scripts/experiments/run_gpu_bank_identity.py should exist"
    spec = importlib.util.spec_from_file_location("run_gpu_bank_identity", script_path)
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
    bank_type: str,
    variant: str,
    roc_auc: float = 0.72,
    pr_auc: float = 0.41,
) -> dict[str, object]:
    return {
        "model": model,
        "benchmark": script.BENCHMARK_LABELS[benchmark],
        "benchmark_key": benchmark,
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
        "feature_source": "features.parquet",
        "metrics_path": "metrics.json",
    }


def _complete_rows(script) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in script.DEFAULT_MODELS:
        for benchmark in script.DEFAULT_BENCHMARKS:
            for bank_type in script.BANK_TYPES:
                for variant in script.VARIANTS:
                    rows.append(
                        _metric_row(
                            script,
                            model=model,
                            benchmark=benchmark,
                            bank_type=bank_type,
                            variant=variant,
                        )
                    )
    return rows


def test_validate_rows_requires_complete_ok_grid() -> None:
    script = _load_script()
    rows = _complete_rows(script)

    script.validate_final_rows(rows)

    missing_one = rows[:-1]
    with pytest.raises(ValueError, match="exactly 72"):
        script.validate_final_rows(missing_one)

    bad_status = rows.copy()
    bad_status[0] = {**bad_status[0], "status": "missing_cache", "reason": "missing eval cache"}
    with pytest.raises(ValueError, match="non-ok"):
        script.validate_final_rows(bad_status)


def test_table_writer_formats_metric_cells(tmp_path: Path) -> None:
    script = _load_script()
    rows = _complete_rows(script)
    rows[0] = {
        **rows[0],
        "roc_auc": 0.71234,
        "roc_auc_ci_lower": 0.61234,
        "roc_auc_ci_upper": 0.81234,
        "pr_auc": 0.42346,
        "pr_auc_ci_lower": 0.32346,
        "pr_auc_ci_upper": 0.52346,
    }
    csv_path = tmp_path / "bank_identity_v2.csv"
    md_path = tmp_path / "bank_identity_v2.md"

    script.write_bank_identity_tables(rows, csv_path=csv_path, markdown_path=md_path)

    csv_rows = pd.read_csv(csv_path)
    assert csv_rows.shape[0] == 72
    markdown = md_path.read_text(encoding="utf-8")
    assert "ROC-AUC 0.7123 [0.6123, 0.8123]; PR-AUC 0.4235 [0.3235, 0.5235]" in markdown
    assert "missing_cache" not in markdown


def test_discrepancy_detection_requires_gap_and_non_overlapping_ci() -> None:
    script = _load_script()
    gpu = _metric_row(
        script,
        model="qwen3-vl-8b",
        benchmark="popular",
        bank_type="object_conditioned",
        variant="full_curve",
        roc_auc=0.80,
    )
    cpu = {
        **gpu,
        "roc_auc": 0.75,
        "roc_auc_ci_lower": 0.73,
        "roc_auc_ci_upper": 0.77,
    }
    assert script.find_roc_discrepancies([gpu], [cpu]) == [
        {
            "model": "qwen3-vl-8b",
            "benchmark": "POPE popular",
            "bank_type": "object_conditioned",
            "variant": "full_curve",
            "gpu_roc_auc": 0.80,
            "cpu_roc_auc": 0.75,
            "difference": 0.05,
            "gpu_ci": (0.79, 0.81),
            "cpu_ci": (0.73, 0.77),
        }
    ]

    overlapping_cpu = {**cpu, "roc_auc_ci_upper": 0.795}
    assert script.find_roc_discrepancies([gpu], [overlapping_cpu]) == []
