from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "compute_cache_geometry_distances.py"
    assert script_path.exists(), "scripts/experiments/compute_cache_geometry_distances.py should exist"
    spec = importlib.util.spec_from_file_location("compute_cache_geometry_distances", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_distance_functions_match_simple_geometry() -> None:
    script = _load_script()
    query = torch.tensor([1.0, 0.0])
    references = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert script.centroid_angular_distance(query, references) == pytest.approx(math.pi / 4.0)
    assert script.knn_angular_distance(query, references, k=1) == pytest.approx(0.0)
    assert script.centroid_euclidean_distance(query, references) == pytest.approx(math.sqrt(0.5))


def test_script_import_uses_cpu_only_lightweight_model_shim() -> None:
    script = _load_script()

    assert script.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert "mind.models" in sys.modules
    assert not hasattr(sys.modules["mind.models"], "__path__")
    assert sys.modules["mind.models"].parse_yes_no_answer("yes") == 1


def test_prompt_full_curve_features_use_raw_curve_and_calibrated_summaries() -> None:
    script = _load_script()
    raw_curve = np.asarray([2.0, 6.0, 10.0], dtype=np.float32)
    stats = {
        3: {"centroid_euclidean_mean": 1.0, "centroid_euclidean_std": 1.0},
        5: {"centroid_euclidean_mean": 2.0, "centroid_euclidean_std": 2.0},
        7: {"centroid_euclidean_mean": 4.0, "centroid_euclidean_std": 3.0},
    }

    features = script.build_prompt_full_curve_features(
        raw_curve=raw_curve,
        selected_layers=[3, 5, 7],
        layer_stats=stats,
        distance_name="centroid_euclidean",
    )

    assert features["raw_drift_0"] == pytest.approx(2.0)
    assert features["raw_drift_2"] == pytest.approx(10.0)
    assert features["cal_mean_drift"] == pytest.approx((1.0 + 2.0 + 2.0) / 3.0)
    assert features["cal_max_drift"] == pytest.approx(2.0)
    assert features["cal_final_drift"] == pytest.approx(2.0)
    assert features["cal_drift_slope"] == pytest.approx(0.5)
    assert features["cal_drift_variance"] == pytest.approx(np.asarray([1.0, 2.0, 2.0], dtype=np.float32).var())
    assert "cal_drift_0" not in features


def test_comparison_table_formats_metric_cells_and_missing_rows(tmp_path: Path) -> None:
    script = _load_script()
    rows = [
        {
            "model": "qwen3-vl-8b",
            "benchmark": "POPE popular",
            "distance_type": "centroid_angular",
            "roc_auc": 0.71,
            "roc_auc_ci_lower": 0.61,
            "roc_auc_ci_upper": 0.81,
            "pr_auc": 0.42,
            "pr_auc_ci_lower": 0.32,
            "pr_auc_ci_upper": 0.52,
            "status": "ok",
        },
        {
            "model": "qwen3-vl-8b",
            "benchmark": "DASH-B",
            "distance_type": "centroid_angular",
            "status": "missing_cache",
            "reason": "missing eval cache",
        },
    ]
    csv_path = tmp_path / "curvature.csv"
    md_path = tmp_path / "curvature.md"

    script.write_curvature_tables(rows, csv_path=csv_path, markdown_path=md_path)

    csv_rows = pd.read_csv(csv_path)
    assert csv_rows.shape[0] == 2
    markdown = md_path.read_text(encoding="utf-8")
    assert "ROC-AUC 0.7100 [0.6100, 0.8100]; PR-AUC 0.4200 [0.3200, 0.5200]" in markdown
    assert "missing_cache: missing eval cache" in markdown


def test_analysis_selects_linear_verdict_when_angular_never_wins(tmp_path: Path) -> None:
    script = _load_script()
    rows = [
        {"model": "a", "benchmark": "x", "distance_type": "euclidean_pca_residual", "pr_auc": 0.80, "roc_auc": 0.80, "status": "ok"},
        {"model": "a", "benchmark": "x", "distance_type": "centroid_angular", "pr_auc": 0.70, "roc_auc": 0.70, "status": "ok"},
        {"model": "a", "benchmark": "x", "distance_type": "knn_angular_k10", "pr_auc": 0.60, "roc_auc": 0.60, "status": "ok"},
        {"model": "a", "benchmark": "x", "distance_type": "centroid_euclidean", "pr_auc": 0.75, "roc_auc": 0.75, "status": "ok"},
        {"model": "a", "benchmark": "x", "distance_type": "no_manifold", "pr_auc": 0.76, "roc_auc": 0.76, "status": "ok"},
    ]
    output_path = tmp_path / "analysis.md"

    script.write_curvature_analysis(rows, output_path=output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "Angular-distance variants do not beat local PCA on any completed comparison." in text
    assert "The evidence points to a mostly linear signal." in text
