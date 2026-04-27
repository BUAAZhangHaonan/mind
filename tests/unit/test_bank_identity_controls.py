from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "compute_bank_identity_controls.py"
    assert script_path.exists(), "scripts/experiments/compute_bank_identity_controls.py should exist"
    spec = importlib.util.spec_from_file_location("compute_bank_identity_controls", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_import_is_cpu_only() -> None:
    sys.modules.pop("mind.models", None)
    script = _load_script()

    assert script.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert "mind.models" in sys.modules
    assert not hasattr(sys.modules["mind.models"], "__path__")


def test_bank_identity_table_formats_metric_cells_and_missing_rows(tmp_path: Path) -> None:
    script = _load_script()
    rows = [
        {
            "model": "qwen3-vl-8b",
            "benchmark": "DASH-B",
            "bank_type": "object_conditioned",
            "variant": "full_curve",
            "roc_auc": 0.91,
            "roc_auc_ci_lower": 0.81,
            "roc_auc_ci_upper": 0.96,
            "pr_auc": 0.72,
            "pr_auc_ci_lower": 0.62,
            "pr_auc_ci_upper": 0.82,
            "status": "ok",
        },
        {
            "model": "qwen3-vl-8b",
            "benchmark": "DASH-B",
            "bank_type": "shared",
            "variant": "no_manifold",
            "status": "missing_cache",
            "reason": "missing eval cache",
        },
    ]
    csv_path = tmp_path / "bank.csv"
    md_path = tmp_path / "bank.md"

    script.write_bank_identity_tables(rows, csv_path=csv_path, markdown_path=md_path)

    assert pd.read_csv(csv_path).shape[0] == 2
    text = md_path.read_text(encoding="utf-8")
    assert "ROC-AUC 0.9100 [0.8100, 0.9600]; PR-AUC 0.7200 [0.6200, 0.8200]" in text
    assert "missing_cache: missing eval cache" in text


def test_batched_no_manifold_matches_nearest_neighbor_formula() -> None:
    script = _load_script()
    cache_entries = [
        {
            "sample_id": "1",
            "image_id": 10,
            "label": 1,
            "parsed_answer": 0,
            "subset": "popular",
            "object_name": "cat",
            "selected_layers": [1],
            "layer_vectors": torch.tensor([[1.0, 0.0]]),
        }
    ]
    reference_bank = {"cat": {1: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}}
    reference_stats = {"cat": {1: {"neighbor_residual_mean": 0.0, "neighbor_residual_std": 1.0}}}

    frame = script.build_batched_no_manifold_feature_frame(
        cache_entries=cache_entries,
        reference_bank=reference_bank,
        reference_stats=reference_stats,
        bank_scope="object",
        batch_size=4,
    )

    assert frame.loc[0, "raw_drift_0"] == pytest.approx(0.0)
    assert frame.loc[0, "cal_mean_drift"] == pytest.approx(0.0)
    assert int(frame.loc[0, "label"]) == 0


def test_bank_analysis_reports_object_identity_value(tmp_path: Path) -> None:
    script = _load_script()
    rows = [
        {"model": "a", "benchmark": "DASH-B", "bank_type": "object_conditioned", "variant": "full_curve", "pr_auc": 0.80, "roc_auc": 0.90, "status": "ok"},
        {"model": "a", "benchmark": "DASH-B", "bank_type": "shared", "variant": "full_curve", "pr_auc": 0.70, "roc_auc": 0.88, "status": "ok"},
        {"model": "a", "benchmark": "DASH-B", "bank_type": "shuffled_object", "variant": "full_curve", "pr_auc": 0.60, "roc_auc": 0.86, "status": "ok"},
        {"model": "a", "benchmark": "DASH-B", "bank_type": "object_conditioned", "variant": "no_manifold", "pr_auc": 0.75, "roc_auc": 0.89, "status": "ok"},
    ]
    output_path = tmp_path / "analysis.md"

    script.write_bank_identity_analysis(rows, output_path=output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "object_conditioned beats shared on 1/1 completed full_curve comparisons" in text
    assert "object_conditioned beats shuffled_object on 1/1 completed full_curve comparisons" in text
    assert "On DASH-B, full_curve beats no_manifold on 1/1 completed bank comparisons" in text
