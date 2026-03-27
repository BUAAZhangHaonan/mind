from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from mind.visualization import (
    plot_ablation_bars,
    plot_drift_curve_comparison,
    plot_roc_curve,
    plot_wavelet_energy_heatmap,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT_PATH)
plot_results = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(plot_results)


def test_plot_drift_curve_comparison_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "drift.png"

    plot_drift_curve_comparison(
        grounded_curve=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        hallucinated_curve=np.array([0.1, 0.4, 0.8], dtype=np.float32),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_wavelet_energy_heatmap_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "heatmap.png"

    plot_wavelet_energy_heatmap(
        energy=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        output_path=output_path,
        row_labels=["grounded", "hallucinated"],
        column_labels=["l1", "l2"],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_roc_curve_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "roc.png"

    plot_roc_curve(
        fpr=np.array([0.0, 0.1, 1.0], dtype=np.float32),
        tpr=np.array([0.0, 0.9, 1.0], dtype=np.float32),
        roc_auc=0.97,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ablation_bars_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "ablations.png"
    frame = pd.DataFrame(
        [
            {"variant": "full", "accuracy": 0.91},
            {"variant": "no-wavelet", "accuracy": 0.82},
            {"variant": "no-manifold", "accuracy": 0.77},
        ]
    )

    plot_ablation_bars(frame, output_path=output_path, metric="accuracy")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_plot_output_path_uses_experiment_directory(tmp_path: Path) -> None:
    output_path = plot_results.build_plot_output_path(
        output_root=tmp_path,
        experiment_name="smoke-qwen3-vl",
        plot_name="roc",
    )

    assert output_path == tmp_path / "smoke-qwen3-vl" / "roc.png"
