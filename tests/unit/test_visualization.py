from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from mind.visualization import (
    plot_ablation_bars,
    plot_drift_curves,
    plot_roc_curve,
    plot_wavelet_heatmap,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT_PATH)
plot_results = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(plot_results)


def test_plot_drift_curves_writes_png(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"label": 0, "drift_0": 0.1, "drift_1": 0.2, "drift_2": 0.3},
            {"label": 1, "drift_0": 0.4, "drift_1": 0.5, "drift_2": 0.6},
        ]
    )
    output_path = tmp_path / "drift.png"

    plot_drift_curves(frame, output_path=output_path)

    assert output_path.exists()


def test_plot_wavelet_heatmap_writes_png(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"detail_energy_l1": 0.2, "detail_energy_l2": 0.1, "approx_energy": 0.7},
            {"detail_energy_l1": 0.4, "detail_energy_l2": 0.3, "approx_energy": 0.9},
        ]
    )
    output_path = tmp_path / "heatmap.png"

    plot_wavelet_heatmap(frame, output_path=output_path)

    assert output_path.exists()


def test_plot_roc_curve_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "roc.png"

    plot_roc_curve(y_true=[0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9], output_path=output_path)

    assert output_path.exists()


def test_plot_ablation_bars_writes_png(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"variant": "full", "metric": 0.82},
            {"variant": "no_wavelet", "metric": 0.74},
        ]
    )
    output_path = tmp_path / "ablation.png"

    plot_ablation_bars(frame, output_path=output_path)

    assert output_path.exists()


def test_build_plot_paths_returns_expected_png_locations(tmp_path: Path) -> None:
    paths = plot_results.build_plot_paths(output_root=tmp_path, experiment_name="smoke-qwen3-vl")

    assert paths["drift"] == tmp_path / "smoke-qwen3-vl" / "drift_curves.png"
    assert paths["wavelet"] == tmp_path / "smoke-qwen3-vl" / "wavelet_heatmap.png"
    assert paths["roc"] == tmp_path / "smoke-qwen3-vl" / "roc_curve.png"
    assert paths["ablation"] == tmp_path / "smoke-qwen3-vl" / "ablation_bars.png"
