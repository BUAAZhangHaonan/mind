#!/usr/bin/env python3
"""Create experiment plots from detector outputs and features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.visualization import (
    plot_ablation_bars,
    plot_drift_curves,
    plot_roc_curve,
    plot_wavelet_heatmap,
)


def build_plot_output_path(
    *,
    output_root: Path,
    experiment_name: str,
    plot_name: str,
) -> Path:
    return output_root / experiment_name / f"{plot_name}.png"


def build_plot_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    return {
        "drift": build_plot_output_path(
            output_root=output_root,
            experiment_name=experiment_name,
            plot_name="drift_curves",
        ),
        "wavelet": build_plot_output_path(
            output_root=output_root,
            experiment_name=experiment_name,
            plot_name="wavelet_heatmap",
        ),
        "roc": build_plot_output_path(
            output_root=output_root,
            experiment_name=experiment_name,
            plot_name="roc_curve",
        ),
        "ablation": build_plot_output_path(
            output_root=output_root,
            experiment_name=experiment_name,
            plot_name="ablation_bars",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--ablation-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = build_plot_paths(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
    )
    if args.features_path is None or args.results_path is None:
        for key, path in output_paths.items():
            print(f"{key}={path}")
        return 0

    features = pd.read_parquet(args.features_path)
    results = pd.read_csv(args.results_path)
    plot_drift_curves(features, output_path=output_paths["drift"])
    plot_wavelet_heatmap(features, output_path=output_paths["wavelet"])
    plot_roc_curve(
        y_true=results["label"].tolist(),
        y_score=results["score"].tolist(),
        output_path=output_paths["roc"],
    )
    if args.ablation_path is not None:
        plot_ablation_bars(pd.read_csv(args.ablation_path), output_path=output_paths["ablation"])
    for key, path in output_paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
