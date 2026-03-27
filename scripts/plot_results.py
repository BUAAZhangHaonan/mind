#!/usr/bin/env python3
"""Create plot artifact paths for MIND experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_plot_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    plot_root = output_root / experiment_name
    paths = {
        "drift": plot_root / "drift_curves.png",
        "heatmap": plot_root / "wavelet_heatmap.png",
        "wavelet": plot_root / "wavelet_heatmap.png",
        "roc": plot_root / "roc_curve.png",
        "ablation": plot_root / "ablation_bars.png",
    }
    return paths


def build_plot_output_path(*, output_root: Path, experiment_name: str, plot_name: str) -> Path:
    return output_root / experiment_name / f"{plot_name}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for key, path in build_plot_paths(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
    ).items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
