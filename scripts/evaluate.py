#!/usr/bin/env python3
"""Aggregate detector outputs into experiment reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mind.evaluation import compute_binary_metrics, evaluate_by_subset, write_metrics_report, write_results_table


def build_report_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    report_root = output_root / experiment_name
    return {
        "metrics": report_root / "metrics.json",
        "results": report_root / "results.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--label-overrides", type=Path, default=None)
    return parser


def load_results_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported results input format: {path.suffix}")


def load_label_overrides(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported label override format: {path.suffix}")


def apply_label_overrides(frame: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    relabel_map = dict(zip(overrides["sample_id"], overrides["label"]))
    relabeled = frame.copy()
    relabeled["label"] = relabeled.apply(
        lambda row: relabel_map.get(row["sample_id"], row["label"]),
        axis=1,
    )
    return relabeled


def run_evaluation(
    *,
    input_path: Path,
    output_root: Path,
    experiment_name: str,
    label_overrides_path: Path | None = None,
) -> dict[str, Path]:
    frame = load_results_frame(input_path)
    if label_overrides_path is not None:
        frame = apply_label_overrides(frame, load_label_overrides(label_overrides_path))

    metrics_payload = {
        "overall": compute_binary_metrics(
            y_true=frame["label"],
            y_pred=frame["prediction"],
            y_score=frame["score"],
        ),
        "by_subset": evaluate_by_subset(frame),
    }
    output_paths = build_report_paths(
        output_root=output_root,
        experiment_name=experiment_name,
    )
    write_metrics_report(metrics_payload, output_paths["metrics"])
    write_results_table(frame, output_paths["results"])
    return output_paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_evaluation(
        input_path=args.input_path,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        label_overrides_path=args.label_overrides,
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
