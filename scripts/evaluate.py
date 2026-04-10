#!/usr/bin/env python3
"""Evaluate detector outputs and write experiment reports."""

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

from mind.evaluation import (
    compute_binary_metrics,
    compute_object_hallucination_label,
    evaluate_by_subset,
    write_metrics_report,
)


def build_report_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    report_root = output_root / experiment_name
    return {
        "metrics": report_root / "metrics.json",
        "results": report_root / "results.csv",
    }


def apply_label_overrides(results: pd.DataFrame, overrides: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(overrides, pd.DataFrame):
        override_frame = overrides
    else:
        override_frame = (
            pd.read_parquet(overrides)
            if overrides.suffix == ".parquet"
            else pd.read_json(overrides, lines=True)
        )
    override_columns = [column for column in ["sample_id", "label"] if column in override_frame.columns]
    if override_columns != ["sample_id", "label"]:
        raise ValueError("Label override file must include sample_id and label columns.")
    merged = results.drop(columns=["label"], errors="ignore").merge(
        override_frame[["sample_id", "label"]].rename(columns={"label": "override_label"}),
        on="sample_id",
        how="left",
    )
    if "ground_truth_label" in merged.columns and "answer_label" in merged.columns:
        merged["ground_truth_label"] = (
            merged["override_label"].fillna(merged["ground_truth_label"]).astype(int)
        )
        merged["label"] = [
            compute_object_hallucination_label(
                ground_truth_label=int(ground_truth),
                answer_label=None if int(answer_label) < 0 else int(answer_label),
            )
            for ground_truth, answer_label in zip(
                merged["ground_truth_label"].tolist(),
                merged["answer_label"].tolist(),
            )
        ]
        return merged.drop(columns=["override_label"])
    merged["label"] = merged["override_label"].fillna(results["label"]).astype(int)
    merged = merged.drop(columns=["override_label"])
    return merged


def build_metrics(results: pd.DataFrame) -> dict[str, object]:
    overall = compute_binary_metrics(
        y_true=results["label"].tolist(),
        y_pred=results["prediction"].tolist(),
        y_score=results["score"].tolist(),
    )
    per_subset = evaluate_by_subset(results)
    return {"overall": overall, "by_subset": per_subset}


def run_evaluation(
    *,
    input_path: Path,
    output_root: Path,
    experiment_name: str,
    label_overrides: Path | pd.DataFrame | None = None,
) -> dict[str, Path]:
    results = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    if label_overrides is not None:
        results = apply_label_overrides(results, label_overrides)
    metrics = build_metrics(results)
    output_paths = build_report_paths(output_root=output_root, experiment_name=experiment_name)
    output_paths["results"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_paths["results"], index=False)
    write_metrics_report(metrics, output_paths["metrics"])
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--label-overrides", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = build_report_paths(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
    )
    if args.input_path is None:
        for key, path in output_paths.items():
            print(f"{key}={path}")
        return 0

    output_paths = run_evaluation(
        input_path=args.input_path,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        label_overrides=args.label_overrides,
    )
    print(output_paths["metrics"])
    print(output_paths["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
