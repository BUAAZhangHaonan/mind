#!/usr/bin/env python3
"""Train lightweight hallucination detectors for MIND."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from mind.detectors import fit_logistic_detector


def build_feature_output_path(
    *,
    output_root: Path,
    experiment_name: str,
    split: str,
) -> Path:
    return output_root / experiment_name / f"{split}.parquet"


def build_detector_output_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    root = output_root / experiment_name
    return {
        "checkpoint": root / "detector.joblib",
        "results": root / "results.csv",
    }


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column
        not in {
            "sample_id",
            "label",
            "subset",
            "object_name",
            "ground_truth_label",
            "answer_label",
        }
    ]


def split_train_eval_frame(
    frame: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, eval_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )
    return train_frame.reset_index(drop=True), eval_frame.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--eval-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_path is None:
        if not args.split:
            raise ValueError("--split is required when only resolving output paths.")
        print(
            build_feature_output_path(
                output_root=args.output_root,
                experiment_name=args.experiment_name,
                split=args.split,
            )
        )
        return 0

    train_frame = pd.read_parquet(args.train_path)
    if train_frame["label"].nunique() < 2:
        raise ValueError("Training frame must contain both hallucinated and non-hallucinated samples.")
    if args.eval_path is None:
        train_frame, eval_frame = split_train_eval_frame(
            train_frame,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    else:
        eval_frame = pd.read_parquet(args.eval_path)
    columns = feature_columns(train_frame)
    detector = fit_logistic_detector(train_frame[columns].to_numpy(), train_frame["label"].to_numpy())
    probabilities = detector.predict_proba(eval_frame[columns].to_numpy())[:, 1]
    predictions = detector.predict(eval_frame[columns].to_numpy())
    output_paths = build_detector_output_paths(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
    )
    output_paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"detector": detector, "columns": columns}, output_paths["checkpoint"])
    eval_frame.assign(prediction=predictions, score=probabilities).to_csv(
        output_paths["results"],
        index=False,
    )
    print(output_paths["checkpoint"])
    print(output_paths["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
