#!/usr/bin/env python3
"""Train lightweight hallucination detectors for MIND."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from mind.detectors import fit_logistic_detector
from mind.evaluation.baselines import build_train_eval_splits


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
            "image_id",
            "label",
            "subset",
            "object_name",
            "ground_truth_label",
            "answer_label",
            "fold",
        }
    ]


def train_detector_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    split_strategy: str = "row",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
) -> tuple[dict[str, object], pd.DataFrame]:
    detector_payloads: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    for fold, train_frame, eval_frame in build_train_eval_splits(
        frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    ):
        detector = fit_logistic_detector(
            train_frame[columns].to_numpy(),
            train_frame["label"].to_numpy(),
        )
        probabilities = detector.predict_proba(eval_frame[columns].to_numpy())[:, 1]
        predictions = detector.predict(eval_frame[columns].to_numpy())
        detector_payloads.append({"fold": fold, "detector": detector})
        result_frames.append(
            eval_frame.assign(prediction=predictions, score=probabilities, fold=fold).reset_index(
                drop=True
            )
        )
    results = pd.concat(result_frames, ignore_index=True)
    if "sample_id" in results.columns:
        results = results.sort_values("sample_id").reset_index(drop=True)
    payload: dict[str, object] = {
        "columns": columns,
        "split_strategy": split_strategy,
        "detectors": detector_payloads,
    }
    if len(detector_payloads) == 1:
        payload["detector"] = detector_payloads[0]["detector"]
    return payload, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--eval-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--split-strategy",
        choices=["row", "image_grouped", "object_heldout"],
        default="row",
    )
    parser.add_argument("--num-folds", type=int, default=5)
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
    columns = feature_columns(train_frame)
    if args.eval_path is None:
        checkpoint_payload, results = train_detector_frame(
            train_frame,
            columns=columns,
            split_strategy=args.split_strategy,
            test_size=args.test_size,
            random_state=args.random_state,
            num_folds=args.num_folds,
        )
    else:
        eval_frame = pd.read_parquet(args.eval_path)
        detector = fit_logistic_detector(train_frame[columns].to_numpy(), train_frame["label"].to_numpy())
        probabilities = detector.predict_proba(eval_frame[columns].to_numpy())[:, 1]
        predictions = detector.predict(eval_frame[columns].to_numpy())
        results = eval_frame.assign(prediction=predictions, score=probabilities, fold=0).reset_index(
            drop=True
        )
        checkpoint_payload = {
            "columns": columns,
            "split_strategy": "explicit_eval",
            "detectors": [{"fold": 0, "detector": detector}],
            "detector": detector,
        }
    output_paths = build_detector_output_paths(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
    )
    output_paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint_payload, output_paths["checkpoint"])
    results.to_csv(output_paths["results"], index=False)
    print(output_paths["checkpoint"])
    print(output_paths["results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
