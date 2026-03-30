#!/usr/bin/env python3
"""Compute MIND baselines and ablation summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mind.evaluation.baselines import (
    build_linear_probe_frame,
    build_no_manifold_feature_frame,
    build_raw_model_yes_no_baseline,
    drift_only_columns,
    evaluate_feature_frame,
    feature_columns,
    load_cache_entries,
    load_reference_bank,
    load_reference_stats,
)


def build_output_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    root = output_root / experiment_name
    return {
        "baselines": root / "baselines.json",
        "ablations": root / "ablations.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--bank-scope", choices=["object", "shared"], default="object")
    parser.add_argument(
        "--split-strategy",
        choices=["row", "image_grouped", "object_heldout"],
        default="row",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = pd.read_parquet(args.features_path)
    cache_entries = load_cache_entries(args.cache_path)
    reference_bank = load_reference_bank(
        args.reference_root,
        args.model_name,
        bank_scope=args.bank_scope,
    )
    reference_stats = load_reference_stats(
        args.reference_root,
        args.model_name,
        bank_scope=args.bank_scope,
    )
    no_manifold_frame = build_no_manifold_feature_frame(
        cache_entries=cache_entries,
        reference_bank=reference_bank,
        reference_stats=reference_stats,
        bank_scope=args.bank_scope,
    )
    linear_probe_frame = build_linear_probe_frame(cache_entries)

    baselines: dict[str, object] = {}
    full_metrics, _ = evaluate_feature_frame(
        features,
        columns=feature_columns(features),
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
    )
    drift_metrics, _ = evaluate_feature_frame(
        features,
        columns=drift_only_columns(features),
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
    )
    no_manifold_metrics, _ = evaluate_feature_frame(
        no_manifold_frame,
        columns=feature_columns(no_manifold_frame),
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
    )
    linear_probe_metrics, _ = evaluate_feature_frame(
        linear_probe_frame,
        columns=feature_columns(linear_probe_frame),
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
    )

    baselines["full"] = full_metrics
    baselines["drift_only"] = drift_metrics
    baselines["no_manifold"] = no_manifold_metrics
    baselines["linear_probe"] = linear_probe_metrics
    baselines["raw_model_yes_no"] = build_raw_model_yes_no_baseline(cache_entries)
    baselines["bank_scope"] = args.bank_scope

    ablations = pd.DataFrame(
        [
            {"variant": "full", **full_metrics},
            {"variant": "drift_only", **drift_metrics},
            {"variant": "no_manifold", **no_manifold_metrics},
            {"variant": "linear_probe", **linear_probe_metrics},
        ]
    )

    output_paths = build_output_paths(output_root=args.output_root, experiment_name=args.experiment_name)
    output_paths["baselines"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["baselines"].write_text(json.dumps(baselines, indent=2, sort_keys=True), encoding="utf-8")
    ablations.to_csv(output_paths["ablations"], index=False)
    print(output_paths["baselines"])
    print(output_paths["ablations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
