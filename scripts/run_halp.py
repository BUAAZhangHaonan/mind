#!/usr/bin/env python3
"""Run HALP reproduction on cached pre-generation readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mind.comparators import HALPProbeConfig
from mind.comparators.halp import (
    build_halp_probe_frame,
    evaluate_halp_nested_from_readout_entries,
    summarize_halp_results,
)
from mind.evaluation.baselines import (
    apply_label_overrides_to_entries,
    load_cache_entries,
    validate_object_heldout_reference_support,
)
from mind.evaluation.metrics import write_results_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split-strategy", choices=["row", "image_grouped", "object_heldout"], default="image_grouped")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--inner-fold-candidates", default="3,2")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--label-overrides", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dims", default="512,256,128")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--bank-scope", choices=["object", "shared", "shuffled_object"], default="object")
    return parser


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_output_paths(output_root: Path, experiment_name: str) -> dict[str, Path]:
    root = output_root / experiment_name
    return {
        "metrics": root / "halp.json",
        "results": root / "halp_results.csv",
        "selection": root / "halp_selection.csv",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readout_entries = load_cache_entries(args.readout_path)
    if args.label_overrides is not None:
        readout_entries = apply_label_overrides_to_entries(readout_entries, args.label_overrides)

    supported_object_names: list[str] | None = None
    if args.split_strategy == "object_heldout":
        if args.reference_root is None or not args.model_name:
            raise ValueError("--reference-root and --model-name are required for object_heldout HALP.")
        base_frame = build_halp_probe_frame(readout_entries, "vision_only")
        filtered_frame, support = validate_object_heldout_reference_support(
            base_frame,
            reference_root=args.reference_root,
            model_name=args.model_name,
            bank_scope=args.bank_scope,
            num_folds=args.num_folds,
        )
        supported_object_names = sorted(filtered_frame["object_name"].drop_duplicates().astype(str).tolist())
        print(
            "[run_halp] object_heldout support "
            f"frame_objects={support['frame_object_count']} "
            f"supported_objects={support['supported_object_count']} "
            f"retained_rows={support['retained_row_count']}"
        )
    metrics, results, selection = evaluate_halp_nested_from_readout_entries(
        readout_entries,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
        inner_candidate_folds=tuple(_parse_int_list(args.inner_fold_candidates)),
        probe_config=HALPProbeConfig(
            hidden_dims=tuple(_parse_int_list(args.hidden_dims)),
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            random_state=args.random_state,
        ),
        supported_object_names=supported_object_names,
    )
    summary = summarize_halp_results(
        results,
        split_strategy=args.split_strategy,
        bootstrap_resamples=args.bootstrap_resamples,
        random_state=args.random_state,
    )
    payload = {
        **metrics,
        "confidence_intervals": summary["confidence_intervals"],
        "selected_probe_counts": (
            results[["fold", "selected_probe"]]
            .drop_duplicates()
            ["selected_probe"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    output_paths = build_output_paths(args.output_root, args.experiment_name)
    output_paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["metrics"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_results_table(results, output_paths["results"], extra_columns=("selected_probe",))
    selection.to_csv(output_paths["selection"], index=False)
    print(output_paths["metrics"])
    print(output_paths["results"])
    print(output_paths["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
