#!/usr/bin/env python3
"""Compute MIND baselines and ablation summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoProcessor

from mind.evaluation.baselines import (
    GROUP_COLUMN_BY_STRATEGY,
    build_feature_variant_frames,
    build_linear_probe_frame,
    build_no_manifold_feature_frame,
    build_output_baseline_frame,
    build_raw_model_yes_no_baseline,
    compute_bootstrap_confidence_intervals,
    drift_only_columns,
    evaluate_feature_frame,
    evaluate_feature_frame_across_random_states,
    feature_columns,
    load_cache_entries,
    load_reference_bank,
    load_reference_stats,
    resolve_yes_no_token_ids,
)


KNOWN_MODEL_IDS = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B-HF",
    "llava-onevision-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "molmo-7b-d-0924": "allenai/Molmo-7B-D-0924",
}


def build_output_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    root = output_root / experiment_name
    return {
        "baselines": root / "baselines.json",
        "ablations": root / "ablations.csv",
        "split_sensitivity": root / "split_sensitivity.csv",
        "variant_results": root / "variant_results",
    }


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_group_column(split_strategy: str) -> str:
    return GROUP_COLUMN_BY_STRATEGY.get(split_strategy, "sample_id")


def resolve_token_ids(
    *,
    model_name: str,
    model_id: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
) -> tuple[list[int], list[int]]:
    if yes_token_ids and no_token_ids:
        return yes_token_ids, no_token_ids
    resolved_model_id = model_id or KNOWN_MODEL_IDS.get(model_name, "")
    if not resolved_model_id:
        raise ValueError("Provide --model-id or both --yes-token-id and --no-token-id.")
    processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    token_map = resolve_yes_no_token_ids(tokenizer)
    return token_map["yes"], token_map["no"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-id", default="")
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
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-random-state", type=int, default=13)
    parser.add_argument("--split-seeds", default="13,17,19,23,29")
    parser.add_argument("--yes-token-id", action="append", type=int, default=[])
    parser.add_argument("--no-token-id", action="append", type=int, default=[])
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
    yes_token_ids, no_token_ids = resolve_token_ids(
        model_name=args.model_name,
        model_id=args.model_id,
        yes_token_ids=args.yes_token_id,
        no_token_ids=args.no_token_id,
    )
    output_baseline_frame = build_output_baseline_frame(
        cache_entries,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
    )
    feature_variants = build_feature_variant_frames(features)
    split_seeds = parse_int_list(args.split_seeds)

    variants: dict[str, tuple[pd.DataFrame, list[str]]] = {
        "full": (features, feature_columns(features)),
        "drift_only": (features, drift_only_columns(features)),
        "no_manifold": (no_manifold_frame, feature_columns(no_manifold_frame)),
        "linear_probe": (linear_probe_frame, feature_columns(linear_probe_frame)),
        "output_p_yes": (output_baseline_frame, ["p_yes"]),
        "output_logit_margin": (output_baseline_frame, ["yes_logit_margin"]),
        "output_chosen_answer_confidence": (output_baseline_frame, ["chosen_answer_confidence"]),
    }
    for variant_name, variant_frame in feature_variants.items():
        variants[variant_name] = (variant_frame, feature_columns(variant_frame))

    baselines: dict[str, object] = {
        "bank_scope": args.bank_scope,
        "presence_answer_summary": build_raw_model_yes_no_baseline(cache_entries),
    }
    ablation_rows: list[dict[str, object]] = []
    split_sensitivity_rows: list[dict[str, object]] = []

    output_paths = build_output_paths(output_root=args.output_root, experiment_name=args.experiment_name)
    output_paths["baselines"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["variant_results"].mkdir(parents=True, exist_ok=True)

    group_column = resolve_group_column(args.split_strategy)
    for variant_name, (variant_frame, columns) in variants.items():
        metrics, results = evaluate_feature_frame(
            variant_frame,
            columns=columns,
            split_strategy=args.split_strategy,
            test_size=args.test_size,
            random_state=args.random_state,
            num_folds=args.num_folds,
        )
        results_path = output_paths["variant_results"] / f"{variant_name}.csv"
        results.to_csv(results_path, index=False)
        confidence_intervals = compute_bootstrap_confidence_intervals(
            results,
            group_column=group_column,
            n_resamples=args.bootstrap_resamples,
            random_state=args.bootstrap_random_state,
        )
        baselines[variant_name] = {
            **metrics,
            "confidence_intervals": confidence_intervals,
            "result_path": str(results_path),
        }
        ablation_rows.append({"variant": variant_name, **metrics})

        sensitivity = evaluate_feature_frame_across_random_states(
            variant_frame,
            columns=columns,
            split_strategy=args.split_strategy,
            test_size=args.test_size,
            random_states=split_seeds,
            num_folds=args.num_folds,
        )
        for row in sensitivity.to_dict(orient="records"):
            split_sensitivity_rows.append({"variant": variant_name, **row})

    ablations = pd.DataFrame(ablation_rows)
    split_sensitivity = pd.DataFrame(split_sensitivity_rows)
    output_paths["baselines"].write_text(json.dumps(baselines, indent=2, sort_keys=True), encoding="utf-8")
    ablations.to_csv(output_paths["ablations"], index=False)
    split_sensitivity.to_csv(output_paths["split_sensitivity"], index=False)
    print(output_paths["baselines"])
    print(output_paths["ablations"])
    print(output_paths["split_sensitivity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
