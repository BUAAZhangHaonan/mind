#!/usr/bin/env python3
"""Compute MIND baselines and ablation summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoProcessor

from mind.evaluation.baselines import (
    DEFAULT_FULL_VARIANT,
    FEATURE_VARIANT_NAMES,
    GROUP_COLUMN_BY_STRATEGY,
    apply_label_overrides_to_entries,
    apply_label_overrides_to_frame,
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
    resolve_feature_variant_frame,
    resolve_yes_no_token_ids,
)
from mind.evaluation.metrics import write_results_table


KNOWN_MODEL_IDS = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B-HF",
    "llava-onevision-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "molmo-7b-d-0924": "allenai/Molmo-7B-D-0924",
}

KNOWN_TOKEN_IDS = {
    "qwen3-vl-8b": {"yes": [9693, 9834, 9454], "no": [2152, 902, 2753]},
    "internvl3.5-8b": {"yes": [9693, 9834, 9454], "no": [2152, 902, 2753]},
    "llava-onevision-7b": {"yes": [9693, 9834, 9454], "no": [2152, 902, 2753]},
    "molmo-7b-d-0924": {"yes": [9693, 9834, 9454], "no": [2152, 902, 2753]},
}

VARIANT_ORDER = (
    "full",
    "drift_only",
    "no_manifold",
    "linear_probe",
    "output_p_yes",
    "output_logit_margin",
    "output_chosen_answer_confidence",
    *FEATURE_VARIANT_NAMES,
)

CACHE_BACKED_VARIANTS = {
    "no_manifold",
    "linear_probe",
    "output_p_yes",
    "output_logit_margin",
    "output_chosen_answer_confidence",
}


def resolve_required_cache_fields(selected_variants: list[str]) -> set[str]:
    keep_fields = {
        "sample_id",
        "image_id",
        "label",
        "parsed_answer",
        "subset",
        "object_name",
    }
    if "no_manifold" in selected_variants:
        keep_fields.update({"selected_layers", "layer_vectors"})
    if "linear_probe" in selected_variants:
        keep_fields.add("layer_vectors")
    if any(
        variant_name in selected_variants
        for variant_name in (
            "output_p_yes",
            "output_logit_margin",
            "output_chosen_answer_confidence",
        )
    ):
        keep_fields.add("first_token_logits")
    return keep_fields


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


def parse_variant_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(VARIANT_ORDER)
    variants: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in VARIANT_ORDER:
            raise ValueError(f"Unsupported baseline variant: {name}")
        if name in seen:
            continue
        seen.add(name)
        variants.append(name)
    if not variants:
        raise ValueError("Provide at least one baseline variant or use --variants all.")
    return variants


def resolve_group_column(split_strategy: str) -> str:
    return GROUP_COLUMN_BY_STRATEGY.get(split_strategy, "sample_id")


def load_existing_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_metric_rows(
    existing: pd.DataFrame,
    updates: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    if existing.empty:
        merged = updates.copy()
    elif updates.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, updates], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_columns, keep="last")
    if merged.empty or "variant" not in merged.columns:
        return merged
    variant_rank = {name: index for index, name in enumerate(VARIANT_ORDER)}
    merged["__variant_rank"] = merged["variant"].map(variant_rank).fillna(len(variant_rank)).astype(int)
    sort_columns = ["__variant_rank"]
    ascending = [True]
    for column in ["random_state"]:
        if column in merged.columns:
            sort_columns.append(column)
            ascending.append(True)
    merged = merged.sort_values(sort_columns, ascending=ascending).drop(columns="__variant_rank")
    return merged.reset_index(drop=True)


def resolve_token_ids(
    *,
    model_name: str,
    model_id: str,
    yes_token_ids: list[int],
    no_token_ids: list[int],
) -> tuple[list[int], list[int]]:
    if yes_token_ids and no_token_ids:
        return yes_token_ids, no_token_ids
    if model_name in KNOWN_TOKEN_IDS:
        token_map = KNOWN_TOKEN_IDS[model_name]
        return list(token_map["yes"]), list(token_map["no"])
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
    parser.add_argument("--bank-scope", choices=["object", "shared", "shuffled_object"], default="object")
    parser.add_argument("--label-overrides", type=Path, default=None)
    parser.add_argument(
        "--full-variant",
        choices=list(FEATURE_VARIANT_NAMES),
        default=DEFAULT_FULL_VARIANT,
    )
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
    parser.add_argument("--variants", default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_variants = parse_variant_list(args.variants)
    features = pd.read_parquet(args.features_path)
    cache_entries = None
    if args.label_overrides is not None:
        features = apply_label_overrides_to_frame(features, args.label_overrides)
    if any(variant_name in CACHE_BACKED_VARIANTS for variant_name in selected_variants):
        cache_entries = load_cache_entries(
            args.cache_path,
            keep_fields=resolve_required_cache_fields(selected_variants),
        )
        if args.label_overrides is not None:
            cache_entries = apply_label_overrides_to_entries(cache_entries, args.label_overrides)
    split_seeds = parse_int_list(args.split_seeds)

    variants: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    if any(
        variant_name in selected_variants
        for variant_name in ("full", "drift_only", *FEATURE_VARIANT_NAMES)
    ):
        feature_variants = build_feature_variant_frames(features)
        if "full" in selected_variants:
            full_frame = resolve_feature_variant_frame(features, args.full_variant)
            variants["full"] = (full_frame, feature_columns(full_frame))
        if "drift_only" in selected_variants:
            variants["drift_only"] = (features, drift_only_columns(features))
        for variant_name in FEATURE_VARIANT_NAMES:
            if variant_name in selected_variants:
                variant_frame = feature_variants[variant_name]
                variants[variant_name] = (variant_frame, feature_columns(variant_frame))

    if "no_manifold" in selected_variants:
        if cache_entries is None:
            raise ValueError("no_manifold evaluation requires cache entries.")
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
        variants["no_manifold"] = (no_manifold_frame, feature_columns(no_manifold_frame))

    if "linear_probe" in selected_variants:
        if cache_entries is None:
            raise ValueError("linear_probe evaluation requires cache entries.")
        linear_probe_frame = build_linear_probe_frame(cache_entries)
        variants["linear_probe"] = (linear_probe_frame, feature_columns(linear_probe_frame))

    if any(
        variant_name in selected_variants
        for variant_name in (
            "output_p_yes",
            "output_logit_margin",
            "output_chosen_answer_confidence",
        )
    ):
        if cache_entries is None:
            raise ValueError("Output-side baseline evaluation requires cache entries.")
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
        if "output_p_yes" in selected_variants:
            variants["output_p_yes"] = (output_baseline_frame, ["p_yes"])
        if "output_logit_margin" in selected_variants:
            variants["output_logit_margin"] = (output_baseline_frame, ["yes_logit_margin"])
        if "output_chosen_answer_confidence" in selected_variants:
            variants["output_chosen_answer_confidence"] = (
                output_baseline_frame,
                ["chosen_answer_confidence"],
            )

    output_paths = build_output_paths(output_root=args.output_root, experiment_name=args.experiment_name)
    output_paths["baselines"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["variant_results"].mkdir(parents=True, exist_ok=True)

    baselines = load_existing_json(output_paths["baselines"])
    baselines["bank_scope"] = args.bank_scope
    baselines["full_variant"] = args.full_variant
    if cache_entries is not None:
        baselines["presence_answer_summary"] = build_raw_model_yes_no_baseline(cache_entries)
    ablations = load_existing_frame(output_paths["ablations"])
    split_sensitivity = load_existing_frame(output_paths["split_sensitivity"])

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
        write_results_table(results, results_path)
        print(f"[compute_baselines] wrote {results_path}")
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
        ablations = merge_metric_rows(
            ablations,
            pd.DataFrame([{"variant": variant_name, **metrics}]),
            key_columns=["variant"],
        )

        output_paths["baselines"].write_text(json.dumps(baselines, indent=2, sort_keys=True), encoding="utf-8")
        ablations.to_csv(output_paths["ablations"], index=False)
        print(f"[compute_baselines] updated {output_paths['baselines']}")
        print(f"[compute_baselines] updated {output_paths['ablations']}")
        for random_state in split_seeds:
            seed_metrics, _ = evaluate_feature_frame(
                variant_frame,
                columns=columns,
                split_strategy=args.split_strategy,
                test_size=args.test_size,
                random_state=int(random_state),
                num_folds=args.num_folds,
            )
            split_sensitivity = merge_metric_rows(
                split_sensitivity,
                pd.DataFrame([{"variant": variant_name, "random_state": int(random_state), **seed_metrics}]),
                key_columns=["variant", "random_state"],
            )
            split_sensitivity.to_csv(output_paths["split_sensitivity"], index=False)
            print(
                f"[compute_baselines] updated {output_paths['split_sensitivity']} "
                f"for {variant_name} seed {int(random_state)}"
            )
    print(output_paths["baselines"])
    print(output_paths["ablations"])
    print(output_paths["split_sensitivity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
