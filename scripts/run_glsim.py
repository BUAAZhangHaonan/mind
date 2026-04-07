#!/usr/bin/env python3
"""Run the GLSim queried-object adaptation on cached pre-generation readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mind.comparators import (
    build_glsim_score_frame,
    build_object_token_contexts,
    evaluate_glsim_nested,
    resolve_glsim_layer_indices,
)
from mind.comparators.glsim import summarize_glsim_results
from mind.config import ModelConfig, load_yaml_config
from mind.evaluation.baselines import (
    apply_label_overrides_to_entries,
    load_cache_entries,
    validate_object_heldout_reference_support,
)
from mind.evaluation.metrics import write_results_table
from mind.models import create_model_wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-path", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-strategy", choices=["row", "image_grouped", "object_heldout"], default="image_grouped")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--inner-fold-candidates", default="3,2")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--label-overrides", type=Path, default=None)
    parser.add_argument("--image-layers", default="")
    parser.add_argument("--text-layers", default="")
    parser.add_argument("--k-values", default="4,8,16,32")
    parser.add_argument("--w-values", default="0.4,0.5,0.6")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--bank-scope", choices=["object", "shared", "shuffled_object"], default="object")
    return parser


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_output_paths(output_root: Path, experiment_name: str) -> dict[str, Path]:
    root = output_root / experiment_name
    return {
        "metrics": root / "glsim.json",
        "results": root / "glsim_results.csv",
        "selection": root / "glsim_selection.csv",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readout_entries = load_cache_entries(args.readout_path)
    if args.label_overrides is not None:
        readout_entries = apply_label_overrides_to_entries(readout_entries, args.label_overrides)

    if not readout_entries:
        raise ValueError("Readout cache is empty.")
    total_layers = int(readout_entries[0]["full_hidden_states"].shape[0])
    default_layers = resolve_glsim_layer_indices(total_layers)
    image_layers = _parse_int_list(args.image_layers) or default_layers
    text_layers = _parse_int_list(args.text_layers) or default_layers

    model_config = load_yaml_config(args.model_config, ModelConfig)
    wrapper = create_model_wrapper(model_config)
    processor = wrapper.load_processor()
    model = wrapper.load_model(device=args.device)
    contexts = build_object_token_contexts(
        readout_entries,
        wrapper=wrapper,
        processor=processor,
    )
    score_frame = build_glsim_score_frame(
        readout_entries,
        contexts=contexts,
        output_embeddings=model.get_output_embeddings(),
        layer_indices=sorted(set(image_layers + text_layers)),
        k_values=_parse_int_list(args.k_values),
    )
    supported_object_names: list[str] | None = None
    if args.split_strategy == "object_heldout":
        if args.reference_root is None or not args.model_name:
            raise ValueError("--reference-root and --model-name are required for object_heldout GLSim.")
        supported_object_names = validate_object_heldout_reference_support(
            score_frame,
            reference_root=args.reference_root,
            model_name=args.model_name,
            bank_scope=args.bank_scope,
            num_folds=args.num_folds,
        )
        print(f"[run_glsim] object_heldout support objects={len(supported_object_names)}")
    metrics, results, selection = evaluate_glsim_nested(
        score_frame,
        image_layers=image_layers,
        text_layers=text_layers,
        k_values=_parse_int_list(args.k_values),
        w_values=_parse_float_list(args.w_values),
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
        inner_candidate_folds=tuple(_parse_int_list(args.inner_fold_candidates)),
        supported_object_names=supported_object_names,
    )
    summary = summarize_glsim_results(
        results,
        split_strategy=args.split_strategy,
        bootstrap_resamples=args.bootstrap_resamples,
        random_state=args.random_state,
    )
    payload = {
        **metrics,
        "confidence_intervals": summary["confidence_intervals"],
        "selected_config_counts": (
            results[["fold", "selected_config"]]
            .drop_duplicates()
            ["selected_config"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    output_paths = build_output_paths(args.output_root, args.experiment_name)
    output_paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["metrics"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_results_table(results, output_paths["results"], extra_columns=("selected_config",))
    selection.to_csv(output_paths["selection"], index=False)
    print(output_paths["metrics"])
    print(output_paths["results"])
    print(output_paths["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
