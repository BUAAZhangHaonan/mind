#!/usr/bin/env python3
"""Compute drift features from cached hidden-state shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from mind.drift import build_drift_features, calibrate_drift_curve, compute_drift_curve
from mind.evaluation import compute_object_hallucination_label
from mind.manifolds import resolve_reference_scope_key


def build_feature_output_path(
    *,
    output_root: Path,
    experiment_name: str,
    split: str,
) -> Path:
    return output_root / experiment_name / f"{split}.parquet"


def load_reference_bank(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, torch.Tensor]] = {}
    model_root = reference_root / model_name
    for layer_path in model_root.glob("*/layer-*.pt"):
        object_name = layer_path.parent.name
        if bank_scope == "object" and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        layer_index = int(layer_path.stem.split("-")[-1])
        bank.setdefault(object_name, {})[layer_index] = torch.load(layer_path, weights_only=False)
    return bank


def load_reference_stats(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, dict[str, float]]]:
    stats_map: dict[str, dict[int, dict[str, float]]] = {}
    model_root = reference_root / model_name
    for stats_path in model_root.glob("*/stats.pt"):
        object_name = stats_path.parent.name
        if bank_scope == "object" and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        payload = torch.load(stats_path, weights_only=False)
        stats_map[object_name] = {
            int(layer_index): {str(key): float(value) for key, value in layer_stats.items()}
            for layer_index, layer_stats in payload.items()
        }
    return stats_map


def load_cache_entries(cache_path: Path) -> list[dict[str, object]]:
    if cache_path.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=False))
        return entries
    return list(torch.load(cache_path, weights_only=False))


def _format_missing_reference_coverage(missing_entries: list[dict[str, object]]) -> str:
    preview = ", ".join(
        f"{entry['sample_id']}[{entry['reason']}]"
        for entry in missing_entries[:5]
    )
    if len(missing_entries) > 5:
        preview += f", ... (+{len(missing_entries) - 5} more)"
    return f"Missing reference coverage for {len(missing_entries)} cache entries: {preview}"


def build_feature_frame(
    *,
    cache_entries: list[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_stats: dict[str, dict[int, dict[str, float]]],
    bank_scope: str = "object",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_entries: list[dict[str, object]] = []
    for entry in cache_entries:
        selected_layers = [int(layer) for layer in entry["selected_layers"]]
        object_name = str(entry["object_name"])
        bank_key = resolve_reference_scope_key(object_name, bank_scope)
        if bank_key not in reference_bank:
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": f"missing bank:{bank_key}"}
            )
            continue
        if bank_key not in reference_stats:
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": f"missing stats:{bank_key}"}
            )
            continue
        missing_bank_layers = [
            layer_index for layer_index in selected_layers if layer_index not in reference_bank[bank_key]
        ]
        missing_stats_layers = [
            layer_index for layer_index in selected_layers if layer_index not in reference_stats[bank_key]
        ]
        if missing_bank_layers or missing_stats_layers:
            reason_parts = []
            if missing_bank_layers:
                reason_parts.append(f"missing bank layers:{missing_bank_layers}")
            if missing_stats_layers:
                reason_parts.append(f"missing stats layers:{missing_stats_layers}")
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": "; ".join(reason_parts)}
            )
            continue
        raw_curve = compute_drift_curve(
            layer_vectors=entry["layer_vectors"],
            selected_layers=selected_layers,
            object_name=object_name,
            reference_bank=reference_bank,
            bank_scope=bank_scope,
        )
        calibrated_curve = calibrate_drift_curve(
            raw_curve,
            selected_layers=selected_layers,
            layer_stats=reference_stats[bank_key],
        )
        features = build_drift_features(
            raw_curve=raw_curve,
            calibrated_curve=calibrated_curve,
        )
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "image_id": int(entry.get("image_id", -1)),
                "ground_truth_label": int(entry["label"]),
                "answer_label": -1 if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
                "label": compute_object_hallucination_label(
                    ground_truth_label=int(entry["label"]),
                    answer_label=(
                        None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
                    ),
                ),
                "subset": entry["subset"],
                "object_name": object_name,
                **features,
            }
        )
    if missing_entries:
        raise ValueError(_format_missing_reference_coverage(missing_entries))
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--bank-scope", choices=["object", "shared"], default="object")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = build_feature_output_path(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        split=args.split,
    )
    if args.cache_path is None or args.reference_root is None or not args.model_name:
        print(output_path)
        return 0

    cache_entries = load_cache_entries(args.cache_path)
    reference_bank = load_reference_bank(args.reference_root, args.model_name, bank_scope=args.bank_scope)
    reference_stats = load_reference_stats(args.reference_root, args.model_name, bank_scope=args.bank_scope)
    frame = build_feature_frame(
        cache_entries=cache_entries,
        reference_bank=reference_bank,
        reference_stats=reference_stats,
        bank_scope=args.bank_scope,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
