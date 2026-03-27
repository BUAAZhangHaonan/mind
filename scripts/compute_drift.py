#!/usr/bin/env python3
"""Compute drift features from cached hidden-state shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from mind.drift import compute_drift_curve, standardize_drift_curve
from mind.wavelets import extract_wavelet_features


def build_feature_output_path(
    *,
    output_root: Path,
    experiment_name: str,
    split: str,
) -> Path:
    return output_root / experiment_name / f"{split}.parquet"


def load_reference_bank(reference_root: Path, model_name: str) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, torch.Tensor]] = {}
    model_root = reference_root / model_name
    for layer_path in model_root.glob("*/*.pt"):
        object_name = layer_path.parent.name
        layer_index = int(layer_path.stem.split("-")[-1])
        bank.setdefault(object_name, {})[layer_index] = torch.load(layer_path, weights_only=False)
    return bank


def load_cache_entries(cache_path: Path) -> list[dict[str, object]]:
    if cache_path.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=False))
        return entries
    return list(torch.load(cache_path, weights_only=False))


def build_feature_frame(
    *,
    cache_entries: list[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in cache_entries:
        curve = compute_drift_curve(
            layer_vectors=entry["layer_vectors"],
            selected_layers=[int(layer) for layer in entry["selected_layers"]],
            object_name=str(entry["object_name"]),
            reference_bank=reference_bank,
        )
        features = extract_wavelet_features(standardize_drift_curve(curve))
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "label": entry["label"],
                "subset": entry["subset"],
                "object_name": entry["object_name"],
                **features,
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split", required=True)
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
    reference_bank = load_reference_bank(args.reference_root, args.model_name)
    frame = build_feature_frame(cache_entries=cache_entries, reference_bank=reference_bank)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
