#!/usr/bin/env python3
"""Build local manifold artifacts from cached reference states."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from mind.manifolds import build_reference_bank, clean_reference_entries, compute_reference_bank_stats


def build_output_path(
    *,
    output_root: Path,
    model_name: str,
    object_name: str,
    layer_index: int,
) -> Path:
    return output_root / model_name / object_name / f"layer-{layer_index:02d}.pt"


def build_stats_output_path(
    *,
    output_root: Path,
    model_name: str,
    object_name: str,
) -> Path:
    return output_root / model_name / object_name / "stats.pt"


def build_counts_output_path(*, output_root: Path, model_name: str) -> Path:
    return output_root / model_name / "reference_counts.csv"


def save_reference_bank(
    *,
    entries: list[dict[str, object]],
    output_root: Path,
    model_name: str,
    k_neighbors: int = 32,
) -> list[Path]:
    cleaned_entries = clean_reference_entries(entries)
    stats_map = compute_reference_bank_stats(cleaned_entries, k_neighbors=k_neighbors)
    bank = build_reference_bank(cleaned_entries, min_points=k_neighbors)
    written_paths: list[Path] = []
    for object_name, layer_map in bank.items():
        for layer_index, tensor in layer_map.items():
            path = build_output_path(
                output_root=output_root,
                model_name=model_name,
                object_name=object_name,
                layer_index=layer_index,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)
            written_paths.append(path)
    count_rows: list[dict[str, object]] = []
    for object_name, layer_stats in stats_map.items():
        stats_path = build_stats_output_path(
            output_root=output_root,
            model_name=model_name,
            object_name=object_name,
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(layer_stats, stats_path)
        written_paths.append(stats_path)
        for layer_index, stats in sorted(layer_stats.items()):
            count_rows.append(
                {
                    "object_name": object_name,
                    "layer_index": int(layer_index),
                    "count": int(stats["count"]),
                    "supports_manifold": bool(stats["supports_manifold"]),
                }
            )
    counts_path = build_counts_output_path(output_root=output_root, model_name=model_name)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_frame = pd.DataFrame(count_rows)
    if not counts_frame.empty:
        counts_frame = counts_frame.sort_values(["object_name", "layer_index"])
    counts_frame.to_csv(counts_path, index=False)
    written_paths.append(counts_path)
    return written_paths


def load_cache_entries(reference_cache: Path) -> list[dict[str, object]]:
    if reference_cache.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(reference_cache.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=False))
        return entries
    return list(torch.load(reference_cache, weights_only=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reference-cache", type=Path, default=None)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--k-neighbors", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reference_cache is None:
        print(
            build_output_path(
                output_root=args.output_root,
                model_name=args.model_name,
                object_name=args.object_name,
                layer_index=args.layer_index,
            )
        )
        return 0

    entries = load_cache_entries(args.reference_cache)
    for path in save_reference_bank(
        entries=entries,
        output_root=args.output_root,
        model_name=args.model_name,
        k_neighbors=args.k_neighbors,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
