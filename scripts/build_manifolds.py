#!/usr/bin/env python3
"""Build local manifold artifacts from cached reference states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.manifolds import (
    SHARED_BANK_KEY,
    SHUFFLED_OBJECT_MAP_FILENAME,
    build_reference_bank,
    build_shuffled_object_mapping,
    clean_reference_entries,
    compute_reference_bank_stats,
    compute_reference_bank_stats_from_bank,
)


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


def build_shuffled_mapping_output_path(*, output_root: Path, model_name: str) -> Path:
    return output_root / model_name / SHUFFLED_OBJECT_MAP_FILENAME


def load_saved_reference_bank(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, torch.Tensor]]:
    model_root = reference_root / model_name
    bank: dict[str, dict[int, torch.Tensor]] = {}
    for layer_path in sorted(model_root.glob("*/layer-*.pt")):
        object_name = layer_path.parent.name
        if bank_scope == "shared":
            if object_name != SHARED_BANK_KEY:
                continue
        elif object_name == SHARED_BANK_KEY:
            continue
        layer_index = int(layer_path.stem.split("-")[-1])
        bank.setdefault(object_name, {})[layer_index] = torch.load(layer_path, weights_only=True)
    if not bank:
        raise ValueError(f"No saved reference bank tensors found under {model_root}.")
    return bank


def _subsample_saved_bank(
    bank: dict[str, dict[int, torch.Tensor]],
    *,
    subsample_size: int | None,
) -> dict[str, dict[int, torch.Tensor]]:
    if subsample_size is None:
        return {
            object_name: {
                int(layer_index): tensor.detach().cpu().clone()
                for layer_index, tensor in layer_map.items()
            }
            for object_name, layer_map in bank.items()
        }
    if subsample_size <= 0:
        raise ValueError("subsample_size must be positive when provided.")
    subsampled: dict[str, dict[int, torch.Tensor]] = {}
    for object_name, layer_map in bank.items():
        if not layer_map:
            continue
        max_rows = min(int(tensor.shape[0]) for tensor in layer_map.values())
        take = min(int(subsample_size), max_rows)
        if take <= 0:
            continue
        subsampled[object_name] = {
            int(layer_index): tensor[:take].detach().cpu().clone()
            for layer_index, tensor in layer_map.items()
        }
    if not subsampled:
        raise ValueError("No reference bank tensors remain after subsampling.")
    return subsampled


def _derive_bank_from_saved_tensors(
    bank: dict[str, dict[int, torch.Tensor]],
    *,
    bank_scope: str,
    shuffle_seed: int,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, str] | None]:
    if bank_scope == "object":
        return bank, None
    if bank_scope == "shared":
        layer_bank: dict[int, list[torch.Tensor]] = {}
        for layer_map in bank.values():
            for layer_index, tensor in layer_map.items():
                layer_bank.setdefault(int(layer_index), []).append(tensor.detach().cpu())
        return {
            SHARED_BANK_KEY: {
                layer_index: torch.cat(tensors, dim=0)
                for layer_index, tensors in layer_bank.items()
            }
        }, None
    if bank_scope == "shuffled_object":
        mapping = build_shuffled_object_mapping(sorted(bank.keys()), shuffle_seed=shuffle_seed)
        derived = {
            destination: {
                int(layer_index): tensor.detach().cpu().clone()
                for layer_index, tensor in bank[source].items()
            }
            for destination, source in mapping.items()
        }
        return derived, mapping
    raise ValueError(f"Unsupported bank scope: {bank_scope}")


def _write_reference_bank_artifacts(
    *,
    bank: dict[str, dict[int, torch.Tensor]],
    stats_map: dict[str, dict[int, dict[str, float]]],
    output_root: Path,
    model_name: str,
    bank_scope: str,
    shuffled_object_mapping: dict[str, str] | None = None,
) -> list[Path]:
    written_paths: list[Path] = []
    for object_name, layer_map in bank.items():
        for layer_index, tensor in layer_map.items():
            path = build_output_path(
                output_root=output_root,
                model_name=model_name,
                object_name=object_name,
                layer_index=int(layer_index),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor.detach().cpu(), path)
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
                    "bank_scope": bank_scope,
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
    if shuffled_object_mapping is not None:
        mapping_path = build_shuffled_mapping_output_path(output_root=output_root, model_name=model_name)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(
            json.dumps(shuffled_object_mapping, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        written_paths.append(mapping_path)
    return written_paths


def save_reference_bank(
    *,
    entries: list[dict[str, object]],
    output_root: Path,
    model_name: str,
    k_neighbors: int = 32,
    bank_scope: str = "object",
    shuffle_seed: int = 13,
) -> list[Path]:
    cleaned_entries = clean_reference_entries(entries)
    shuffled_object_mapping: dict[str, str] | None = None
    if bank_scope == "shuffled_object":
        shuffled_object_mapping = build_shuffled_object_mapping(
            [str(entry["object_name"]) for entry in cleaned_entries],
            shuffle_seed=shuffle_seed,
        )
    stats_map = compute_reference_bank_stats(
        cleaned_entries,
        k_neighbors=k_neighbors,
        bank_scope=bank_scope,
        shuffle_seed=shuffle_seed,
        shuffled_object_mapping=shuffled_object_mapping,
    )
    bank = build_reference_bank(
        cleaned_entries,
        min_points=0,
        bank_scope=bank_scope,
        shuffle_seed=shuffle_seed,
        shuffled_object_mapping=shuffled_object_mapping,
    )
    return _write_reference_bank_artifacts(
        bank=bank,
        stats_map=stats_map,
        output_root=output_root,
        model_name=model_name,
        bank_scope=bank_scope,
        shuffled_object_mapping=shuffled_object_mapping,
    )


def save_reference_bank_from_saved_tensors(
    *,
    reference_root: Path,
    output_root: Path,
    model_name: str,
    k_neighbors: int = 32,
    bank_scope: str = "object",
    shuffle_seed: int = 13,
    subsample_size: int | None = None,
) -> list[Path]:
    source_bank_scope = "object" if bank_scope in {"shared", "shuffled_object"} else bank_scope
    source_bank = load_saved_reference_bank(reference_root, model_name, bank_scope=source_bank_scope)
    source_bank = _subsample_saved_bank(source_bank, subsample_size=subsample_size)
    derived_bank, shuffled_object_mapping = _derive_bank_from_saved_tensors(
        source_bank,
        bank_scope=bank_scope,
        shuffle_seed=shuffle_seed,
    )
    stats_map = compute_reference_bank_stats_from_bank(
        derived_bank,
        k_neighbors=k_neighbors,
    )
    return _write_reference_bank_artifacts(
        bank=derived_bank,
        stats_map=stats_map,
        output_root=output_root,
        model_name=model_name,
        bank_scope=bank_scope,
        shuffled_object_mapping=shuffled_object_mapping,
    )


def load_cache_entries(reference_cache: Path) -> list[dict[str, object]]:
    if reference_cache.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(reference_cache.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=True))
        return entries
    return list(torch.load(reference_cache, weights_only=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reference-cache", type=Path, default=None)
    parser.add_argument("--reference-bank-root", type=Path, default=None)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--bank-scope", choices=["object", "shared", "shuffled_object"], default="object")
    parser.add_argument("--shuffle-seed", type=int, default=13)
    parser.add_argument("--subsample-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reference_cache is not None and args.reference_bank_root is not None:
        raise ValueError("Provide either --reference-cache or --reference-bank-root, not both.")
    if args.reference_cache is None:
        if args.reference_bank_root is not None:
            for path in save_reference_bank_from_saved_tensors(
                reference_root=args.reference_bank_root,
                output_root=args.output_root,
                model_name=args.model_name,
                k_neighbors=args.k_neighbors,
                bank_scope=args.bank_scope,
                shuffle_seed=args.shuffle_seed,
                subsample_size=args.subsample_size,
            ):
                print(path)
            return 0
        raise ValueError("Provide either --reference-cache or --reference-bank-root.")

    entries = load_cache_entries(args.reference_cache)
    for path in save_reference_bank(
        entries=entries,
        output_root=args.output_root,
        model_name=args.model_name,
        k_neighbors=args.k_neighbors,
        bank_scope=args.bank_scope,
        shuffle_seed=args.shuffle_seed,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
