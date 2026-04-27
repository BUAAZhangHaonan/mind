#!/usr/bin/env python3
"""Verify normalized record IDs are covered by cached hidden-state shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import torch


CHUNKED_CACHE_SHARD_FORMAT = "chunked_cache_shard_v1"
CHUNKED_CACHE_PART_FILE_PATTERN = re.compile(r"\.part-\d{5}\.pt$")


def load_record_sample_ids(records_path: Path) -> list[str]:
    sample_ids: list[str] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "sample_id" not in row:
            raise ValueError(f"{records_path}:{line_number} is missing sample_id")
        sample_ids.append(str(row["sample_id"]))
    return sample_ids


def resolve_cache_path(*, cache_root: Path, model: str, dataset_name: str, split: str) -> Path:
    split_candidate = cache_root / model / dataset_name / split
    if split_candidate.exists():
        return split_candidate
    if cache_root.is_file():
        return cache_root
    if cache_root.name == split or any(cache_root.glob("*.pt")):
        return cache_root
    return split_candidate


def _load_cache_shard_entries(shard_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(shard_path, weights_only=True, map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == CHUNKED_CACHE_SHARD_FORMAT:
        part_names = payload.get("parts")
        if not isinstance(part_names, list):
            raise ValueError(f"Chunked cache shard manifest at {shard_path} is missing a parts list.")
        entries: list[dict[str, Any]] = []
        for part_name in part_names:
            part_payload = torch.load(shard_path.parent / str(part_name), weights_only=True, map_location="cpu")
            entries.extend(dict(entry) for entry in part_payload)
        return entries
    return [dict(entry) for entry in payload]


def load_cache_entries(cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.is_dir():
        entries: list[dict[str, Any]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            if CHUNKED_CACHE_PART_FILE_PATTERN.search(shard_path.name):
                continue
            entries.extend(_load_cache_shard_entries(shard_path))
        return entries
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")
    return _load_cache_shard_entries(cache_path)


def _entry_layers(entry: dict[str, Any]) -> tuple[int, ...] | None:
    if "selected_layers" not in entry:
        return None
    return tuple(int(layer_index) for layer_index in entry["selected_layers"])


def _entry_vector_shape(entry: dict[str, Any]) -> tuple[int, int | None] | None:
    if "layer_vectors" not in entry:
        return None
    tensor = torch.as_tensor(entry["layer_vectors"])
    if tensor.ndim == 0:
        return (0, None)
    if tensor.ndim == 1:
        return (1, int(tensor.shape[0]))
    return (int(tensor.shape[0]), int(tensor.shape[-1]))


def summarize_cache(cache_entries: list[dict[str, Any]]) -> dict[str, Any]:
    selected_layers = sorted(
        {layers for entry in cache_entries if (layers := _entry_layers(entry)) is not None}
    )
    vector_shapes = [
        shape
        for entry in cache_entries
        if (shape := _entry_vector_shape(entry)) is not None
    ]
    vector_dims = sorted({shape[1] for shape in vector_shapes if shape[1] is not None})
    selected_layer_counts = {len(layers) for layers in selected_layers}
    vector_layer_counts = {shape[0] for shape in vector_shapes}
    layers_match_vectors = all(
        (layers := _entry_layers(entry)) is None
        or (shape := _entry_vector_shape(entry)) is None
        or len(layers) == shape[0]
        for entry in cache_entries
    )
    return {
        "selected_layers": [list(layers) for layers in selected_layers],
        "selected_layers_consistent": len(selected_layers) <= 1,
        "vector_dims": vector_dims,
        "layer_dim_consistent": (
            len(selected_layer_counts) <= 1
            and len(vector_layer_counts) <= 1
            and len(vector_dims) <= 1
            and layers_match_vectors
        ),
    }


def print_report(
    *,
    total_records: int,
    cached_entries: int,
    missing_ids: list[str],
    duplicate_count: int,
    cache_summary: dict[str, Any],
) -> None:
    print(f"total_records: {total_records}")
    print(f"cached_entries: {cached_entries}")
    print(f"missing_count: {len(missing_ids)}")
    print(f"duplicate_count: {duplicate_count}")
    print(f"selected_layers: {json.dumps(cache_summary['selected_layers'])}")
    print(f"selected_layers_consistent: {str(cache_summary['selected_layers_consistent']).lower()}")
    print(f"vector_dims: {json.dumps(cache_summary['vector_dims'])}")
    print(f"layer_dim_consistent: {str(cache_summary['layer_dim_consistent']).lower()}")
    print("missing_ids:")
    for sample_id in missing_ids:
        print(f"  {sample_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record_ids = load_record_sample_ids(args.records)
    cache_path = resolve_cache_path(
        cache_root=args.cache_root,
        model=args.model,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    cache_entries = load_cache_entries(cache_path)
    cached_ids = [str(entry["sample_id"]) for entry in cache_entries if "sample_id" in entry]
    cached_counts = Counter(cached_ids)
    cached_id_set = set(cached_ids)
    missing_ids = sorted(sample_id for sample_id in record_ids if sample_id not in cached_id_set)
    duplicate_count = sum(count - 1 for count in cached_counts.values() if count > 1)

    print_report(
        total_records=len(record_ids),
        cached_entries=len(cache_entries),
        missing_ids=missing_ids,
        duplicate_count=duplicate_count,
        cache_summary=summarize_cache(cache_entries),
    )
    if (missing_ids or duplicate_count) and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
