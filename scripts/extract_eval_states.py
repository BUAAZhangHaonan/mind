#!/usr/bin/env python3
"""Extract pre-generation hidden-state cache shards for evaluation splits."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.config import ModelConfig, load_yaml_config
from mind.data import HallucinationRecord
from mind.extractors import (
    extract_prefill_entries,
    save_prefill_cache_shard,
    select_layer_range,
)
from mind.models import create_model_wrapper


def build_cache_output_path(
    *,
    output_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    shard_index: int,
) -> Path:
    return output_root / model_name / dataset_name / split / f"shard-{shard_index:05d}.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selected-layers", type=int, default=16)
    parser.add_argument("--layer-range", default="middle")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def load_normalized_records(path: Path) -> list[HallucinationRecord]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [HallucinationRecord(**row) for row in rows]


def resolve_image_paths(
    records: Sequence[HallucinationRecord],
    *,
    image_root: Path | None,
) -> list[HallucinationRecord]:
    if image_root is None:
        return list(records)
    resolved_records: list[HallucinationRecord] = []
    for record in records:
        image_path = Path(record.image_path)
        if image_path.is_absolute():
            resolved_records.append(record)
            continue
        resolved_records.append(replace(record, image_path=str(image_root / image_path)))
    return resolved_records


def iter_record_shards(
    records: Sequence[HallucinationRecord],
    *,
    shard_size: int,
) -> Iterable[list[HallucinationRecord]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    for start in range(0, len(records), shard_size):
        yield list(records[start : start + shard_size])


def resolve_total_layers(model: object) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "num_hidden_layers", None),
        getattr(getattr(config, "text_config", None), "num_hidden_layers", None),
        getattr(getattr(config, "llm_config", None), "num_hidden_layers", None),
        getattr(getattr(config, "language_config", None), "num_hidden_layers", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError("Could not resolve num_hidden_layers from model config.")


def run_extraction(
    *,
    records_path: Path,
    model_config_path: Path,
    output_root: Path,
    dataset_name: str,
    split: str,
    image_root: Path | None,
    device: str,
    shard_size: int,
    batch_size: int,
    selected_layer_count: int,
    layer_range: str,
    max_new_tokens: int,
    limit: int = 0,
) -> list[Path]:
    model_config = load_yaml_config(model_config_path, ModelConfig)
    wrapper = create_model_wrapper(model_config)
    processor = wrapper.load_processor()
    model = wrapper.load_model(device=device)
    total_layers = resolve_total_layers(model)
    selected_layers = select_layer_range(
        total_layers=total_layers,
        count=selected_layer_count,
        range_name=layer_range,
    )
    records = load_normalized_records(records_path)
    records = resolve_image_paths(records, image_root=image_root)
    if limit > 0:
        records = records[:limit]

    output_paths: list[Path] = []
    with torch.inference_mode():
        for shard_index, shard_records in enumerate(
            iter_record_shards(records, shard_size=shard_size)
        ):
            entries = []
            for start in range(0, len(shard_records), batch_size):
                entries.extend(
                    extract_prefill_entries(
                        model=model,
                        processor=processor,
                        wrapper=wrapper,
                        records=shard_records[start : start + batch_size],
                        selected_layers=selected_layers,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
            output_path = build_cache_output_path(
                output_root=output_root,
                model_name=model_config.name,
                dataset_name=dataset_name,
                split=split,
                shard_index=shard_index,
            )
            save_prefill_cache_shard(entries, output_path)
            output_paths.append(output_path)
    return output_paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = run_extraction(
        records_path=args.records,
        model_config_path=args.model_config,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        split=args.split,
        image_root=args.image_root,
        device=args.device,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        selected_layer_count=args.selected_layers,
        layer_range=args.layer_range,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
