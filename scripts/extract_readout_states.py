#!/usr/bin/env python3
"""Extract separate pre-generation readout shards for HALP and GLSim."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import torch

from mind.config import ModelConfig, load_yaml_config
from mind.data import HallucinationRecord
from mind.extractors import extract_prefill_readout_entries, save_prefill_cache_shard
from mind.models import create_model_wrapper
from mind.utils import output_root_lock


def build_cache_output_path(
    *,
    output_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    shard_index: int,
) -> Path:
    return output_root / model_name / dataset_name / split / f"shard-{shard_index:05d}.pt"


SHARD_FILE_PATTERN = re.compile(r"^shard-(\d{5})\.pt$")


def collect_completed_shard_indices(
    output_dir: Path,
    *,
    expected_shards: int | None = None,
) -> list[int]:
    if not output_dir.exists():
        return []
    indices: list[int] = []
    for shard_path in sorted(output_dir.glob("shard-*.pt")):
        match = SHARD_FILE_PATTERN.match(shard_path.name)
        if match is None:
            raise ValueError(f"Unexpected shard filename in {output_dir}: {shard_path.name}")
        indices.append(int(match.group(1)))
    expected_prefix = list(range(len(indices)))
    if indices != expected_prefix:
        missing_indices = sorted(set(expected_prefix) - set(indices))
        raise ValueError(
            f"Readout output directory {output_dir} is non-contiguous; it must contain a "
            f"contiguous shard prefix starting at 0. Missing indices {missing_indices!r}, "
            f"found {indices!r}."
        )
    if expected_shards is not None:
        extra_indices = [index for index in indices if index >= expected_shards]
        if extra_indices:
            raise ValueError(
                f"Readout output directory {output_dir} contains shard indices beyond the "
                f"expected range 0..{expected_shards - 1}: {extra_indices!r}."
            )
    return indices


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser


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
    max_new_tokens: int,
    limit: int = 0,
) -> list[Path]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_config = load_yaml_config(model_config_path, ModelConfig)
    records = load_normalized_records(records_path)
    records = resolve_image_paths(records, image_root=image_root)
    if limit > 0:
        records = records[:limit]

    output_dir = output_root / model_config.name / dataset_name / split
    total_shards = (len(records) + shard_size - 1) // shard_size if records else 0
    completed_indices = set(
        collect_completed_shard_indices(
            output_dir,
            expected_shards=total_shards,
        )
    )
    output_paths: list[Path] = []

    if len(completed_indices) == total_shards:
        return [
            build_cache_output_path(
                output_root=output_root,
                model_name=model_config.name,
                dataset_name=dataset_name,
                split=split,
                shard_index=shard_index,
            )
            for shard_index in range(total_shards)
        ]

    wrapper = create_model_wrapper(model_config)
    processor = wrapper.load_processor()
    model = wrapper.load_model(device=device)

    with output_root_lock(
        output_dir,
        command=f"extract_readout_states:{model_config.name}:{dataset_name}:{split}",
    ):
        with torch.inference_mode():
            for shard_index, shard_records in enumerate(
                iter_record_shards(records, shard_size=shard_size)
            ):
                output_path = build_cache_output_path(
                    output_root=output_root,
                    model_name=model_config.name,
                    dataset_name=dataset_name,
                    split=split,
                    shard_index=shard_index,
                )
                output_paths.append(output_path)
                if shard_index in completed_indices:
                    continue
                shard_entries: list[dict[str, object]] = []
                for start in range(0, len(shard_records), batch_size):
                    shard_entries.extend(
                        extract_prefill_readout_entries(
                            model=model,
                            processor=processor,
                            wrapper=wrapper,
                            records=shard_records[start : start + batch_size],
                            device=device,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                save_prefill_cache_shard(shard_entries, output_path)
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
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
