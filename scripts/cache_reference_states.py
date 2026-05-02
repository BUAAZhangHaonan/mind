#!/usr/bin/env python3
"""Cache grounded reference hidden-state shards for manifold construction."""

from __future__ import annotations

import argparse
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

from mind.cache import BudgetExceededError, DiskBudget
from mind.config import ModelConfig, load_yaml_config
from mind.data import HallucinationRecord
from mind.utils.dtypes import resolve_torch_dtype


DEFAULT_PROMPT_TEMPLATE = "Is there a {object_name} in the image? Answer yes or no."


def build_reference_cache_output_path(
    *,
    output_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    shard_index: int,
) -> Path:
    return output_root / model_name / dataset_name / split / f"shard-{shard_index:05d}.pt"


def load_reference_candidates(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return list(json.loads(path.read_text(encoding="utf-8")))


def build_reference_records(
    *,
    candidates: Sequence[dict[str, object]],
    image_root: Path,
    split: str,
    prompt_template: str,
    subset: str = "reference",
    source_dataset: str = "coco_reference",
) -> list[HallucinationRecord]:
    records: list[HallucinationRecord] = []
    sorted_candidates = sorted(candidates, key=lambda row: (int(row["image_id"]), str(row["file_name"])))
    for candidate in sorted_candidates:
        image_id = int(candidate["image_id"])
        image_path = str(image_root / str(candidate["file_name"]))
        object_names = sorted(str(name) for name in candidate.get("object_names", []))
        for object_name in object_names:
            records.append(
                HallucinationRecord(
                    sample_id=f"ref-{image_id}-{object_name}",
                    image_id=image_id,
                    image_path=image_path,
                    question=prompt_template.format(object_name=object_name),
                    label=1,
                    object_name=object_name,
                    split=split,
                    subset=subset,
                    source_dataset=source_dataset,
                )
            )
    return records


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


def run_reference_caching(
    *,
    references_path: Path,
    image_root: Path,
    model_config_path: Path,
    output_root: Path,
    dataset_name: str,
    split: str,
    device: str,
    shard_size: int,
    batch_size: int,
    selected_layer_count: int,
    layer_range: str,
    max_new_tokens: int,
    prompt_template: str,
    limit: int = 0,
    dtype: torch.dtype = torch.float16,
    disk_budget: str | None = None,
) -> list[Path]:
    model_config = load_yaml_config(model_config_path, ModelConfig)
    from mind.extractors import (
        cleanup_prefill_cache_shard,
        estimate_prefill_cache_tensor_bytes,
        extract_prefill_entries,
        save_prefill_cache_shard,
        select_layer_range,
    )
    from mind.models.factory import create_model_wrapper

    wrapper = create_model_wrapper(model_config)
    processor = wrapper.load_processor()
    model = wrapper.load_model(device=device)
    total_layers = resolve_total_layers(model)
    selected_layers = select_layer_range(
        total_layers=total_layers,
        count=selected_layer_count,
        range_name=layer_range,
    )
    records = build_reference_records(
        candidates=load_reference_candidates(references_path),
        image_root=image_root,
        split=split,
        prompt_template=prompt_template,
    )
    if limit > 0:
        records = records[:limit]

    budget = None if disk_budget is None else DiskBudget(output_root, disk_budget)
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
            output_path = build_reference_cache_output_path(
                output_root=output_root,
                model_name=model_config.name,
                dataset_name=dataset_name,
                split=split,
                shard_index=shard_index,
            )
            estimated_bytes = estimate_prefill_cache_tensor_bytes(entries, dtype=dtype)
            if budget is not None:
                budget.allocate(estimated_bytes, label=str(output_path))
            sidecar = save_prefill_cache_shard(
                entries,
                output_path,
                dtype=dtype,
                estimated_tensor_bytes=estimated_bytes,
                metadata={
                    "model_name": model_config.name,
                    "dataset_name": dataset_name,
                    "split": split,
                    "shard_index": shard_index,
                    "selected_layers": selected_layers,
                },
            )
            actual_bytes = int(sidecar["actual_file_bytes"])
            if budget is not None:
                try:
                    budget.record_actual(
                        estimated_bytes=estimated_bytes,
                        actual_bytes=actual_bytes,
                        label=str(output_path),
                    )
                except BudgetExceededError:
                    cleanup_prefill_cache_shard(output_path)
                    raise
            print(
                f"{output_path}: wrote {actual_bytes} bytes "
                f"(estimated tensor payload {estimated_bytes} bytes)",
                file=sys.stderr,
            )
            output_paths.append(output_path)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selected-layers", type=int, default=16)
    parser.add_argument("--layer-range", default="middle")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Floating tensor dtype to store in cache shards. Default: float16.",
    )
    parser.add_argument(
        "--disk-budget",
        default=None,
        help=(
            "Maximum total bytes allowed under --output-root, including existing files. "
            "Accepts values like 500MB, 20GiB, or raw bytes. Omit for no limit."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.references is None or args.model_config is None or args.image_root is None:
        if not args.model_name:
            raise ValueError("--model-name is required when only resolving output paths.")
        print(
            build_reference_cache_output_path(
                output_root=args.output_root,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                split=args.split,
                shard_index=args.shard_index,
            )
        )
        return 0

    output_paths = run_reference_caching(
        references_path=args.references,
        image_root=args.image_root,
        model_config_path=args.model_config,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        split=args.split,
        device=args.device,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        selected_layer_count=args.selected_layers,
        layer_range=args.layer_range,
        max_new_tokens=args.max_new_tokens,
        prompt_template=args.prompt_template,
        limit=args.limit,
        dtype=resolve_torch_dtype(args.dtype),
        disk_budget=args.disk_budget,
    )
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
