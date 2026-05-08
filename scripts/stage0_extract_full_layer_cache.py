#!/usr/bin/env python3
"""Extract Stage 0 full-layer prefill cache shards."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import torch
import yaml

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.config import ModelConfig, load_yaml_config
from mind.data import HallucinationRecord
from mind.extractors.prefill import (
    estimate_prefill_cache_tensor_bytes,
    extract_prefill_entries,
    save_prefill_cache_shard,
)
from mind.models.factory import create_model_wrapper
from mind.models.types import resolve_torch_dtype
from mind.trajectory.dataset import validate_extraction_ready_row


LAYER_CONFIG_PATHS = (
    ("num_hidden_layers",),
    ("text_config", "num_hidden_layers"),
    ("llm_config", "num_hidden_layers"),
    ("language_config", "num_hidden_layers"),
)
REQUIRED_RECORD_FIELDS = tuple(field.name for field in fields(HallucinationRecord))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage0/cache"))
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--token-index", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, paths, and selected layers without loading the model.",
    )
    return parser


def load_normalized_records(path: Path) -> list[HallucinationRecord]:
    if not path.exists():
        raise FileNotFoundError(f"records file does not exist: {path}")
    records: list[HallucinationRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing required record fields: {', '.join(missing)}"
            )
        validate_extraction_ready_row(
            row,
            path=path,
            record_number=line_number,
            required_fields=REQUIRED_RECORD_FIELDS,
        )
        payload = {field: row[field] for field in REQUIRED_RECORD_FIELDS}
        try:
            records.append(HallucinationRecord(**payload))
        except TypeError as error:
            raise ValueError(f"{path}:{line_number}: invalid HallucinationRecord: {error}") from error
    return records


def resolve_image_paths(
    records: Sequence[HallucinationRecord],
    *,
    image_root: Path | None,
) -> list[HallucinationRecord]:
    if image_root is None:
        return list(records)
    resolved: list[HallucinationRecord] = []
    for record in records:
        image_path = Path(record.image_path)
        if image_path.is_absolute():
            resolved.append(record)
        else:
            resolved.append(replace(record, image_path=str(image_root / image_path)))
    return resolved


def iter_record_shards(
    records: Sequence[HallucinationRecord],
    *,
    shard_size: int,
) -> Iterable[list[HallucinationRecord]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    for start in range(0, len(records), shard_size):
        yield list(records[start : start + shard_size])


def build_cache_output_path(
    *,
    output_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    shard_index: int,
) -> Path:
    return output_root / model_name / dataset_name / split / f"shard-{shard_index:05d}.pt"


def resolve_total_layers(model_or_config: object) -> int:
    config = getattr(model_or_config, "config", model_or_config)
    for path in LAYER_CONFIG_PATHS:
        candidate = _lookup_path(config, path)
        if candidate is not None:
            return _positive_int(candidate, ".".join(path))
    raise ValueError(
        "Could not resolve total layers from model config. Checked "
        "num_hidden_layers, text_config.num_hidden_layers, "
        "llm_config.num_hidden_layers, and language_config.num_hidden_layers."
    )


def resolve_total_layers_from_config_only(model_config_path: Path) -> int:
    payload = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"model config must be a YAML mapping: {model_config_path}")
    for path in LAYER_CONFIG_PATHS:
        candidate = _lookup_path(payload, path)
        if candidate is not None:
            return _positive_int(candidate, ".".join(path))

    config = load_yaml_config(model_config_path, ModelConfig)
    model_id = str(config.model_id)
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=config.trust_remote_code,
            local_files_only=True,
        )
        return resolve_total_layers(hf_config)
    except Exception as error:
        raise ValueError(
            "Dry-run could not resolve total layers exactly from YAML or local "
            f"model config for {model_id}. Add num_hidden_layers to the YAML, or "
            f"ensure the model config is cached locally. Last error: {error}"
        ) from error


def _lookup_path(obj: object, path: Sequence[str]) -> object | None:
    current = obj
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def _positive_int(value: object, label: str) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer, got {value!r}") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {result}")
    return result


def selected_layers_for_total(total_layers: int) -> list[int]:
    return list(range(_positive_int(total_layers, "total_layers")))


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hidden_dim_from_entries(entries: Sequence[dict[str, object]]) -> int | None:
    for entry in entries:
        layer_vectors = entry.get("layer_vectors")
        shape = getattr(layer_vectors, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1])
    return None


def build_sidecar_metadata(
    *,
    model_config: ModelConfig,
    dataset_name: str,
    source_dataset: str,
    subset: str,
    split: str,
    total_layers: int,
    selected_layers: Sequence[int],
    hidden_dim: int | None,
    token_index: int,
    max_new_tokens: int,
    dtype: torch.dtype,
    num_entries: int,
    records_path: Path,
    image_root: Path | None,
) -> dict[str, object]:
    return {
        "stage": "stage0",
        "cache_type": "full_layer_prefill",
        "model_name": model_config.name,
        "model_id": model_config.model_id,
        "model_family": model_config.family,
        "dataset_name": dataset_name,
        "source_dataset": source_dataset,
        "subset": subset,
        "split": split,
        "total_layers": int(total_layers),
        "selected_layers": [int(layer) for layer in selected_layers],
        "num_selected_layers": len(selected_layers),
        "hidden_dim": hidden_dim,
        "token_index": int(token_index),
        "max_new_tokens": int(max_new_tokens),
        "dtype": str(dtype).removeprefix("torch."),
        "num_entries": int(num_entries),
        "script": str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1])),
        "git_commit": get_git_commit(),
        "created_at_utc": utc_now_iso(),
        "records_path": str(records_path),
        "image_root": None if image_root is None else str(image_root),
    }


def merge_top_level_sidecar_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    sidecar_path = Path(str(path) + ".json")
    if not sidecar_path.exists():
        return
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError(f"sidecar metadata must be a JSON object: {sidecar_path}")
    sidecar.update(metadata)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_source_dataset(entries: Sequence[Mapping[str, object]], *, fallback: str) -> str:
    values: set[str] = set()
    for entry in entries:
        value = entry.get("source_dataset")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            values.add(text)
    if len(values) == 1:
        return next(iter(values))
    return fallback


def run_extraction(
    *,
    records_path: Path,
    model_config_path: Path,
    output_root: Path,
    dataset_name: str,
    subset: str,
    split: str,
    image_root: Path | None,
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int,
    token_index: int,
    limit: int,
    shard_size: int,
    batch_size: int,
) -> list[Path]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    records = load_normalized_records(records_path)
    if limit > 0:
        records = records[:limit]
    records = resolve_image_paths(records, image_root=image_root)

    model_config = load_yaml_config(model_config_path, ModelConfig)
    wrapper = create_model_wrapper(model_config)
    processor = wrapper.load_processor()
    model = wrapper.load_model(device=device)
    total_layers = resolve_total_layers(model)
    selected_layers = selected_layers_for_total(total_layers)

    output_paths: list[Path] = []
    with torch.inference_mode():
        for shard_index, shard_records in enumerate(iter_record_shards(records, shard_size=shard_size)):
            entries: list[dict[str, object]] = []
            for start in range(0, len(shard_records), batch_size):
                batch_records = shard_records[start : start + batch_size]
                batch_entries = extract_prefill_entries(
                    model=model,
                    processor=processor,
                    wrapper=wrapper,
                    records=batch_records,
                    selected_layers=selected_layers,
                    device=device,
                    token_index=token_index,
                    max_new_tokens=max_new_tokens,
                )
                for entry in batch_entries:
                    entry["model_name"] = model_config.name
                    entry["dataset_name"] = dataset_name
                    entry.setdefault("source_dataset", dataset_name)
                entries.extend(batch_entries)

            output_path = build_cache_output_path(
                output_root=output_root,
                model_name=model_config.name,
                dataset_name=dataset_name,
                split=split,
                shard_index=shard_index,
            )
            estimated_bytes = estimate_prefill_cache_tensor_bytes(
                entries,
                dtype=dtype,
                cast_all_floating_tensors=False,
            )
            metadata = build_sidecar_metadata(
                model_config=model_config,
                dataset_name=dataset_name,
                source_dataset=resolve_source_dataset(entries, fallback=dataset_name),
                subset=subset,
                split=split,
                total_layers=total_layers,
                selected_layers=selected_layers,
                hidden_dim=hidden_dim_from_entries(entries),
                token_index=token_index,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                num_entries=len(entries),
                records_path=records_path,
                image_root=image_root,
            )
            sidecar = save_prefill_cache_shard(
                entries,
                output_path,
                dtype=dtype,
                cast_all_floating_tensors=False,
                estimated_tensor_bytes=estimated_bytes,
                metadata=metadata,
            )
            merge_top_level_sidecar_metadata(output_path, metadata)
            print(
                f"shard={output_path} entries={len(entries)} "
                f"estimated_tensor_bytes={estimated_bytes} "
                f"actual_file_bytes={sidecar.get('actual_file_bytes', 'unknown')}"
            )
            output_paths.append(output_path)
    print(
        f"summary shards={len(output_paths)} records={len(records)} "
        f"model={model_config.name} dataset={dataset_name}/{split} "
        f"layers={len(selected_layers)}"
    )
    for output_path in output_paths:
        print(output_path)
    return output_paths


def run_dry_run(
    *,
    records_path: Path,
    model_config_path: Path,
    output_root: Path,
    dataset_name: str,
    subset: str,
    split: str,
    image_root: Path | None,
    limit: int,
) -> int:
    model_config = load_yaml_config(model_config_path, ModelConfig)
    total_layers = resolve_total_layers_from_config_only(model_config_path)
    selected_layers = selected_layers_for_total(total_layers)
    records = load_normalized_records(records_path)
    if limit > 0:
        records = records[:limit]
    resolved_records = resolve_image_paths(records, image_root=image_root)
    output_dir = output_root / model_config.name / dataset_name / split

    print("dry_run=true")
    print(f"model_name={model_config.name}")
    print(f"model_id={model_config.model_id}")
    print(f"dataset={dataset_name}/{subset} split={split}")
    print(f"records={len(resolved_records)} records_path={records_path}")
    print(f"output_dir={output_dir}")
    print(f"total_layers={total_layers}")
    print(f"selected_layers={selected_layers}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = args.split or args.subset
    try:
        if args.dry_run:
            return run_dry_run(
                records_path=args.records,
                model_config_path=args.model_config,
                output_root=args.output_root,
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=split,
                image_root=args.image_root,
                limit=args.limit,
            )
        run_extraction(
            records_path=args.records,
            model_config_path=args.model_config,
            output_root=args.output_root,
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=split,
            image_root=args.image_root,
            device=args.device,
            dtype=resolve_torch_dtype(args.dtype),
            max_new_tokens=args.max_new_tokens,
            token_index=args.token_index,
            limit=args.limit,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
        )
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
