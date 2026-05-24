#!/usr/bin/env python3
"""Extract official HALP readout cache for the wavelet-course RePOPE split.

This script intentionally writes only the compact readout fields needed by the
official HALP probes. It keeps the wavelet-course primary population metadata
from the Stage 0 cache, so labels, parsed answers, grouped split, and stable
population keys stay identical to the paired wavelet experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.config import ModelConfig, load_yaml_config as load_typed_yaml_config
from mind.data import HallucinationRecord
from mind.extractors.readouts import compact_halp_readout_entry, extract_halp_readout_entries
from mind.models.factory import create_model_wrapper
from mind.wavelet_course.cache_loading import load_repope_qwen_cache_entries
from mind.wavelet_course.population import WaveletPopulation, build_wavelet_population, population_key
from mind.wavelet_course.utils import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_NAMES,
    ensure_finite_array,
    parse_yes_no_label,
    require_text,
    seed_everything,
    write_json_object,
)


DEFAULT_CONFIG = Path("configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/models/qwen3_vl_8b.yaml")
DEFAULT_HALP_CACHE_DIR = Path("outputs/wavelet_course_v2/halp_cache/qwen3-vl-8b/repope/primary")
HALP_READOUT_FIELDS = (
    "vision_features",
    "query_hidden_states",
    "vision_token_hidden_states",
    "query_token_index",
    "vision_token_span",
)
PRIMARY_METADATA_FIELDS = (
    "model_name",
    "dataset_name",
    "source_dataset",
    "subset",
    "split",
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "object_name",
    "label",
    "parsed_answer",
    "answer_text",
    "wavelet_split",
    "wavelet_label",
    "wavelet_population_key",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke-test limit. Default 0 extracts the full primary population.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, Stage 0 cache, and primary population without loading the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(load_yaml_config(Path(args.config)), args)
    seed_everything(int(config["seed"]))
    ensure_batch_size_one(int(config["batch_size"]))
    ensure_device_available(str(config["device"]), allow_cpu=bool(config["allow_cpu"]))
    output_dir = Path(str(config["output_dir"]))
    prepare_output_dir(output_dir, overwrite=bool(config["overwrite"]))

    population = build_population(config)
    validate_population(population)
    primary_entries = population.primary_entries
    primary_entries = partition_entries(
        primary_entries,
        num_parts=int(config["num_parts"]),
        part_index=int(config["part_index"]),
    )
    if int(config["limit"]) > 0:
        primary_entries = primary_entries[: int(config["limit"])]
    if not primary_entries:
        raise ValueError("HALP extraction population is empty")

    if bool(config["dry_run"]):
        manifest = build_manifest(
            config,
            population=population,
            primary_entries=primary_entries,
            shards=[],
            started_at=None,
            finished_at=None,
            total_seconds=0.0,
        )
        write_json_object(output_dir / "manifest.dry_run.json", manifest)
        print_summary(manifest, output_dir=output_dir)
        return 0

    model_config_path = Path(str(config["model_config"]))
    model_config = load_typed_yaml_config(model_config_path, ModelConfig)
    if model_config.name != str(config["model_name"]):
        raise ValueError(
            f"model config name {model_config.name!r} does not match "
            f"experiment model_name {config['model_name']!r}"
        )

    wrapper = create_model_wrapper(model_config)
    processor = wrapper.load_processor()
    model = wrapper.load_model(device=str(config["device"]))
    model.eval()

    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    shards: list[dict[str, object]] = []
    with torch.inference_mode():
        for shard_index, shard_rows in enumerate(iter_shards(primary_entries, shard_size=int(config["shard_size"]))):
            shard_entries = extract_shard_entries(
                shard_rows,
                model=model,
                processor=processor,
                wrapper=wrapper,
                device=str(config["device"]),
                max_new_tokens=int(config["max_new_tokens"]),
                expected_num_layers=int(config["expected_num_layers"]),
                expected_hidden_dim=int(config["expected_hidden_dim"]),
            )
            shard_path = output_dir / f"shard-{shard_index:05d}.pt"
            sidecar = save_halp_shard(
                shard_entries,
                shard_path=shard_path,
                shard_index=shard_index,
                config=config,
            )
            print(
                f"shard={shard_path} entries={len(shard_entries)} "
                f"file_bytes={shard_path.stat().st_size}"
            )
            shards.append(sidecar)

    finished = datetime.now(timezone.utc)
    total_seconds = time.perf_counter() - start_time
    manifest = build_manifest(
        config,
        population=population,
        primary_entries=primary_entries,
        shards=shards,
        started_at=started,
        finished_at=finished,
        total_seconds=total_seconds,
    )
    write_json_object(output_dir / "manifest.json", manifest)
    print_summary(manifest, output_dir=output_dir)
    return 0


def load_yaml_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return dict(payload)


def resolve_config(config: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("seed", 20260506)
    resolved.setdefault("stage0_root", "outputs/stage0")
    resolved.setdefault("output_root", "outputs/wavelet_course_v2")
    resolved.setdefault("model_name", "qwen3-vl-8b")
    resolved.setdefault("dataset_name", "repope")
    resolved.setdefault("subsets", ["popular", "random", "adversarial"])
    resolved.setdefault("expected_num_layers", 36)
    resolved.setdefault("expected_hidden_dim", 4096)
    resolved.setdefault("split_ratios", {"train": 0.60, "validation": 0.20, "test": 0.20})
    halp_cache_cfg = dict(resolved.get("halp_cache", {}) or {})
    domain_cfg = dict(resolved.get("domain_baselines", {}) or {})
    resolved["device"] = args.device or str(resolved.get("device", "cuda:0"))
    resolved["allow_cpu"] = bool(args.allow_cpu or resolved.get("allow_cpu", False))
    resolved["output_dir"] = str(
        args.output_dir
        or halp_cache_cfg.get("output_dir")
        or domain_cfg.get("halp_readout_cache_path")
        or DEFAULT_HALP_CACHE_DIR
    )
    if int(args.num_parts) > 1:
        resolved["output_dir"] = str(Path(str(resolved["output_dir"])) / f"part-{int(args.part_index):03d}")
    resolved["model_config"] = str(args.model_config or halp_cache_cfg.get("model_config") or DEFAULT_MODEL_CONFIG)
    resolved["shard_size"] = int(args.shard_size)
    resolved["batch_size"] = int(args.batch_size)
    resolved["max_new_tokens"] = int(args.max_new_tokens)
    resolved["num_parts"] = int(args.num_parts)
    resolved["part_index"] = int(args.part_index)
    resolved["limit"] = int(args.limit)
    resolved["overwrite"] = bool(args.overwrite)
    resolved["dry_run"] = bool(args.dry_run)
    return resolved


def partition_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    num_parts: int,
    part_index: int,
) -> list[Mapping[str, Any]]:
    if num_parts <= 0:
        raise ValueError("--num-parts must be positive")
    if part_index < 0 or part_index >= num_parts:
        raise ValueError(f"--part-index must be in [0, {num_parts - 1}], got {part_index}")
    return [entry for index, entry in enumerate(entries) if index % num_parts == part_index]


def ensure_batch_size_one(batch_size: int) -> None:
    if batch_size != 1:
        raise ValueError("official HALP Qwen3-VL readout extraction requires --batch-size 1")


def ensure_device_available(device: str, *, allow_cpu: bool) -> None:
    normalized = str(device).strip().lower()
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r}, but CUDA is not available")
        return
    if normalized == "cpu" and allow_cpu:
        return
    raise RuntimeError(f"device {device!r} is not allowed; pass --allow-cpu for CPU")


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("shard-*.pt")) + sorted(output_dir.glob("shard-*.pt.json"))
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"HALP cache output already contains shard files under {output_dir}; "
            f"pass --overwrite to replace them. Existing: {preview}"
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for path in (output_dir / "manifest.json", output_dir / "manifest.dry_run.json"):
            if path.exists():
                path.unlink()


def build_population(config: Mapping[str, object]) -> WaveletPopulation:
    stage0_root = Path(str(config["stage0_root"]))
    entries = load_repope_qwen_cache_entries(
        stage0_root=stage0_root,
        manifest_path=stage0_root / "manifests" / "cache_manifest.json",
        model_name=str(config["model_name"]),
        dataset_name=str(config["dataset_name"]),
        subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
        expected_num_layers=int(config["expected_num_layers"]),
        expected_hidden_dim=int(config["expected_hidden_dim"]),
    )
    return build_wavelet_population(
        entries,
        manifest_dir=stage0_root / "manifests",
        dataset_name=str(config["dataset_name"]),
        subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
        seed=int(config["seed"]),
        ratios=split_ratio_values(config),
    )


def split_ratio_values(config: Mapping[str, object]) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", {})
    if isinstance(ratios, Mapping):
        return (
            float(ratios.get("train", DEFAULT_SPLIT_RATIOS[0])),
            float(ratios.get("validation", DEFAULT_SPLIT_RATIOS[1])),
            float(ratios.get("test", DEFAULT_SPLIT_RATIOS[2])),
        )
    values = list(ratios)  # type: ignore[arg-type]
    if len(values) != 3:
        raise ValueError("split_ratios must contain train/validation/test")
    return (float(values[0]), float(values[1]), float(values[2]))


def validate_population(population: WaveletPopulation) -> None:
    labels = list(population.labels)
    if not labels:
        raise ValueError("primary population is empty")
    if sorted(set(labels)) != [0, 1]:
        raise ValueError("primary population must contain both classes")
    for split in SPLIT_NAMES:
        split_labels = [
            int(label)
            for entry, label in zip(population.primary_entries, labels, strict=True)
            if str(entry.get("wavelet_split")) == split
        ]
        if sorted(set(split_labels)) != [0, 1]:
            raise ValueError(f"{split} split must contain both classes")


def iter_shards(entries: Sequence[Mapping[str, Any]], *, shard_size: int) -> list[list[Mapping[str, Any]]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return [list(entries[start : start + shard_size]) for start in range(0, len(entries), shard_size)]


def extract_shard_entries(
    shard_rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    device: str,
    max_new_tokens: int,
    expected_num_layers: int,
    expected_hidden_dim: int,
) -> list[dict[str, object]]:
    shard_entries: list[dict[str, object]] = []
    for row_index, primary in enumerate(shard_rows, start=1):
        record = hallucination_record_from_primary(primary)
        extracted = extract_halp_readout_entries(
            model=model,
            processor=processor,
            wrapper=wrapper,
            records=[record],
            device=device,
            max_new_tokens=max_new_tokens,
        )
        if len(extracted) != 1:
            raise RuntimeError(f"expected one extracted HALP readout entry, got {len(extracted)}")
        compact = compact_halp_readout_entry(extracted[0])
        cache_entry = build_compact_cache_entry(compact, primary=primary)
        validate_halp_cache_entry(
            cache_entry,
            context=f"shard row {row_index} sample_id={primary.get('sample_id')!r}",
            expected_num_layers=expected_num_layers,
            expected_hidden_dim=expected_hidden_dim,
        )
        shard_entries.append(cache_entry)
    return shard_entries


def hallucination_record_from_primary(primary: Mapping[str, Any]) -> HallucinationRecord:
    context = f"primary sample_id={primary.get('sample_id')!r}"
    image_path = require_text(primary.get("image_path"), field="image_path", context=context)
    resolved_image_path = Path(image_path)
    if not resolved_image_path.is_absolute():
        resolved_image_path = REPO_ROOT / resolved_image_path
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"{context}: image_path does not exist: {resolved_image_path}")
    label = parse_yes_no_label(primary.get("label"))
    if label is None:
        raise ValueError(f"{context}: label is not binary: {primary.get('label')!r}")
    return HallucinationRecord(
        sample_id=require_text(primary.get("sample_id"), field="sample_id", context=context),
        image_id=coerce_image_id(primary.get("image_id"), context=context),
        image_path=str(resolved_image_path),
        question=require_text(primary.get("question"), field="question", context=context),
        label=int(label),
        object_name=require_text(primary.get("object_name"), field="object_name", context=context),
        split=require_text(primary.get("split"), field="split", context=context),
        subset=require_text(primary.get("subset"), field="subset", context=context),
        source_dataset=require_text(primary.get("source_dataset"), field="source_dataset", context=context),
    )


def coerce_image_id(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: image_id must not be boolean")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: image_id must be int-like, got {value!r}") from error


def build_compact_cache_entry(compact: Mapping[str, Any], *, primary: Mapping[str, Any]) -> dict[str, object]:
    cache_entry = {
        field: compact[field]
        for field in HALP_READOUT_FIELDS
        if field in compact
    }
    for field in ("readout_format", "total_layers"):
        if field in compact:
            cache_entry[field] = compact[field]
    for field in PRIMARY_METADATA_FIELDS:
        if field in primary:
            cache_entry[field] = primary[field]
    cache_entry["readout_format"] = "compact_halp_cache_v1"
    cache_entry["halp_cache_source"] = "wavelet_course_primary_population"
    cache_entry["halp_cache_key"] = population_key(cache_entry)
    if cache_entry["halp_cache_key"] != primary.get("wavelet_population_key"):
        raise ValueError(
            f"HALP cache key mismatch for sample {primary.get('sample_id')!r}: "
            f"{cache_entry['halp_cache_key']!r} != {primary.get('wavelet_population_key')!r}"
        )
    return cache_entry


def validate_halp_cache_entry(
    entry: Mapping[str, object],
    *,
    context: str,
    expected_num_layers: int,
    expected_hidden_dim: int,
) -> None:
    for field in (*PRIMARY_METADATA_FIELDS, *HALP_READOUT_FIELDS):
        if field not in entry or entry[field] is None:
            raise ValueError(f"{context}: missing HALP cache field {field!r}")
    label = parse_yes_no_label(entry["label"])
    parsed = parse_yes_no_label(entry["parsed_answer"])
    if label is None:
        raise ValueError(f"{context}: label is not binary: {entry['label']!r}")
    if parsed is None:
        raise ValueError(f"{context}: parsed_answer is not binary: {entry['parsed_answer']!r}")
    if int(entry["wavelet_label"]) not in {0, 1}:
        raise ValueError(f"{context}: wavelet_label must be binary")
    if str(entry["wavelet_split"]) not in SPLIT_NAMES:
        raise ValueError(f"{context}: invalid wavelet_split={entry['wavelet_split']!r}")
    if int(entry["total_layers"]) != int(expected_num_layers):
        raise ValueError(f"{context}: total_layers must be {expected_num_layers}, got {entry['total_layers']}")
    validate_layer_state_tensor(
        entry["query_hidden_states"],
        field="query_hidden_states",
        context=context,
        expected_num_layers=expected_num_layers,
        expected_hidden_dim=expected_hidden_dim,
    )
    validate_layer_state_tensor(
        entry["vision_token_hidden_states"],
        field="vision_token_hidden_states",
        context=context,
        expected_num_layers=expected_num_layers,
        expected_hidden_dim=expected_hidden_dim,
    )
    validate_vision_features(entry["vision_features"], context=context)
    query_token_index = int(entry["query_token_index"])
    if query_token_index < 0:
        raise ValueError(f"{context}: query_token_index must be non-negative")
    span = entry["vision_token_span"]
    if not isinstance(span, (tuple, list)) or len(span) != 2:
        raise ValueError(f"{context}: vision_token_span must contain start and stop")
    start, stop = int(span[0]), int(span[1])
    if start < 0 or stop < start:
        raise ValueError(f"{context}: invalid vision_token_span={span!r}")


def validate_layer_state_tensor(
    value: object,
    *,
    field: str,
    context: str,
    expected_num_layers: int,
    expected_hidden_dim: int,
) -> None:
    shape = tuple(getattr(value, "shape", ()))
    expected = (int(expected_num_layers), int(expected_hidden_dim))
    if shape != expected:
        raise ValueError(f"{context}: {field} shape must be {expected}, got {shape}")
    ensure_finite_array(value, field=field, context=context)


def validate_vision_features(value: object, *, context: str) -> None:
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) not in {1, 2} or not shape or min(shape) <= 0:
        raise ValueError(f"{context}: vision_features must be non-empty 1D or 2D, got {shape}")
    ensure_finite_array(value, field="vision_features", context=context)


def save_halp_shard(
    entries: Sequence[dict[str, object]],
    *,
    shard_path: Path,
    shard_index: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    if not entries:
        raise ValueError(f"{shard_path}: refusing to save empty shard")
    torch.save(list(entries), shard_path)
    sidecar = {
        "format": "wavelet_course_halp_cache_shard_v1",
        "shard_index": shard_index,
        "path": str(shard_path),
        "sidecar_path": str(shard_path.with_suffix(shard_path.suffix + ".json")),
        "model_name": str(config["model_name"]),
        "dataset_name": str(config["dataset_name"]),
        "subset_scope": list(config["subsets"]),  # type: ignore[index]
        "num_entries": len(entries),
        "required_fields": list(PRIMARY_METADATA_FIELDS + HALP_READOUT_FIELDS),
        "tensor_fields": tensor_field_metadata(entries[0]),
        "population_keys_sha256": row_order_hash([str(entry["halp_cache_key"]) for entry in entries]),
        "actual_file_bytes": shard_path.stat().st_size,
    }
    write_json_object(Path(str(sidecar["sidecar_path"])), sidecar)
    return sidecar


def tensor_field_metadata(entry: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in ("vision_features", "query_hidden_states", "vision_token_hidden_states"):
        tensor = torch.as_tensor(entry[field])
        rows.append(
            {
                "field": field,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
            }
        )
    return rows


def build_manifest(
    config: Mapping[str, object],
    *,
    population: WaveletPopulation,
    primary_entries: Sequence[Mapping[str, object]],
    shards: Sequence[Mapping[str, object]],
    started_at: datetime | None,
    finished_at: datetime | None,
    total_seconds: float,
) -> dict[str, object]:
    split_counts = Counter(str(entry["wavelet_split"]) for entry in primary_entries)
    labels = [int(entry["wavelet_label"]) for entry in primary_entries]
    keys = [str(entry["wavelet_population_key"]) for entry in primary_entries]
    return {
        "format": "wavelet_course_halp_cache_manifest_v1",
        "created_at": None if finished_at is None else finished_at.isoformat(),
        "started_at": None if started_at is None else started_at.isoformat(),
        "total_seconds": float(total_seconds),
        "config": {
            "experiment_name": config.get("experiment_name"),
            "seed": int(config["seed"]),
            "stage0_root": str(config["stage0_root"]),
            "model_name": str(config["model_name"]),
            "dataset_name": str(config["dataset_name"]),
            "subsets": list(config["subsets"]),  # type: ignore[index]
            "expected_num_layers": int(config["expected_num_layers"]),
            "expected_hidden_dim": int(config["expected_hidden_dim"]),
            "device": str(config["device"]),
            "model_config": str(config["model_config"]),
            "batch_size": int(config["batch_size"]),
            "max_new_tokens": int(config["max_new_tokens"]),
            "num_parts": int(config["num_parts"]),
            "part_index": int(config["part_index"]),
            "limit": int(config["limit"]),
            "dry_run": bool(config["dry_run"]),
        },
        "population": {
            "full_primary_population": len(population.primary_entries),
            "written_entries": len(primary_entries),
            "positive_count": int(sum(labels)),
            "negative_count": int(len(labels) - sum(labels)),
            "positive_rate": float(sum(labels) / len(labels)) if labels else None,
            "split_source": population.split_source,
            "split_counts": dict(split_counts),
            "row_order_hash": row_order_hash(keys),
        },
        "readout_fields": list(HALP_READOUT_FIELDS),
        "primary_metadata_fields": list(PRIMARY_METADATA_FIELDS),
        "shards": [dict(shard) for shard in shards],
    }


def row_order_hash(keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def print_summary(manifest: Mapping[str, object], *, output_dir: Path) -> None:
    population = dict(manifest["population"])  # type: ignore[index]
    print(f"cache_output_dir={output_dir}")
    print(f"full_primary_population={population['full_primary_population']}")
    print(f"written_entries={population['written_entries']}")
    print(f"hard_hallucinations={population['positive_count']}")
    print(f"positive_rate={population['positive_rate']}")
    print(f"num_shards={len(manifest['shards'])}")  # type: ignore[arg-type,index]
    manifest_name = "manifest.dry_run.json" if dict(manifest["config"]).get("dry_run") else "manifest.json"  # type: ignore[index]
    print(f"manifest_path={output_dir / manifest_name}")


if __name__ == "__main__":
    raise SystemExit(main())
