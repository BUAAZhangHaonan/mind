"""Fail-closed loading for Qwen RePOPE Stage 0 cache shards."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_SUBSETS,
    ensure_finite_array,
    read_json_object,
    require_equal,
    require_field,
    require_file,
    require_list,
    require_mapping,
    require_text,
    resolve_existing_path,
)

DEFAULT_STAGE0_ROOT = Path("outputs/stage0")
DEFAULT_MANIFEST_PATH = DEFAULT_STAGE0_ROOT / "manifests" / "cache_manifest.json"
REQUIRED_ENTRY_FIELDS = (
    "layer_vectors",
    "first_token_logits",
    "label",
    "parsed_answer",
    "sample_id",
    "image_id",
    "question",
    "object_name",
    "subset",
    "model_name",
    "dataset_name",
)


@dataclass(frozen=True)
class CacheShard:
    path: Path
    sidecar_path: Path
    manifest_row: Mapping[str, Any]
    sidecar: Mapping[str, Any]


def load_cache_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    return read_json_object(path)


def iter_repope_qwen_cache_shards(
    *,
    stage0_root: Path | str = DEFAULT_STAGE0_ROOT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
) -> Iterator[CacheShard]:
    manifest = load_cache_manifest(manifest_path)
    yield from iter_cache_shards(
        stage0_root=stage0_root,
        cache_manifest=manifest,
        model_name=model_name,
        dataset_name=dataset_name,
        subsets=subsets,
    )


def iter_cache_shards(
    *,
    stage0_root: Path | str,
    cache_manifest: Mapping[str, Any],
    model_name: str,
    dataset_name: str,
    subsets: Sequence[str],
) -> Iterator[CacheShard]:
    subset_set = set(subsets)
    shards = require_list(cache_manifest.get("shards"), name="cache manifest field 'shards'")
    matched: list[CacheShard] = []
    for index, raw_shard in enumerate(shards, start=1):
        shard = require_mapping(raw_shard, name=f"cache manifest shard {index}")
        if shard.get("model_name") != model_name:
            continue
        if shard.get("dataset_name") != dataset_name:
            continue
        if shard.get("subset") not in subset_set:
            continue
        matched.append(_build_cache_shard(stage0_root, shard, index=index))

    if not matched:
        raise ValueError(
            f"no cache shards found for model={model_name!r}, dataset={dataset_name!r}, "
            f"subsets={sorted(subset_set)!r}"
        )
    found_subsets = {str(shard.manifest_row["subset"]) for shard in matched}
    missing = sorted(subset_set - found_subsets)
    if missing:
        raise ValueError(f"missing cache shards for subsets: {missing}")
    for shard in matched:
        yield shard


def stream_repope_qwen_cache_entries(
    *,
    stage0_root: Path | str = DEFAULT_STAGE0_ROOT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    expected_num_layers: int = 36,
    expected_hidden_dim: int = 4096,
) -> Iterator[dict[str, Any]]:
    for shard in iter_repope_qwen_cache_shards(
        stage0_root=stage0_root,
        manifest_path=manifest_path,
        model_name=model_name,
        dataset_name=dataset_name,
        subsets=subsets,
    ):
        payload = _load_torch_payload(shard.path)
        entries = _payload_entries(payload, shard.path)
        _validate_shard_entry_count(entries, shard)
        for entry_index, entry in enumerate(entries, start=1):
            row = dict(require_mapping(entry, name=f"{shard.path}: entry {entry_index}"))
            validate_cache_entry(
                row,
                shard=shard,
                entry_index=entry_index,
                model_name=model_name,
                dataset_name=dataset_name,
                expected_num_layers=expected_num_layers,
                expected_hidden_dim=expected_hidden_dim,
            )
            yield row


def load_repope_qwen_cache_entries(**kwargs: Any) -> list[dict[str, Any]]:
    return list(stream_repope_qwen_cache_entries(**kwargs))


def validate_cache_entry(
    entry: Mapping[str, Any],
    *,
    shard: CacheShard,
    entry_index: int,
    model_name: str,
    dataset_name: str,
    expected_num_layers: int,
    expected_hidden_dim: int,
) -> None:
    context = f"{shard.path}: entry {entry_index}"
    for field in REQUIRED_ENTRY_FIELDS:
        require_field(entry, field, context=context)
    require_equal(entry["model_name"], model_name, field="model_name", context=context)
    require_equal(entry["dataset_name"], dataset_name, field="dataset_name", context=context)
    require_equal(entry["subset"], shard.manifest_row["subset"], field="subset", context=context)
    require_text(entry["sample_id"], field="sample_id", context=context)
    require_text(entry["image_id"], field="image_id", context=context)
    require_text(entry["question"], field="question", context=context)
    require_text(entry["object_name"], field="object_name", context=context)
    validate_layer_vectors(
        entry["layer_vectors"],
        expected_num_layers=expected_num_layers,
        expected_hidden_dim=expected_hidden_dim,
        context=context,
    )
    validate_first_token_logits(entry["first_token_logits"], context=context)


def validate_layer_vectors(
    value: object,
    *,
    expected_num_layers: int,
    expected_hidden_dim: int,
    context: str,
) -> None:
    shape = tuple(getattr(value, "shape", ()))
    expected = (int(expected_num_layers), int(expected_hidden_dim))
    if shape != expected:
        raise ValueError(f"{context}: layer_vectors shape must be {expected}, got {shape}")
    ensure_finite_array(value, field="layer_vectors", context=context)


def validate_first_token_logits(value: object, *, context: str) -> None:
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) != 1 or shape[0] <= 0:
        raise ValueError(f"{context}: first_token_logits must be a non-empty 1D tensor, got {shape}")
    ensure_finite_array(value, field="first_token_logits", context=context)


def _build_cache_shard(stage0_root: Path | str, shard: Mapping[str, Any], *, index: int) -> CacheShard:
    context = f"cache manifest shard {index}"
    if shard.get("status") != "passed":
        raise ValueError(f"{context}: expected status='passed', got {shard.get('status')!r}")
    path = resolve_existing_path(stage0_root, require_field(shard, "path", context=context))
    sidecar_path = resolve_existing_path(
        stage0_root,
        require_field(shard, "sidecar_path", context=context),
    )
    require_file(sidecar_path)
    sidecar = read_json_object(sidecar_path)
    _validate_sidecar(sidecar, shard=shard, sidecar_path=sidecar_path)
    return CacheShard(path=path, sidecar_path=sidecar_path, manifest_row=shard, sidecar=sidecar)


def _validate_sidecar(
    sidecar: Mapping[str, Any],
    *,
    shard: Mapping[str, Any],
    sidecar_path: Path,
) -> None:
    context = str(sidecar_path)
    for field in ("model_name", "dataset_name", "subset", "path", "num_entries"):
        require_field(sidecar, field, context=context)
    for field in ("model_name", "dataset_name", "subset", "num_entries"):
        require_equal(sidecar[field], shard[field], field=field, context=context)
    fields = require_list(sidecar.get("tensor_fields"), name=f"{context}: tensor_fields")
    expected_fields = {
        "layer_vectors": ("float16", (int(sidecar["num_selected_layers"]), int(sidecar["hidden_dim"]))),
        "first_token_logits": ("float32", None),
    }
    seen = {str(require_mapping(item, name=f"{context}: tensor_fields item").get("field")): item for item in fields}
    for field, (expected_dtype, expected_shape) in expected_fields.items():
        item = require_mapping(seen.get(field), name=f"{context}: tensor_fields[{field}]")
        require_equal(item.get("dtype"), expected_dtype, field=f"{field}.dtype", context=context)
        if expected_shape is not None:
            shape = tuple(require_list(item.get("shape"), name=f"{context}: {field}.shape"))
            require_equal(shape, expected_shape, field=f"{field}.shape", context=context)


def _load_torch_payload(path: Path) -> object:
    import torch

    return torch.load(path, weights_only=False, map_location="cpu")


def _payload_entries(payload: object, path: Path) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("entries", "records", "samples"):
            value = payload.get(key)
            if value is not None:
                return require_list(value, name=f"{path}: payload field '{key}'")
    raise ValueError(f"{path}: expected payload list or mapping with entries/records/samples")


def _validate_shard_entry_count(entries: Sequence[object], shard: CacheShard) -> None:
    expected = int(shard.manifest_row["num_entries"])
    if len(entries) != expected:
        raise ValueError(f"{shard.path}: expected {expected} entries, got {len(entries)}")
