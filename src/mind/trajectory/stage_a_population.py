"""Stage A population construction from Stage 0 cache entries."""

from __future__ import annotations

from collections import Counter
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .utils import parse_yes_no_label


class PopulationClass(str, Enum):
    """Primary Stage A answer classes."""

    CORRECT = "correct"
    HARD_HALLUCINATION = "hard_hallucination"
    FALSE_NEGATIVE_ERROR = "false_negative_error"
    PARSED_NONE = "parsed_none"
    INVALID_LABEL = "invalid_label"


def classify_entry(entry: Mapping[str, object]) -> PopulationClass:
    """Classify one cached answer row for Stage A population construction."""

    label = parse_yes_no_label(entry.get("label"))
    parsed_answer = parse_yes_no_label(entry.get("parsed_answer"))
    if label is None:
        return PopulationClass.INVALID_LABEL
    if parsed_answer is None:
        return PopulationClass.PARSED_NONE
    if parsed_answer == label:
        return PopulationClass.CORRECT
    if label == 0 and parsed_answer == 1:
        return PopulationClass.HARD_HALLUCINATION
    return PopulationClass.FALSE_NEGATIVE_ERROR


def summarize_population(entries: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Count Stage A answer classes and the primary population size."""

    counts: Counter[PopulationClass] = Counter()
    total = 0
    for entry in entries:
        total += 1
        counts[classify_entry(entry)] += 1

    num_correct = counts[PopulationClass.CORRECT]
    num_hard_hallucination = counts[PopulationClass.HARD_HALLUCINATION]
    num_primary_population = num_correct + num_hard_hallucination
    hallucination_rate = (
        None
        if num_primary_population == 0
        else num_hard_hallucination / num_primary_population
    )
    return {
        "num_entries": total,
        "num_correct": num_correct,
        "num_hard_hallucination": num_hard_hallucination,
        "num_false_negative_error": counts[PopulationClass.FALSE_NEGATIVE_ERROR],
        "num_parsed_none": counts[PopulationClass.PARSED_NONE],
        "num_invalid_label": counts[PopulationClass.INVALID_LABEL],
        "num_primary_population": num_primary_population,
        "hallucination_rate_in_primary_population": hallucination_rate,
    }


def load_cache_manifest(stage0_root: Path | str, path: Path | str | None = None) -> dict[str, object]:
    """Load a Stage 0 cache manifest."""

    manifest_path = Path(path) if path is not None else Path(stage0_root) / "manifests" / "cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cache manifest must be a JSON object: {manifest_path}")
    return payload


def stream_stage0_cache_entries(
    stage0_root: Path | str,
    *,
    cache_manifest: Mapping[str, object] | None = None,
    cache_manifest_path: Path | str | None = None,
    dataset_names: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    include_tensors: bool = True,
) -> Iterator[dict[str, object]]:
    """Yield Stage 0 cache rows one shard at a time.

    Stage 0 stores tensors in ``.pt`` shards, so each shard is the smallest safe
    loading unit. This generator avoids accumulating all shard rows in memory.
    """

    root = Path(stage0_root)
    manifest = (
        dict(cache_manifest)
        if cache_manifest is not None
        else load_cache_manifest(root, cache_manifest_path)
    )
    dataset_filter = set(dataset_names) if dataset_names is not None else None
    model_filter = set(model_names) if model_names is not None else None

    for shard_path, shard in iter_cache_shards(root, manifest):
        payload = _load_torch_payload(shard_path)
        for entry in _iter_cache_payload_entries(payload):
            row = dict(entry)
            _fill_entry_metadata(row, shard)
            if dataset_filter is not None and str(row.get("dataset_name", "")) not in dataset_filter:
                continue
            if model_filter is not None and str(row.get("model_name", "")) not in model_filter:
                continue
            if not include_tensors:
                row.pop("layer_vectors", None)
                row.pop("first_token_logits", None)
            yield row


def iter_cache_shards(
    stage0_root: Path | str,
    cache_manifest: Mapping[str, object],
) -> Iterator[tuple[Path, Mapping[str, object]]]:
    """Yield shard paths from a Stage 0 cache manifest."""

    root = Path(stage0_root)
    shards = cache_manifest.get("shards", [])
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("cache manifest field 'shards' must be a list")
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        path_value = shard.get("path")
        if path_value is None:
            continue
        shard_path = _resolve_stage0_path(root, Path(str(path_value)))
        yield shard_path, shard


def _resolve_stage0_path(stage0_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = stage0_root / path
    if candidate.exists():
        return candidate
    cache_relative = stage0_root / "cache" / path
    if cache_relative.exists():
        return cache_relative
    return candidate


def _load_torch_payload(path: Path) -> object:
    import torch

    return torch.load(path, weights_only=False)


def _iter_cache_payload_entries(payload: object) -> Iterator[Mapping[str, object]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if not isinstance(payload, Mapping):
        return
    for key in ("entries", "records", "samples"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield item
            return
    for sample_id, value in payload.items():
        if isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("sample_id", sample_id)
            yield row


def _fill_entry_metadata(row: dict[str, object], shard: Mapping[str, object]) -> None:
    for field in ("model_name", "dataset_name", "source_dataset", "subset", "split"):
        value = shard.get(field)
        if value is not None and row.get(field) in (None, ""):
            row[field] = value
