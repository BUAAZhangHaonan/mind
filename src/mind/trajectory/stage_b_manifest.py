"""Stage B manifest helpers for the unified full-cache panel."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from mind.models.registry import REQUIRED_MODEL_ALIASES
import torch

from .stage_a_closeout import FAMILY_SUBSETS


EXPECTED_STAGE_B_PANEL_SIZE = 16
UNIFIED_FULL_CACHE_MANIFEST_NAME = "unified_full_cache_manifest.json"


@dataclass(frozen=True)
class StageBPanelManifest:
    """Validated Stage B view of the unified full-cache manifest."""

    path: Path
    models: list[dict[str, object]]
    payload: dict[str, object]


def load_stage_b_panel_manifest(
    full_cache_root: Path | str,
    *,
    expected_panel_models: Sequence[str] | None = REQUIRED_MODEL_ALIASES,
    expected_panel_size: int = EXPECTED_STAGE_B_PANEL_SIZE,
) -> StageBPanelManifest:
    """Load and validate the Stage B full-cache panel manifest."""

    root = Path(full_cache_root)
    manifest_path = root / "manifests" / UNIFIED_FULL_CACHE_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unified full-cache manifest must be a JSON object: {manifest_path}")

    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("unified full-cache manifest must contain a models list")

    expected_aliases = tuple(expected_panel_models) if expected_panel_models is not None else None
    required_size = len(expected_aliases) if expected_aliases is not None else int(expected_panel_size)
    if len(models) != required_size:
        raise ValueError(
            f"Stage B requires {required_size} panel models; found {len(models)}"
        )

    rows: list[dict[str, object]] = []
    aliases: list[str] = []
    for row in models:
        if not isinstance(row, Mapping):
            raise ValueError("each unified manifest model row must be an object")
        model_row = dict(row)
        alias = str(model_row.get("model_alias", "")).strip()
        if not alias:
            raise ValueError("each unified manifest model row must have model_alias")
        aliases.append(alias)
        _validate_available_model_row(model_row)
        resolve_stage_b_cache_root(model_row, root)
        rows.append(model_row)

    duplicates = sorted(alias for alias, count in Counter(aliases).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate panel models in unified manifest: " + ", ".join(duplicates))

    if expected_aliases is not None:
        expected_set = set(expected_aliases)
        actual_set = set(aliases)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            details = []
            if missing:
                details.append("missing panel model(s): " + ", ".join(missing))
            if extra:
                details.append("unexpected panel model(s): " + ", ".join(extra))
            raise ValueError("Stage B 16 panel models do not match registry: " + "; ".join(details))
        if aliases != list(expected_aliases):
            raise ValueError("Stage B panel model order must match the registry order")

    return StageBPanelManifest(path=manifest_path, models=rows, payload=payload)


def resolve_stage_b_cache_root(
    model_row: Mapping[str, object],
    full_cache_root: Path | str,
) -> Path:
    """Resolve a Stage B model cache root from a unified manifest row."""

    root = Path(full_cache_root)
    for field in ("cache_root", "source_cache_root"):
        value = model_row.get(field)
        if value not in (None, ""):
            return _resolve_path(root, Path(str(value)))
    alias = model_row.get("model_alias", "<unknown>")
    raise ValueError(f"missing cache_root or source_cache_root for panel model {alias}")


def _validate_available_model_row(model_row: Mapping[str, object]) -> None:
    alias = str(model_row.get("model_alias", "<unknown>"))
    validation_status = str(model_row.get("validation_status", ""))
    if validation_status and validation_status != "passed":
        raise ValueError(f"{alias} validation_status is not passed: {validation_status}")

    status = str(model_row.get("status", ""))
    if status in {"failed_extraction", "failed_validation", "blocked"}:
        raise ValueError(f"{alias} is not available for Stage B: {status}")


def _resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def iter_stage_b_full_cache_shards(
    model_row: Mapping[str, object],
    full_cache_root: Path | str,
    *,
    dataset_family: str,
) -> Iterator[Path]:
    """Yield shard paths from the physical root recorded in the unified manifest."""

    family = _normalize_family(dataset_family)
    cache_root = resolve_stage_b_cache_root(model_row, full_cache_root)
    for subset in FAMILY_SUBSETS[family]:
        subset_dir = cache_root / family / subset
        for shard_path in sorted(subset_dir.glob("*.pt")):
            yield shard_path


def stream_stage_b_full_cache_entries(
    model_row: Mapping[str, object],
    full_cache_root: Path | str,
    *,
    dataset_family: str,
    include_tensors: bool = True,
) -> Iterator[dict[str, object]]:
    """Stream full-cache entries for Stage B using the unified manifest root only."""

    alias = str(model_row["model_alias"])
    family = _normalize_family(dataset_family)
    for shard_path in iter_stage_b_full_cache_shards(
        model_row,
        full_cache_root,
        dataset_family=family,
    ):
        payload = torch.load(shard_path, weights_only=False, map_location="cpu")
        for entry in _iter_payload_entries(payload):
            row = dict(entry)
            row.setdefault("model_alias", alias)
            row.setdefault("model_name", alias)
            row.setdefault("dataset_name", family)
            row.setdefault("source_dataset", family)
            row.setdefault("subset", shard_path.parent.name)
            if not include_tensors:
                row.pop("layer_vectors", None)
                row.pop("first_token_logits", None)
            yield row


def _iter_payload_entries(payload: object) -> Iterator[Mapping[str, object]]:
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, Mapping):
                yield entry
        return
    if isinstance(payload, Mapping):
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, Mapping):
                    yield entry


def _normalize_family(dataset_family: str) -> str:
    family = str(dataset_family).strip().lower()
    if family not in FAMILY_SUBSETS:
        raise ValueError(f"unsupported Stage B dataset family: {dataset_family}")
    return family
