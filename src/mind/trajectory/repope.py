"""RePOPE cache materialization from lossless POPE cache relabels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .utils import read_jsonl


MATCH_FIELDS = ("sample_id", "image_id", "image_path", "object_name", "question")


def materialize_repope_cache(
    *,
    source_cache_root: Path | str,
    source_records_path: Path | str,
    target_records_path: Path | str,
    output_root: Path | str,
    target_dataset_name: str = "repope",
    target_subset: str,
) -> list[Path]:
    """Copy POPE cache shards to RePOPE only when records match exactly."""

    source_root = Path(source_cache_root)
    output = Path(output_root)
    source_rows = read_jsonl(source_records_path)
    target_rows = read_jsonl(target_records_path)
    target_by_sample_id = _validate_lossless_relabel(source_rows, target_rows)

    shard_paths = sorted(path for path in source_root.rglob("*.pt") if path.is_file())
    if not shard_paths:
        raise FileNotFoundError(f"No source cache shards found under {source_root}")

    written_paths: list[Path] = []
    seen_samples: set[str] = set()
    for shard_path in shard_paths:
        sidecar = _load_sidecar(shard_path)
        payload = torch.load(shard_path, weights_only=False)
        if not isinstance(payload, list):
            raise ValueError(f"source shard payload must be a list of dicts: {shard_path}")

        materialized_entries: list[dict[str, object]] = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, Mapping):
                raise ValueError(f"{shard_path}: entry {index} is not a dict")
            sample_id = _required_text(entry.get("sample_id"), field="sample_id", context=str(shard_path))
            target_row = target_by_sample_id.get(sample_id)
            if target_row is None:
                raise ValueError(f"{shard_path}: cache entry sample_id is not present in target records: {sample_id}")
            _validate_entry_matches_target(entry, target_row, context=f"{shard_path}: entry {index}")
            seen_samples.add(sample_id)
            materialized_entries.append(
                _materialized_entry(
                    entry,
                    target_row=target_row,
                    target_dataset_name=target_dataset_name,
                    target_subset=target_subset,
                )
            )

        destination = _destination_path(
            shard_path,
            source_root=source_root,
            output_root=output,
            sidecar=sidecar,
            target_dataset_name=target_dataset_name,
            target_subset=target_subset,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(materialized_entries, destination)
        _write_sidecar(
            destination,
            sidecar=sidecar,
            num_entries=len(materialized_entries),
            source_cache_root=source_root,
            source_records_path=Path(source_records_path),
            target_records_path=Path(target_records_path),
            target_dataset_name=target_dataset_name,
            target_subset=target_subset,
        )
        written_paths.append(destination)

    missing = sorted(set(target_by_sample_id) - seen_samples)
    if missing:
        for path in written_paths:
            path.unlink(missing_ok=True)
            Path(str(path) + ".json").unlink(missing_ok=True)
        raise ValueError(f"target records are missing from source cache: {', '.join(missing[:5])}")
    return written_paths


def _validate_lossless_relabel(
    source_rows: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if len(source_rows) != len(target_rows):
        raise ValueError(
            "RePOPE materialization requires equal record counts: "
            f"source={len(source_rows)} target={len(target_rows)}"
        )

    target_by_sample_id: dict[str, Mapping[str, object]] = {}
    for index, (source, target) in enumerate(zip(source_rows, target_rows, strict=True), start=1):
        for field in MATCH_FIELDS:
            source_value = _match_value(source.get(field))
            target_value = _match_value(target.get(field))
            if source_value != target_value:
                raise ValueError(
                    "RePOPE materialization requires exact "
                    f"{field} match at record {index}: source={source_value!r} target={target_value!r}"
                )
        sample_id = _required_text(target.get("sample_id"), field="sample_id", context=f"record {index}")
        if sample_id in target_by_sample_id:
            raise ValueError(f"duplicate target sample_id: {sample_id}")
        target_by_sample_id[sample_id] = target
    return target_by_sample_id


def _validate_entry_matches_target(
    entry: Mapping[str, object],
    target_row: Mapping[str, object],
    *,
    context: str,
) -> None:
    for field in MATCH_FIELDS:
        entry_value = _match_value(entry.get(field))
        target_value = _match_value(target_row.get(field))
        if entry_value != target_value:
            raise ValueError(
                f"{context}: cache entry {field} does not match target record: "
                f"entry={entry_value!r} target={target_value!r}"
            )


def _materialized_entry(
    entry: Mapping[str, object],
    *,
    target_row: Mapping[str, object],
    target_dataset_name: str,
    target_subset: str,
) -> dict[str, object]:
    materialized = dict(entry)
    for field in ("sample_id", "image_id", "image_path", "question", "object_name", "label"):
        if field in target_row:
            materialized[field] = target_row[field]
    materialized["dataset_name"] = target_dataset_name
    materialized["source_dataset"] = target_dataset_name
    materialized["subset"] = target_subset
    materialized["split"] = target_subset
    return materialized


def _destination_path(
    shard_path: Path,
    *,
    source_root: Path,
    output_root: Path,
    sidecar: Mapping[str, object],
    target_dataset_name: str,
    target_subset: str,
) -> Path:
    model_name = _optional_text(sidecar.get("model_name"))
    if model_name is not None:
        return output_root / model_name / target_dataset_name / target_subset / shard_path.name

    relative = shard_path.relative_to(source_root)
    parts = list(relative.parts)
    if len(parts) >= 3:
        parts[-3] = target_dataset_name
        parts[-2] = target_subset
        return output_root.joinpath(*parts)
    return output_root / target_dataset_name / target_subset / shard_path.name


def _load_sidecar(shard_path: Path) -> dict[str, object]:
    sidecar_path = Path(str(shard_path) + ".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"source sidecar metadata is missing: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source sidecar metadata must be a JSON object: {sidecar_path}")
    return dict(payload)


def _write_sidecar(
    shard_path: Path,
    *,
    sidecar: Mapping[str, object],
    num_entries: int,
    source_cache_root: Path,
    source_records_path: Path,
    target_records_path: Path,
    target_dataset_name: str,
    target_subset: str,
) -> None:
    payload = dict(sidecar)
    payload.update(
        {
            "dataset_name": target_dataset_name,
            "source_dataset": target_dataset_name,
            "subset": target_subset,
            "split": target_subset,
            "num_entries": int(num_entries),
            "records_path": str(target_records_path),
            "materialization_source": "pope_cache_relabel",
            "materialization_source_cache_root": str(source_cache_root),
            "materialization_source_records_path": str(source_records_path),
        }
    )
    Path(str(shard_path) + ".json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _match_value(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_text(value: object | None) -> str | None:
    return _match_value(value)


def _required_text(value: object | None, *, field: str, context: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{context}: missing required {field}")
    return text
