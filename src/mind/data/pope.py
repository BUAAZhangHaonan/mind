"""POPE-style benchmark loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from .types import HallucinationRecord


_IMAGE_ID_PATTERN = re.compile(r"(\d+)")


class DatasetUnavailableError(FileNotFoundError):
    """Raised when an optional benchmark is not available locally."""


def _parse_label(value: str | int | bool) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value > 0)
    normalized = str(value).strip().lower()
    if normalized in {"yes", "1", "true", "present"}:
        return 1
    if normalized in {"no", "0", "false", "absent"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def _parse_image_id(image_path: str) -> int:
    matches = _IMAGE_ID_PATTERN.findall(image_path)
    if not matches:
        raise ValueError(f"Could not parse image id from: {image_path}")
    return int(matches[-1])


def _coerce_rows(source: Path | Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix == ".json":
            return list(json.loads(source.read_text(encoding="utf-8")))
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return list(source)


def load_pope_records(
    source: Path | Sequence[dict[str, object]],
    *,
    subset: str,
    split: str,
    source_dataset: str = "pope",
) -> list[HallucinationRecord]:
    rows = _coerce_rows(source)
    records: list[HallucinationRecord] = []
    for index, row in enumerate(rows):
        image_path = str(row.get("image") or row.get("image_path") or "")
        sample_id = str(row.get("sample_id") or row.get("question_id") or f"{subset}-{index}")
        question = str(row.get("text") or row.get("question") or "")
        object_name = str(row.get("object") or row.get("object_name") or "unknown")
        records.append(
            HallucinationRecord(
                sample_id=sample_id,
                image_id=_parse_image_id(image_path),
                image_path=image_path,
                question=question,
                label=_parse_label(row.get("label", 0)),
                object_name=object_name,
                split=split,
                subset=subset,
                source_dataset=source_dataset,
            )
        )
    return records


def apply_repope_labels(
    records: Sequence[HallucinationRecord],
    relabel_rows: Iterable[dict[str, object]],
) -> list[HallucinationRecord]:
    relabel_map = {
        str(row["sample_id"]): _parse_label(row["label"])
        for row in relabel_rows
        if "sample_id" in row and "label" in row
    }
    relabeled: list[HallucinationRecord] = []
    for record in records:
        if record.sample_id in relabel_map:
            relabeled.append(record.with_label(relabel_map[record.sample_id], "repope"))
        else:
            relabeled.append(record)
    return relabeled


def load_hpope_records(
    source: Path,
    *,
    subset: str,
    split: str,
) -> list[HallucinationRecord]:
    if not source.exists():
        raise DatasetUnavailableError(
            f"H-POPE assets were not found at {source}. "
            "Keep the loader wired in place and document the missing public files."
        )
    return load_pope_records(source, subset=subset, split=split, source_dataset="hpope")
