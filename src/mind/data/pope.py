"""POPE-style benchmark loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from .types import HallucinationRecord


_IMAGE_ID_PATTERN = re.compile(r"(\d+)")
_OBJECT_FROM_QUESTION_PATTERN = re.compile(
    r"^\s*is there\s+(?:a|an)\s+(?P<object>.+?)\s+in the image\?\s*$",
    re.IGNORECASE,
)


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


def _parse_object_name(row: dict[str, object], question: str) -> str:
    explicit = row.get("object") or row.get("object_name")
    if explicit is not None:
        value = str(explicit).strip()
        if value:
            return value
    match = _OBJECT_FROM_QUESTION_PATTERN.match(question)
    if match is None:
        return "unknown"
    return match.group("object").strip().lower()


def _build_question(
    row: dict[str, object],
    *,
    object_name: str,
    question_template: str | None,
) -> str:
    question = str(row.get("text") or row.get("question") or row.get("prompt") or "").strip()
    if question:
        return question
    if question_template is None:
        return ""
    return question_template.format(object_name=object_name)


def load_object_yes_no_records(
    source: Path | Sequence[dict[str, object]],
    *,
    subset: str,
    split: str,
    source_dataset: str = "pope",
    question_template: str | None = None,
) -> list[HallucinationRecord]:
    rows = _coerce_rows(source)
    records: list[HallucinationRecord] = []
    for index, row in enumerate(rows):
        image_path = str(row.get("image") or row.get("image_path") or "")
        sample_id = str(
            row.get("sample_id")
            or row.get("question_id")
            or row.get("id")
            or f"{subset}-{index}"
        )
        question = str(row.get("text") or row.get("question") or row.get("prompt") or "")
        object_name = _parse_object_name(row, question)
        question = _build_question(
            row,
            object_name=object_name,
            question_template=question_template,
        )
        records.append(
            HallucinationRecord(
                sample_id=sample_id,
                image_id=_parse_image_id(image_path),
                image_path=image_path,
                question=question,
                label=_parse_label(row.get("label", row.get("answer", 0))),
                object_name=object_name,
                split=split,
                subset=subset,
                source_dataset=source_dataset,
            )
        )
    return records


def load_pope_records(
    source: Path | Sequence[dict[str, object]],
    *,
    subset: str,
    split: str,
    source_dataset: str = "pope",
) -> list[HallucinationRecord]:
    return load_object_yes_no_records(
        source,
        subset=subset,
        split=split,
        source_dataset=source_dataset,
    )


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
