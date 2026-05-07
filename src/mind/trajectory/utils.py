"""Small utilities for v2 Stage 0 dataset auditing."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from .metadata import UNKNOWN_OBJECT_NAME

_IMAGE_ID_PATTERN = re.compile(r"(\d+)")
_OBJECT_FROM_QUESTION_PATTERN = re.compile(
    r"^\s*is there\s+(?:a|an)\s+(?P<object>.+?)\s+in the image\?\s*$",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: line {line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number}: expected JSON object")
        rows.append(value)
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, object | None]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: object | None) -> object:
    if value is None:
        return ""
    return value


def parse_yes_no_label(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in {0, 1}:
            return value
        return int(value > 0)
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "1", "true", "present"}:
        return 1
    if normalized in {"no", "n", "0", "false", "absent"}:
        return 0
    return None


def parse_image_id(row: Mapping[str, object], image_path: str) -> str:
    explicit = row.get("image_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    match = _IMAGE_ID_PATTERN.findall(image_path)
    if not match:
        return ""
    return str(int(match[-1]))


def parse_object_name(row: Mapping[str, object], question: str) -> str:
    explicit = row.get("object_name") or row.get("object")
    if explicit is not None:
        value = str(explicit).strip()
        if value:
            return value
    match = _OBJECT_FROM_QUESTION_PATTERN.match(question)
    if match is None:
        return UNKNOWN_OBJECT_NAME
    value = match.group("object").strip().lower()
    return value or UNKNOWN_OBJECT_NAME


def normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()
