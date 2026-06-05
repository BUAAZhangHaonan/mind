"""Shared helpers for wavelet-course data construction."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_SEED = 20260506
DEFAULT_MODEL_NAME = "qwen3-vl-8b"
DEFAULT_DATASET_NAME = "repope"
DEFAULT_SUBSETS = ("popular", "random", "adversarial")
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.60, 0.20, 0.20)

T = TypeVar("T")


def read_json_object(path: Path | str) -> dict[str, Any]:
    """Read a JSON object and fail if the payload is not an object."""

    json_path = require_file(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{json_path}: invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path}: expected a JSON object")
    return payload


def write_json_object(path: Path | str, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_csv_rows(
    path: Path | str,
    rows: Iterable[Mapping[str, object | None]],
    columns: Sequence[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def require_file(path: Path | str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"required file is missing: {value}")
    return value


def resolve_existing_path(root: Path | str, path: Path | str) -> Path:
    value = Path(path)
    candidates = [value]
    if not value.is_absolute():
        root_path = Path(root)
        candidates.extend([root_path / value, root_path / "cache" / value])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"required path is missing: {value}")


def require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def require_list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def require_field(row: Mapping[str, Any], field: str, *, context: str) -> Any:
    if field not in row:
        raise ValueError(f"{context}: missing required field '{field}'")
    value = row[field]
    if value is None:
        raise ValueError(f"{context}: field '{field}' is null")
    return value


def require_text(value: object, *, field: str, context: str) -> str:
    if value is None:
        raise ValueError(f"{context}: missing required text field '{field}'")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context}: blank required text field '{field}'")
    return text


def optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_equal(actual: object, expected: object, *, field: str, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {field}={expected!r}, got {actual!r}")


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


def require_binary_label(value: object, *, field: str, context: str) -> int:
    label = parse_yes_no_label(value)
    if label is None:
        raise ValueError(f"{context}: field '{field}' is not a yes/no label: {value!r}")
    return label


def is_finite_scalar(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def ensure_finite_array(value: object, *, field: str, context: str) -> None:
    """Check finite values for numpy arrays, torch tensors, or flat scalars."""

    if hasattr(value, "isfinite"):
        finite = value.isfinite()
        if hasattr(finite, "all"):
            result = finite.all()
            if hasattr(result, "item"):
                result = result.item()
            if not bool(result):
                raise ValueError(f"{context}: field '{field}' contains non-finite values")
            return
    try:
        import numpy as np

        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"{context}: field '{field}' contains non-finite values")
        return
    except TypeError as error:
        raise ValueError(f"{context}: field '{field}' cannot be checked for finiteness") from error


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
    except ImportError:
        pass


def validate_ratios(ratios: Sequence[float], *, split_names: Sequence[str]) -> tuple[float, ...]:
    if len(ratios) != len(split_names):
        raise ValueError(f"expected {len(split_names)} split ratios, got {len(ratios)}")
    values = tuple(float(ratio) for ratio in ratios)
    if any(not math.isfinite(ratio) or ratio < 0.0 for ratio in values):
        raise ValueError("split ratios must be finite non-negative values")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")
    return values


def largest_remainder_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def require_non_empty(values: Sequence[T], *, name: str) -> Sequence[T]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values
