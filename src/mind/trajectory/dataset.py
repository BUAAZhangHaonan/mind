"""Dataset discovery and normalization for v2 Stage 0 audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .metadata import DASH_B_DATASET_NAME, DASH_B_SUBSET, KNOWN_DATASET_NAMES, KNOWN_SUBSETS
from .utils import normalize_text, parse_image_id, parse_object_name, parse_yes_no_label, read_jsonl


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    subset: str
    path: Path


@dataclass(frozen=True)
class NormalizedRecord:
    sample_id: str
    image_id: str
    image_path: str
    question: str
    label: int | None
    object_name: str
    dataset_name: str
    subset: str
    path: str


def discover_known_datasets(repo_root: Path | str = ".") -> list[DatasetSpec]:
    root = Path(repo_root)
    specs: list[DatasetSpec] = []
    for dataset_name in KNOWN_DATASET_NAMES:
        for subset in KNOWN_SUBSETS:
            normalized = (
                root
                / "outputs"
                / "round2_2026_04"
                / "normalized"
                / dataset_name
                / f"{subset}.jsonl"
            )
            raw = root / "data" / dataset_name / f"{subset}.jsonl"
            specs.append(DatasetSpec(dataset_name, subset, normalized if normalized.exists() or not raw.exists() else raw))

    dash_b = (
        root
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / DASH_B_DATASET_NAME
        / f"{DASH_B_SUBSET}.jsonl"
    )
    if dash_b.exists():
        specs.append(DatasetSpec(DASH_B_DATASET_NAME, DASH_B_SUBSET, dash_b))
    return specs


def load_dataset_records(spec: DatasetSpec) -> list[NormalizedRecord]:
    return [
        normalize_record(row, spec=spec, index=index)
        for index, row in enumerate(read_jsonl(spec.path))
    ]


def normalize_record(row: Mapping[str, object], *, spec: DatasetSpec, index: int) -> NormalizedRecord:
    image_path = normalize_text(row.get("image_path") or row.get("image") or row.get("path"))
    question = normalize_text(row.get("question") or row.get("text") or row.get("prompt"))
    object_name = parse_object_name(row, question)
    sample_id = normalize_text(
        row.get("sample_id")
        or row.get("question_id")
        or row.get("id")
        or f"{spec.dataset_name}-{spec.subset}-{index}"
    )
    return NormalizedRecord(
        sample_id=sample_id,
        image_id=parse_image_id(row, image_path),
        image_path=image_path,
        question=question,
        label=parse_yes_no_label(row.get("label", row.get("answer", ""))),
        object_name=object_name,
        dataset_name=spec.dataset_name,
        subset=spec.subset,
        path=str(spec.path),
    )
