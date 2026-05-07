"""Dataset discovery and normalization for v2 Stage 0 audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .metadata import DASH_B_DATASET_NAME, DASH_B_SUBSET, KNOWN_DATASET_NAMES, KNOWN_SUBSETS
from .utils import normalize_text, parse_image_id, parse_object_name, parse_yes_no_label, read_jsonl

EXTRACTION_READY_REQUIRED_FIELDS = (
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
)


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


def normalized_dataset_path(
    *,
    repo_root: Path | str,
    dataset_name: str,
    subset: str,
) -> Path:
    root = Path(repo_root)
    return (
        root
        / "outputs"
        / "round2_2026_04"
        / "normalized"
        / dataset_name
        / f"{subset}.jsonl"
    )


def raw_dataset_path(
    *,
    repo_root: Path | str,
    dataset_name: str,
    subset: str,
) -> Path:
    return Path(repo_root) / "data" / dataset_name / f"{subset}.jsonl"


def validate_extraction_ready_dataset_specs(
    specs: Sequence[DatasetSpec],
    *,
    repo_root: Path | str | None = None,
    require_normalized: bool = False,
) -> None:
    for spec in specs:
        if require_normalized:
            if repo_root is None:
                raise ValueError("repo_root is required when require_normalized=True")
            expected = normalized_dataset_path(
                repo_root=repo_root,
                dataset_name=spec.dataset_name,
                subset=spec.subset,
            )
            if spec.path != expected:
                raise ValueError(
                    "Full-run requires normalized extraction-ready records for "
                    f"{spec.dataset_name}/{spec.subset}: expected {expected}, got {spec.path}. "
                    "Raw POPE files are not accepted for full-run extraction."
                )

        if not spec.path.exists():
            if require_normalized and repo_root is not None:
                raw = raw_dataset_path(
                    repo_root=repo_root,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                )
                raw_note = (
                    f"; raw file exists at {raw} but full-run does not accept raw POPE files"
                    if raw.exists()
                    else ""
                )
                raise FileNotFoundError(
                    "Normalized extraction-ready dataset is missing for full-run: "
                    f"{spec.dataset_name}/{spec.subset} at {spec.path}{raw_note}"
                )
            raise FileNotFoundError(
                f"Required dataset is missing: {spec.dataset_name}/{spec.subset} at {spec.path}"
            )

        validate_extraction_ready_rows(read_jsonl(spec.path), path=spec.path)


def validate_extraction_ready_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    path: Path | str,
    required_fields: Sequence[str] = EXTRACTION_READY_REQUIRED_FIELDS,
) -> None:
    for index, row in enumerate(rows, start=1):
        validate_extraction_ready_row(
            row,
            path=path,
            record_number=index,
            required_fields=required_fields,
        )


def validate_extraction_ready_row(
    row: Mapping[str, object],
    *,
    path: Path | str,
    record_number: int,
    required_fields: Sequence[str] = EXTRACTION_READY_REQUIRED_FIELDS,
) -> None:
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(
            f"{path}: record {record_number}: missing required extraction-ready fields: "
            f"{', '.join(missing)}. Raw POPE files must be normalized before extraction."
        )

    blank = [
        field
        for field in required_fields
        if field != "label" and not normalize_text(row.get(field))
    ]
    if blank:
        raise ValueError(
            f"{path}: record {record_number}: blank required extraction-ready fields: "
            f"{', '.join(blank)}"
        )

    label = row.get("label")
    if isinstance(label, bool) or label not in {0, 1}:
        raise ValueError(
            f"{path}: record {record_number}: label must be 0 or 1, got {label!r}"
        )


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
