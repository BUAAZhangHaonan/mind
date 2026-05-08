"""Dataset discovery and normalization for Stage 0 audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from mind.data import load_object_yes_no_records

from .metadata import (
    DASH_B_DATASET_NAME,
    DASH_B_LEGACY_SUBSET,
    DASH_B_SUBSET,
    DATASET_SUBSETS,
    POPE_STYLE_DATASET_NAMES,
)
from .utils import normalize_text, parse_image_id, parse_object_name, parse_yes_no_label, read_jsonl

EXTRACTION_READY_REQUIRED_FIELDS = (
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
)
NORMALIZED_DATASET_FIELDS = (
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
    "source_dataset",
    "split",
    "subset",
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
    for dataset_name in POPE_STYLE_DATASET_NAMES:
        for subset in DATASET_SUBSETS[dataset_name]:
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

    dash_b = normalized_dataset_path(
        repo_root=root,
        dataset_name=DASH_B_DATASET_NAME,
        subset=DASH_B_SUBSET,
    )
    dash_b_raw = raw_dataset_path(
        repo_root=root,
        dataset_name=DASH_B_DATASET_NAME,
        subset=DASH_B_SUBSET,
    )
    if dash_b.exists():
        specs.append(DatasetSpec(DASH_B_DATASET_NAME, DASH_B_SUBSET, dash_b))
    elif dash_b_raw.exists():
        specs.append(DatasetSpec(DASH_B_DATASET_NAME, DASH_B_SUBSET, dash_b_raw))

    legacy_dash_b = normalized_dataset_path(
        repo_root=root,
        dataset_name=DASH_B_DATASET_NAME,
        subset=DASH_B_LEGACY_SUBSET,
    )
    if legacy_dash_b.exists():
        specs.append(DatasetSpec(DASH_B_DATASET_NAME, DASH_B_LEGACY_SUBSET, legacy_dash_b))
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
    if dataset_name == DASH_B_DATASET_NAME and subset == DASH_B_SUBSET:
        candidates = _dash_b_raw_dataset_paths(Path(repo_root))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return Path(repo_root) / "data" / dataset_name / f"{subset}.jsonl"


def _dash_b_raw_dataset_paths(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "data" / "dash_b",
        repo_root / "data" / "dash-b",
    )


def planned_missing_normalized_dataset_materializations(
    specs: Sequence[DatasetSpec],
    *,
    repo_root: Path | str,
) -> list[dict[str, object]]:
    plans: list[dict[str, object]] = []
    for spec in specs:
        plan = missing_normalized_dataset_materialization_plan(spec, repo_root=repo_root)
        if plan is not None:
            plans.append(plan)
    return plans


def missing_normalized_dataset_materialization_plan(
    spec: DatasetSpec,
    *,
    repo_root: Path | str,
) -> dict[str, object] | None:
    source_path = materializable_normalized_source_path(spec, repo_root=repo_root)
    if source_path is None:
        return None
    rows = load_materializable_normalized_rows(spec, repo_root=repo_root)
    if rows is None:
        return None
    return {
        "dataset_name": spec.dataset_name,
        "subset": spec.subset,
        "source_path": str(source_path),
        "output_path": str(spec.path),
        "record_count": len(rows),
        "status": "planned",
    }


def materializable_normalized_source_path(
    spec: DatasetSpec,
    *,
    repo_root: Path | str,
) -> Path | None:
    if spec.path.exists():
        return None
    root = Path(repo_root)
    expected = normalized_dataset_path(
        repo_root=root,
        dataset_name=spec.dataset_name,
        subset=spec.subset,
    )
    if not _same_path(spec.path, expected):
        return None
    if spec.dataset_name in POPE_STYLE_DATASET_NAMES:
        raw = raw_dataset_path(
            repo_root=root,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
        )
        return raw if raw.exists() and raw.is_file() else None
    if spec.dataset_name == DASH_B_DATASET_NAME and spec.subset == DASH_B_SUBSET:
        raw = raw_dataset_path(
            repo_root=root,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
        )
        return raw if raw.exists() and raw.is_dir() else None
    return None


def load_materializable_normalized_rows(
    spec: DatasetSpec,
    *,
    repo_root: Path | str,
) -> list[dict[str, object]] | None:
    source_path = materializable_normalized_source_path(spec, repo_root=repo_root)
    if source_path is None:
        return None
    if spec.dataset_name in POPE_STYLE_DATASET_NAMES:
        return build_pope_family_normalized_rows(
            repo_root=repo_root,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            source_path=source_path,
        )
    if spec.dataset_name == DASH_B_DATASET_NAME and spec.subset == DASH_B_SUBSET:
        return build_dash_b_normalized_rows(repo_root=repo_root)
    return None


def build_pope_family_normalized_rows(
    *,
    repo_root: Path | str,
    dataset_name: str,
    subset: str,
    source_path: Path | str | None = None,
) -> list[dict[str, object]]:
    root = Path(repo_root)
    source = (
        Path(source_path)
        if source_path is not None
        else raw_dataset_path(repo_root=root, dataset_name=dataset_name, subset=subset)
    )
    if dataset_name not in POPE_STYLE_DATASET_NAMES:
        raise ValueError(f"POPE-family normalization does not support dataset: {dataset_name}")
    if subset not in DATASET_SUBSETS[dataset_name]:
        raise ValueError(f"Unknown subset for {dataset_name}: {subset}")
    if not source.exists():
        raise FileNotFoundError(f"POPE-family raw file is missing: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"POPE-family raw path is not a file: {source}")

    records = load_object_yes_no_records(
        source,
        subset=subset,
        split=subset,
        source_dataset=dataset_name,
    )
    rows = [_normalized_record_row(asdict(record)) for record in records]
    validate_extraction_ready_rows(rows, path=source)
    _validate_pope_family_image_paths(rows, repo_root=root, source_path=source)
    return rows


def materialize_pope_family_normalized_records(
    *,
    repo_root: Path | str,
    dataset_name: str,
    subset: str,
    output_path: Path | str | None = None,
) -> Path:
    root = Path(repo_root)
    destination = (
        Path(output_path)
        if output_path is not None
        else normalized_dataset_path(
            repo_root=root,
            dataset_name=dataset_name,
            subset=subset,
        )
    )
    rows = build_pope_family_normalized_rows(
        repo_root=root,
        dataset_name=dataset_name,
        subset=subset,
    )
    _write_jsonl_rows(destination, rows)
    return destination


def build_dash_b_normalized_rows(*, repo_root: Path | str) -> list[dict[str, object]]:
    root = Path(repo_root)
    raw_root = raw_dataset_path(
        repo_root=root,
        dataset_name=DASH_B_DATASET_NAME,
        subset=DASH_B_SUBSET,
    )
    if not raw_root.exists():
        raise FileNotFoundError(f"DASH-B raw directory is missing: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"DASH-B raw path is not a directory: {raw_root}")

    records = load_object_yes_no_records(
        raw_root,
        subset=DASH_B_SUBSET,
        split=DASH_B_SUBSET,
        source_dataset=DASH_B_DATASET_NAME,
    )
    rows = [_normalized_record_row(asdict(record)) for record in records]
    validate_extraction_ready_rows(rows, path=raw_root)
    return rows


def materialize_dash_b_normalized_records(
    *,
    repo_root: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Write canonical DASH-B ``all.jsonl`` records from ``data/dash_b``."""

    root = Path(repo_root)
    destination = (
        Path(output_path)
        if output_path is not None
        else normalized_dataset_path(
            repo_root=root,
            dataset_name=DASH_B_DATASET_NAME,
            subset=DASH_B_SUBSET,
        )
    )
    rows = build_dash_b_normalized_rows(repo_root=root)
    _write_jsonl_rows(destination, rows)
    return destination


def materialize_missing_normalized_dataset_specs(
    specs: Sequence[DatasetSpec],
    *,
    repo_root: Path | str,
) -> list[DatasetSpec]:
    materialized: list[DatasetSpec] = []
    for spec in specs:
        rows = load_materializable_normalized_rows(spec, repo_root=repo_root)
        if rows is not None:
            path = spec.path
            _write_jsonl_rows(path, rows)
            materialized.append(DatasetSpec(spec.dataset_name, spec.subset, path))
        else:
            materialized.append(spec)
    return materialized


def _normalized_record_row(row: Mapping[str, object]) -> dict[str, object]:
    return {field: row[field] for field in NORMALIZED_DATASET_FIELDS}


def _write_jsonl_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _validate_pope_family_image_paths(
    rows: Sequence[Mapping[str, object]],
    *,
    repo_root: Path,
    source_path: Path,
) -> None:
    image_root = repo_root / "data" / "coco" / "val2014"
    if not image_root.exists():
        raise FileNotFoundError(f"POPE-family image root is missing: {image_root}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"POPE-family image root is not a directory: {image_root}")
    image_root_resolved = image_root.resolve()
    for index, row in enumerate(rows, start=1):
        image_text = normalize_text(row.get("image_path"))
        image_path = Path(image_text)
        candidate = image_path if image_path.is_absolute() else image_root / image_path
        resolved = candidate.resolve()
        try:
            resolved.relative_to(image_root_resolved)
        except ValueError as error:
            raise ValueError(
                f"{source_path}: record {index}: image_path must resolve under {image_root}: "
                f"{image_text}"
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{source_path}: record {index}: image file is missing under {image_root}: "
                f"{image_text}"
            )


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


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
    if (
        spec.dataset_name == DASH_B_DATASET_NAME
        and spec.subset == DASH_B_SUBSET
        and spec.path.is_dir()
    ):
        rows = build_dash_b_normalized_rows(repo_root=_repo_root_for_raw_dash_b_path(spec.path))
        return [normalize_record(row, spec=spec, index=index) for index, row in enumerate(rows)]
    return [
        normalize_record(row, spec=spec, index=index)
        for index, row in enumerate(read_jsonl(spec.path))
    ]


def _repo_root_for_raw_dash_b_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved.name in {"dash_b", "dash-b"} and resolved.parent.name == "data":
        return resolved.parent.parent
    return path.parent


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
