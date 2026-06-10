"""Full-cache validation and manifest helpers for MIND Experiment 2."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


SHARD_SCHEMA_VERSION = "mind_full_cache_shard_v1"
ROUTE_MANIFEST_SCHEMA_VERSION = "mind_full_cache_route_manifest_v1"
MODEL_MANIFEST_SCHEMA_VERSION = "mind_full_cache_model_manifest_v1"
UNIFIED_MANIFEST_SCHEMA_VERSION = "mind_full_cache_unified_manifest_v1"
PANEL_CONFIG_SCHEMA_VERSION = "mind_full_cache_model_panel_v1"

DATASET_MATRIX: tuple[tuple[str, str], ...] = (
    ("pope", "popular"),
    ("pope", "random"),
    ("pope", "adversarial"),
    ("repope", "popular"),
    ("repope", "random"),
    ("repope", "adversarial"),
    ("dash-b", "all"),
)

STAGE0_ACCEPT_MODEL_ALIASES: tuple[str, ...] = (
    "qwen3-vl-8b",
    "internvl3.5-8b",
)
SEPARATE_ENV_ACCEPT_MODEL_ALIASES: tuple[str, ...] = ("gemma-4-12b-it",)
SEPARATE_ENV_EXTRACT_MODEL_ALIASES: tuple[str, ...] = ("molmo-7b-d-0924",)

VALID_ROUTES: tuple[str, ...] = (
    "accept_existing_stage0",
    "accept_existing_separate_env",
    "extract_default_env",
    "extract_separate_env",
)
VALID_FULL_CACHE_STATUSES: tuple[str, ...] = (
    "accepted_existing_stage0",
    "extracted_main_env",
    "accepted_existing_separate_env",
    "extracted_separate_env",
    "failed_extraction",
    "failed_validation",
    "needs_extraction_separate_env",
    "rejected",
)

REQUIRED_ENTRY_FIELDS: tuple[str, ...] = (
    "model_alias",
    "model_family",
    "dataset_name",
    "source_dataset",
    "subset",
    "split",
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
    "answer_text",
    "parsed_answer",
    "selected_layers",
    "layer_vectors",
    "first_token_logits",
    "token_index",
    "prompt_template_id",
    "logit_source",
)

REQUIRED_SIDECAR_FIELDS: tuple[str, ...] = (
    "schema_version",
    "cache_type",
    "cache_origin",
    "model_alias",
    "model_family",
    "dataset_name",
    "source_dataset",
    "subset",
    "split",
    "total_layers",
    "selected_layers",
    "num_selected_layers",
    "hidden_dim",
    "token_index",
    "dtype",
    "num_entries",
    "prompt_template_id",
    "logit_source",
)

DERIVABLE_LEGACY_ENTRY_FIELDS: frozenset[str] = frozenset(
    ("model_alias", "model_family", "token_index", "prompt_template_id", "logit_source")
)

QuestionRecord = tuple[str, str, str, str, str | None]


class FullCacheValidationError(ValueError):
    """Raised when a full-cache root fails validation."""

    def __init__(self, manifest: Mapping[str, object]) -> None:
        self.manifest = dict(manifest)
        errors = manifest.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, str) and errors:
            message = "; ".join(str(error) for error in errors[:3])
        else:
            message = "Full-cache validation failed"
        super().__init__(message)


def validate_full_cache_root(
    *,
    cache_root: str | Path,
    expected_model_alias: str,
    cache_origin: str,
    output: str | Path | None = None,
    extraction_env_name: str | None = None,
    raise_on_error: bool = True,
) -> dict[str, object]:
    """Validate every ``*.pt`` full-cache shard under ``cache_root``.

    The validator checks sidecar schema fields, list-of-dict shard payloads,
    finite ``layer_vectors`` and ``first_token_logits`` tensors, all-layer
    contiguous selection, logit source metadata, and sample-question
    preservation.
    """

    manifest = build_full_cache_validation_manifest(
        cache_root=cache_root,
        expected_model_alias=expected_model_alias,
        cache_origin=cache_origin,
        extraction_env_name=extraction_env_name,
    )
    if output is not None:
        write_json_manifest(manifest, output)
    if raise_on_error and manifest["status"] != "passed":
        raise FullCacheValidationError(manifest)
    return manifest


def build_full_cache_validation_manifest(
    *,
    cache_root: str | Path,
    expected_model_alias: str,
    cache_origin: str,
    extraction_env_name: str | None = None,
) -> dict[str, object]:
    root = Path(cache_root)
    errors: list[str] = []
    shards: list[dict[str, object]] = []
    duplicate_counter: Counter[tuple[str, str, str, str, str]] = Counter()
    question_records: list[QuestionRecord] = []
    total_entries = 0
    dataset_counts: Counter[str] = Counter()

    if not root.exists():
        errors.append(f"Cache root does not exist: {root}")
    elif not root.is_dir():
        errors.append(f"Cache root is not a directory: {root}")
    else:
        shard_paths = sorted(path for path in root.rglob("*.pt") if path.is_file())
        if not shard_paths:
            errors.append(f"No .pt cache shards found under {root}")
        for shard_path in shard_paths:
            shard, keys, sample_questions = _validate_full_cache_shard(
                shard_path,
                expected_model_alias=expected_model_alias,
                expected_cache_origin=cache_origin,
                expected_extraction_env_name=extraction_env_name,
            )
            shards.append(shard)
            total_entries += int(shard.get("num_entries") or 0)
            dataset_key = _dataset_key(shard.get("dataset_name"), shard.get("subset"))
            if dataset_key is not None:
                dataset_counts[dataset_key] += int(shard.get("num_entries") or 0)
            duplicate_counter.update(keys)
            question_records.extend(sample_questions)

    duplicate_keys = sorted(key for key, count in duplicate_counter.items() if count > 1)
    for key in duplicate_keys:
        errors.append(
            "duplicate full-cache key "
            f"model_alias={key[0]} dataset_name={key[1]} subset={key[2]} "
            f"split={key[3]} sample_id={key[4]}"
        )

    question_preservation = _question_preservation_report(question_records)
    question_errors = [
        str(error)
        for error in question_preservation.get("errors", [])
    ]
    errors.extend(question_errors)

    for shard in shards:
        for error in shard["errors"]:  # type: ignore[index]
            errors.append(f"{shard['path']}: {error}")

    manifest = {
        "schema_version": "mind_full_cache_validation_manifest_v1",
        "status": "failed" if errors else "passed",
        "cache_root": str(root),
        "expected_model_alias": expected_model_alias,
        "cache_origin": cache_origin,
        "extraction_env_name": extraction_env_name,
        "total_entries": total_entries,
        "num_shards": len(shards),
        "datasets": {
            key: {"num_entries": count}
            for key, count in sorted(dataset_counts.items())
        },
        "shards": shards,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "question_preservation": question_preservation,
        "errors": errors,
    }
    return manifest


def accept_existing_stage0_cache(
    *,
    model_alias: str,
    stage0_cache_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Accept an existing Stage 0 full-layer cache without copying tensors."""

    validation = validate_full_cache_root(
        cache_root=stage0_cache_root,
        expected_model_alias=model_alias,
        cache_origin="stage0",
    )
    report = _acceptance_report(
        status="accepted_existing_stage0",
        route="accept_existing_stage0",
        model_alias=model_alias,
        cache_origin="stage0",
        source_cache_root=Path(stage0_cache_root),
        output_root=Path(output_root),
        validation=validation,
    )
    write_json_manifest(report, report["manifest_path"])
    return report


def accept_separate_env_cache(
    *,
    model_alias: str,
    separate_env_cache_root: str | Path,
    output_root: str | Path,
    extraction_env_name: str,
) -> dict[str, object]:
    """Accept an existing separate-environment cache without copying tensors."""

    if not str(extraction_env_name).strip():
        raise ValueError("extraction_env_name is required for separate-env acceptance")
    validation = validate_full_cache_root(
        cache_root=separate_env_cache_root,
        expected_model_alias=model_alias,
        cache_origin="separate_env",
        extraction_env_name=extraction_env_name,
    )
    report = _acceptance_report(
        status="accepted_existing_separate_env",
        route="accept_existing_separate_env",
        model_alias=model_alias,
        cache_origin="separate_env",
        source_cache_root=Path(separate_env_cache_root),
        output_root=Path(output_root),
        validation=validation,
        extraction_env_name=extraction_env_name,
    )
    write_json_manifest(report, report["manifest_path"])
    return report


def build_route_manifest(
    *,
    output: str | Path | None = None,
    model_aliases: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build the pre-extraction route manifest for the final 16-model panel."""

    aliases = list(model_aliases) if model_aliases is not None else list(_required_model_aliases())
    _require_exact_panel_order(aliases)
    models: list[dict[str, object]] = []
    execution_plan: list[dict[str, object]] = []
    for alias in aliases:
        route = route_for_model(alias)
        requires_extraction = route in {"extract_default_env", "extract_separate_env"}
        status = "needs_extraction" if requires_extraction else "planned"
        action = _route_action(route, alias)
        command = "" if not requires_extraction else _route_command(route, alias)
        model_row = {
            "model_alias": alias,
            "route": route,
            "status": status,
            "config_path": model_config_path(alias),
            "execution_plan": {
                "steps": ["validate_source", "accept_or_extract", "write_model_manifest"],
                "requires_extraction": requires_extraction,
                "action": action,
                "command": command,
            },
        }
        models.append(model_row)
        execution_plan.append(
            {
                "model_alias": alias,
                "route": route,
                "status": status,
                "action": action,
                "command": command,
            }
        )

    manifest = {
        "schema_version": ROUTE_MANIFEST_SCHEMA_VERSION,
        "extraction_started": False,
        "dataset_matrix": [
            {"dataset_name": dataset_name, "subset": subset}
            for dataset_name, subset in DATASET_MATRIX
        ],
        "models": models,
        "execution_plan": execution_plan,
    }
    if output is not None:
        write_json_manifest(manifest, output)
    return manifest


def build_unified_full_cache_manifest(
    *,
    route_manifest: Mapping[str, object],
    model_manifests: Sequence[Mapping[str, object]],
    output: str | Path | None = None,
) -> dict[str, object]:
    """Combine per-model manifests into one ordered panel manifest."""

    required_model_aliases = _required_model_aliases()
    route_by_alias = {
        str(row["model_alias"]): str(row["route"])
        for row in _mapping_rows(route_manifest.get("models"))
        if "model_alias" in row and "route" in row
    }
    manifests_by_alias = {
        str(row["model_alias"]): dict(row)
        for row in model_manifests
        if "model_alias" in row
    }

    models: list[dict[str, object]] = []
    for alias in required_model_aliases:
        route = route_by_alias.get(alias, route_for_model(alias))
        source = manifests_by_alias.get(alias)
        if source is None:
            source = {
                "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
                "model_alias": alias,
                "route": route,
                "status": "failed_extraction",
                "total_entries": 0,
                "num_shards": 0,
                "failed_reason": "missing full-cache model manifest",
                "datasets": {},
            }
        model = dict(source)
        model.setdefault("schema_version", MODEL_MANIFEST_SCHEMA_VERSION)
        model["model_alias"] = alias
        model["route"] = str(model.get("route") or route)
        model["status"] = _normalize_full_cache_status(model.get("status"), route=model["route"])
        model.setdefault("total_entries", 0)
        model.setdefault("num_shards", 0)
        model.setdefault("failed_reason", "")
        model.setdefault("datasets", {})
        if model["status"] not in VALID_FULL_CACHE_STATUSES:
            raise ValueError(f"Unsupported full-cache status for {alias}: {model['status']}")
        models.append(model)

    by_status = Counter(str(model["status"]) for model in models)
    manifest = {
        "schema_version": UNIFIED_MANIFEST_SCHEMA_VERSION,
        "route_manifest_schema_version": route_manifest.get("schema_version"),
        "models": models,
        "aggregate_counts": {
            "total_models": len(models),
            "total_entries": sum(_safe_int(model.get("total_entries"), default=0) for model in models),
            "by_status": dict(sorted(by_status.items())),
        },
    }
    if output is not None:
        write_json_manifest(manifest, output)
    return manifest


def render_full_cache_report(unified_manifest: Mapping[str, object]) -> str:
    """Render a short Markdown status report for a unified full-cache manifest."""

    aggregate = _as_mapping(unified_manifest.get("aggregate_counts"))
    lines = [
        "# MIND Full-Cache Panel",
        "",
        f"total_models: {aggregate.get('total_models', 0)}",
        f"total_entries: {aggregate.get('total_entries', 0)}",
        "",
        "| model_alias | route | status | total_entries |",
        "| --- | --- | --- | --- |",
    ]
    for model in _mapping_rows(unified_manifest.get("models")):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model.get("model_alias", "")),
                    str(model.get("route", "")),
                    str(model.get("status", "")),
                    str(model.get("total_entries", 0)),
                ]
            )
            + " |"
        )
    by_status = _as_mapping(aggregate.get("by_status"))
    if by_status:
        lines.extend(["", "status_counts:"])
        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
    return "\n".join(lines) + "\n"


def build_model_panel_config() -> dict[str, object]:
    """Return the static Experiment 2 full-cache model panel config."""

    models = []
    for alias in _required_model_aliases():
        route = route_for_model(alias)
        route_label = "extract_main_env" if route == "extract_default_env" else route
        models.append(
            {
                "alias": alias,
                "config_path": model_config_path(alias),
                "route": route_label,
                "manifest_route": route,
            }
        )
    return {
        "schema_version": PANEL_CONFIG_SCHEMA_VERSION,
        "experiment": "mind_experiment_2",
        "name": "full-cache-model-panel",
        "output_root": "outputs/full_cache",
        "dataset_matrix": [
            {"family": dataset_name, "subset": subset}
            for dataset_name, subset in DATASET_MATRIX
        ],
        "routes": {
            "accept_existing_stage0": list(STAGE0_ACCEPT_MODEL_ALIASES),
            "accept_existing_separate_env": list(SEPARATE_ENV_ACCEPT_MODEL_ALIASES),
            "extract_separate_env": list(SEPARATE_ENV_EXTRACT_MODEL_ALIASES),
            "extract_main_env": [
                alias
                for alias in _required_model_aliases()
                if route_for_model(alias) == "extract_default_env"
            ],
        },
        "models": models,
    }


def load_model_panel_config(path: str | Path) -> dict[str, object]:
    """Load a full-cache model panel YAML file."""

    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"model panel config must be a mapping: {path}")
    return dict(payload)


def write_model_panel_config(config: Mapping[str, object], output: str | Path) -> None:
    """Write a full-cache model panel YAML file."""

    import yaml

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )


def write_json_manifest(manifest: Mapping[str, object], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )


def route_for_model(model_alias: str) -> str:
    if model_alias in STAGE0_ACCEPT_MODEL_ALIASES:
        return "accept_existing_stage0"
    if model_alias in SEPARATE_ENV_ACCEPT_MODEL_ALIASES:
        return "accept_existing_separate_env"
    if model_alias in SEPARATE_ENV_EXTRACT_MODEL_ALIASES:
        return "extract_separate_env"
    return "extract_default_env"


def model_config_path(model_alias: str) -> str:
    return "configs/models/" + model_alias.replace("-", "_").replace(".", "_") + ".yaml"


def _validate_full_cache_shard(
    shard_path: Path,
    *,
    expected_model_alias: str,
    expected_cache_origin: str,
    expected_extraction_env_name: str | None,
) -> tuple[dict[str, object], list[tuple[str, str, str, str, str]], list[QuestionRecord]]:
    import torch

    errors: list[str] = []
    keys: list[tuple[str, str, str, str, str]] = []
    sample_questions: list[QuestionRecord] = []
    sidecar_path = Path(str(shard_path) + ".json")
    sidecar, sidecar_errors = _load_sidecar(sidecar_path)
    errors.extend(sidecar_errors)

    payload: object | None = None
    load_failed = False
    try:
        payload = torch.load(shard_path, weights_only=False)
    except Exception as error:  # pragma: no cover - exact torch error is environment-specific.
        load_failed = True
        errors.append(f"torch.load failed: {error}")

    entries: list[Mapping[str, object]] = []
    if not load_failed:
        if not isinstance(payload, list):
            errors.append("payload must be a list of dicts")
        else:
            for index, item in enumerate(payload):
                if isinstance(item, Mapping):
                    entries.append(item)
                else:
                    errors.append(f"entry {index} is not a dict")

    normalized_sidecar, normalization_errors, legacy_sidecar = _normalized_sidecar_metadata(
        sidecar,
        expected_model_alias=expected_model_alias,
        expected_cache_origin=expected_cache_origin,
        expected_extraction_env_name=expected_extraction_env_name,
        num_entries=len(entries),
    )
    errors.extend(normalization_errors)

    total_layers = _optional_int(normalized_sidecar.get("total_layers"))
    selected_layers = _int_list(normalized_sidecar.get("selected_layers"))
    num_selected_layers = _optional_int(normalized_sidecar.get("num_selected_layers"))
    hidden_dim = _optional_int(normalized_sidecar.get("hidden_dim"))
    errors.extend(
        _selected_layer_errors(
            total_layers,
            selected_layers,
            num_selected_layers,
        )
    )

    observed_hidden_dim: int | None = None
    for index, entry in enumerate(entries):
        entry_errors, entry_hidden_dim, key, question = _validate_full_cache_entry(
            entry,
            index=index,
            sidecar=normalized_sidecar,
            expected_model_alias=expected_model_alias,
            selected_layers=selected_layers,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            legacy_sidecar=legacy_sidecar,
        )
        errors.extend(entry_errors)
        if entry_hidden_dim is not None:
            if observed_hidden_dim is None:
                observed_hidden_dim = entry_hidden_dim
            elif observed_hidden_dim != entry_hidden_dim:
                errors.append(
                    f"entry {index} hidden_dim={entry_hidden_dim} does not match "
                    f"observed shard hidden_dim={observed_hidden_dim}"
                )
        if key is not None:
            keys.append(key)
        if question is not None:
            sample_questions.append(question)

    if hidden_dim is not None and observed_hidden_dim is not None and hidden_dim != observed_hidden_dim:
        errors.append(
            f"sidecar hidden_dim={hidden_dim} does not match observed hidden_dim={observed_hidden_dim}"
        )

    shard = {
        "path": str(shard_path),
        "sidecar_path": str(sidecar_path),
        "status": "failed" if errors else "passed",
        "source_sidecar_format": _optional_text(sidecar.get("schema_version"))
        or _optional_text(sidecar.get("format")),
        "normalized_metadata": legacy_sidecar,
        "model_alias": _optional_text(normalized_sidecar.get("model_alias")),
        "model_family": _optional_text(normalized_sidecar.get("model_family")),
        "dataset_name": _optional_text(normalized_sidecar.get("dataset_name")),
        "source_dataset": _optional_text(normalized_sidecar.get("source_dataset")),
        "subset": _optional_text(normalized_sidecar.get("subset")),
        "split": _optional_text(normalized_sidecar.get("split")),
        "cache_origin": _optional_text(normalized_sidecar.get("cache_origin")),
        "prompt_template_id": _optional_text(normalized_sidecar.get("prompt_template_id")),
        "logit_source": _optional_text(normalized_sidecar.get("logit_source")),
        "extraction_env_name": _optional_text(normalized_sidecar.get("extraction_env_name")),
        "source_records_path": _optional_text(normalized_sidecar.get("source_records_path")),
        "num_entries": len(entries),
        "total_layers": total_layers,
        "selected_layers": selected_layers,
        "hidden_dim": observed_hidden_dim,
        "errors": errors,
    }
    return shard, keys, sample_questions


def _validate_full_cache_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    sidecar: Mapping[str, object],
    expected_model_alias: str,
    selected_layers: list[int] | None,
    total_layers: int | None,
    hidden_dim: int | None,
    legacy_sidecar: bool,
) -> tuple[list[str], int | None, tuple[str, str, str, str, str] | None, QuestionRecord | None]:
    import torch

    errors: list[str] = []
    normalized_entry = _normalized_entry_metadata(entry, sidecar=sidecar, legacy_sidecar=legacy_sidecar)
    missing_fields = _missing_entry_fields(entry, normalized_entry, legacy_sidecar=legacy_sidecar)
    if missing_fields:
        errors.append(f"entry {index} missing required fields: {', '.join(missing_fields)}")

    errors.extend(
        _entry_metadata_errors(
            normalized_entry,
            index=index,
            sidecar=sidecar,
            expected_model_alias=expected_model_alias,
            selected_layers=selected_layers,
            raw_entry=entry,
        )
    )

    layer_vectors = entry.get("layer_vectors")
    observed_hidden_dim: int | None = None
    if not isinstance(layer_vectors, torch.Tensor):
        errors.append(f"entry {index} layer_vectors must be a tensor")
    else:
        if layer_vectors.ndim != 2:
            errors.append(f"entry {index} layer_vectors.ndim must be 2, got {layer_vectors.ndim}")
        else:
            expected_layers = total_layers if total_layers is not None else (
                len(selected_layers) if selected_layers is not None else None
            )
            observed_layers = int(layer_vectors.shape[0])
            observed_hidden_dim = int(layer_vectors.shape[1])
            if expected_layers is not None and observed_layers != expected_layers:
                errors.append(
                    f"entry {index} layer_vectors.shape[0]={observed_layers} "
                    f"does not match total_layers={expected_layers}"
                )
            if hidden_dim is not None and observed_hidden_dim != hidden_dim:
                errors.append(
                    f"entry {index} layer_vectors hidden_dim={observed_hidden_dim} "
                    f"does not match sidecar hidden_dim={hidden_dim}"
                )
        if not torch.isfinite(layer_vectors).all().item():
            errors.append(f"entry {index} layer_vectors must be finite")

    logits = entry.get("first_token_logits")
    if not isinstance(logits, torch.Tensor):
        errors.append(f"entry {index} first_token_logits must be a tensor")
    elif not torch.isfinite(logits).all().item():
        errors.append(f"entry {index} first_token_logits must be finite")

    key = _entry_key(normalized_entry)
    question = _entry_question(normalized_entry, sidecar=sidecar)
    if key is None and "sample_id" in entry:
        errors.append(f"entry {index} is missing identity metadata for duplicate check")
    return errors, observed_hidden_dim, key, question


def _load_sidecar(sidecar_path: Path) -> tuple[dict[str, object], list[str]]:
    if not sidecar_path.exists():
        return {}, [f"missing sidecar metadata: {sidecar_path}"]
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {}, [f"sidecar is not valid JSON: {error}"]
    if not isinstance(payload, dict):
        return {}, ["sidecar metadata must be a JSON object"]
    return dict(payload), []


def _normalized_sidecar_metadata(
    sidecar: Mapping[str, object],
    *,
    expected_model_alias: str,
    expected_cache_origin: str,
    expected_extraction_env_name: str | None,
    num_entries: int,
) -> tuple[dict[str, object], list[str], bool]:
    errors: list[str] = []
    legacy_sidecar = _is_legacy_sidecar(sidecar)
    metadata = dict(sidecar)
    config = _as_mapping(sidecar.get("config"))
    model_defaults = _model_metadata(expected_model_alias)
    panel_defaults = _full_cache_panel_defaults()

    for field in ("dataset_name", "source_dataset", "subset", "split"):
        _derive_metadata_field(metadata, field, _nested_value(sidecar, config, field), allow=legacy_sidecar)
    if _optional_text(metadata.get("source_dataset")) is None:
        _derive_metadata_field(metadata, "source_dataset", _optional_text(metadata.get("dataset_name")), allow=legacy_sidecar)

    _derive_metadata_field(metadata, "schema_version", SHARD_SCHEMA_VERSION, allow=legacy_sidecar)
    _derive_metadata_field(metadata, "cache_origin", expected_cache_origin, allow=legacy_sidecar)
    observed_model_name = _nested_value(sidecar, config, "model_name")
    _derive_metadata_field(
        metadata,
        "model_alias",
        _nested_value(sidecar, config, "model_alias")
        or observed_model_name
        or _optional_text(model_defaults.get("alias"))
        or _optional_text(model_defaults.get("name"))
        or expected_model_alias,
        allow=legacy_sidecar or observed_model_name is not None,
    )
    _derive_metadata_field(
        metadata,
        "model_family",
        _nested_value(sidecar, config, "model_family")
        or _nested_value(sidecar, config, "family")
        or _optional_text(model_defaults.get("model_family"))
        or _optional_text(model_defaults.get("family")),
        allow=legacy_sidecar,
    )
    for field in ("cache_type", "selected_layers", "num_selected_layers", "hidden_dim", "token_index", "dtype", "num_entries"):
        _derive_metadata_field(metadata, field, _nested_value(sidecar, config, field), allow=legacy_sidecar)
    _derive_metadata_field(metadata, "total_layers", _nested_value(sidecar, config, "total_layers"), allow=legacy_sidecar)
    _derive_metadata_field(
        metadata,
        "prompt_template_id",
        _nested_value(sidecar, config, "prompt_template_id")
        or _optional_text(model_defaults.get("prompt_template_id")),
        allow=legacy_sidecar,
    )
    _derive_metadata_field(
        metadata,
        "logit_source",
        _nested_value(sidecar, config, "logit_source")
        or _optional_text(panel_defaults.get("logit_source")),
        allow=legacy_sidecar,
    )
    source_records_path = (
        _nested_value(sidecar, config, "source_records_path")
        or _nested_value(sidecar, config, "records_path")
    )
    if source_records_path is None and legacy_sidecar:
        source_records_path = _default_source_records_path(
            _optional_text(metadata.get("dataset_name")),
            _optional_text(metadata.get("subset")),
        )
    _derive_metadata_field(metadata, "source_records_path", source_records_path, allow=legacy_sidecar)

    observed_aliases = {
        field: value
        for field, value in {
            "model_alias": _nested_value(sidecar, config, "model_alias"),
            "model_name": _nested_value(sidecar, config, "model_name"),
        }.items()
        if value is not None
    }
    for field, observed in observed_aliases.items():
        if observed != expected_model_alias:
            errors.append(f"sidecar {field}={observed} does not match expected model_alias={expected_model_alias}")

    comparisons = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "cache_origin": expected_cache_origin,
        "model_alias": expected_model_alias,
    }
    for field, expected in comparisons.items():
        observed = _optional_text(metadata.get(field))
        raw_observed = _optional_text(sidecar.get(field))
        if raw_observed is not None and raw_observed != expected:
            errors.append(f"sidecar {field}={raw_observed} does not match expected {field}={expected}")
        elif observed is not None and observed != expected:
            errors.append(f"sidecar {field}={observed} does not match expected {field}={expected}")

    sidecar_entries = _optional_int(metadata.get("num_entries"))
    if sidecar_entries is not None and sidecar_entries != num_entries:
        errors.append(f"sidecar num_entries={metadata.get('num_entries')} does not match payload length {num_entries}")

    if expected_cache_origin == "separate_env":
        observed_env = _optional_text(sidecar.get("extraction_env_name"))
        if observed_env is not None and expected_extraction_env_name is not None and observed_env != expected_extraction_env_name:
            errors.append(
                "sidecar extraction_env_name="
                f"{observed_env} does not match expected extraction_env_name={expected_extraction_env_name}"
            )
        elif observed_env is None and expected_extraction_env_name is not None:
            if legacy_sidecar:
                _derive_metadata_field(metadata, "extraction_env_name", expected_extraction_env_name, allow=True)
            else:
                errors.append("sidecar missing required separate-env metadata: extraction_env_name")

    missing = [
        field
        for field in REQUIRED_SIDECAR_FIELDS
        if not _has_nonempty_field(metadata, field)
    ]
    if missing:
        errors.append("sidecar missing required metadata: " + ", ".join(missing))
    return metadata, errors, legacy_sidecar


def _derive_metadata_field(metadata: dict[str, object], field: str, value: object | None, *, allow: bool) -> None:
    if not allow or _has_nonempty_field(metadata, field) or value is None:
        return
    metadata[field] = value


def _is_legacy_sidecar(sidecar: Mapping[str, object]) -> bool:
    return any(
        key in sidecar
        for key in ("format", "stage", "model_name", "metadata_version", "records_path")
    ) or isinstance(sidecar.get("config"), Mapping)


def _nested_value(sidecar: Mapping[str, object], config: Mapping[str, object], field: str) -> object | None:
    if sidecar.get(field) is not None:
        return sidecar.get(field)
    if config.get(field) is not None:
        return config.get(field)
    return None


def _selected_layer_errors(
    total_layers: int | None,
    selected_layers: list[int] | None,
    num_selected_layers: int | None,
) -> list[str]:
    errors: list[str] = []
    if selected_layers is None:
        return errors
    if num_selected_layers is not None and num_selected_layers != len(selected_layers):
        errors.append(
            f"num_selected_layers={num_selected_layers} does not match selected_layers length {len(selected_layers)}"
        )
    if not selected_layers:
        errors.append("selected_layers must not be empty")
        return errors
    expected_contiguous = list(range(selected_layers[0], selected_layers[0] + len(selected_layers)))
    if selected_layers != expected_contiguous:
        errors.append(f"selected_layers must be contiguous, got {selected_layers}")
    if total_layers is not None:
        expected_full = list(range(total_layers))
        if selected_layers != expected_full:
            errors.append(
                "selected_layers must cover every full-cache layer in order: "
                f"expected {expected_full}, got {selected_layers}"
            )
        if num_selected_layers is not None and num_selected_layers != total_layers:
            errors.append(
                f"num_selected_layers={num_selected_layers} does not match total_layers={total_layers}"
            )
    return errors


def _entry_metadata_errors(
    entry: Mapping[str, object],
    *,
    index: int,
    sidecar: Mapping[str, object],
    expected_model_alias: str,
    selected_layers: list[int] | None,
    raw_entry: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    entry_model_alias = _optional_text(entry.get("model_alias"))
    if entry_model_alias is not None and entry_model_alias != expected_model_alias:
        errors.append(
            f"entry {index} model_alias={entry_model_alias} "
            f"does not match expected model_alias={expected_model_alias}"
        )
    entry_model_name = _optional_text(raw_entry.get("model_name"))
    if entry_model_name is not None and entry_model_name != expected_model_alias:
        errors.append(
            f"entry {index} model_name={entry_model_name} "
            f"does not match expected model_alias={expected_model_alias}"
        )
    for field in (
        "model_alias",
        "model_family",
        "dataset_name",
        "source_dataset",
        "subset",
        "split",
        "prompt_template_id",
        "logit_source",
    ):
        expected = _optional_text(sidecar.get(field))
        observed = _optional_text(entry.get(field))
        if expected is not None and observed is not None and observed != expected:
            errors.append(f"entry {index} {field}={observed} does not match sidecar {field}={expected}")
    expected_token_index = _optional_int(sidecar.get("token_index"))
    observed_token_index = _optional_int(entry.get("token_index"))
    if expected_token_index is not None and observed_token_index is not None and observed_token_index != expected_token_index:
        errors.append(
            f"entry {index} token_index={observed_token_index} "
            f"does not match sidecar token_index={expected_token_index}"
        )
    entry_layers = _int_list(entry.get("selected_layers"))
    if selected_layers is not None and entry_layers is not None and entry_layers != selected_layers:
        errors.append(
            f"entry {index} selected_layers {entry_layers} does not match sidecar selected_layers {selected_layers}"
        )
    return errors


def _normalized_entry_metadata(
    entry: Mapping[str, object],
    *,
    sidecar: Mapping[str, object],
    legacy_sidecar: bool,
) -> dict[str, object]:
    normalized = dict(entry)
    allow_derived = legacy_sidecar
    if _optional_text(normalized.get("model_alias")) is None:
        model_name = _optional_text(normalized.get("model_name"))
        _derive_metadata_field(
            normalized,
            "model_alias",
            model_name or _optional_text(sidecar.get("model_alias")),
            allow=model_name is not None,
        )
    for field in DERIVABLE_LEGACY_ENTRY_FIELDS:
        _derive_metadata_field(normalized, field, sidecar.get(field), allow=allow_derived)
    return normalized


def _missing_entry_fields(
    raw_entry: Mapping[str, object],
    normalized_entry: Mapping[str, object],
    *,
    legacy_sidecar: bool,
) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_ENTRY_FIELDS:
        if field == "model_alias":
            if not _has_nonempty_field(raw_entry, "model_alias") and not _has_nonempty_field(raw_entry, "model_name"):
                missing.append("model_alias/model_name")
            continue
        source = normalized_entry if legacy_sidecar and field in DERIVABLE_LEGACY_ENTRY_FIELDS else raw_entry
        if field == "parsed_answer":
            if field not in source:
                missing.append(field)
        elif not _has_nonempty_field(source, field):
            missing.append(field)
    return missing


def _acceptance_report(
    *,
    status: str,
    route: str,
    model_alias: str,
    cache_origin: str,
    source_cache_root: Path,
    output_root: Path,
    validation: Mapping[str, object],
    extraction_env_name: str | None = None,
) -> dict[str, object]:
    manifest_path = output_root / model_alias / "full_cache_acceptance_manifest.json"
    report: dict[str, object] = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "model_alias": model_alias,
        "route": route,
        "status": status,
        "cache_origin": cache_origin,
        "source_cache_root": str(source_cache_root),
        "copied_tensors": False,
        "source_tensors_mutated": False,
        "modern_sidecar_policy": "report_source_metadata_as_observed_without_mutating_source_tensors",
        "total_entries": int(validation.get("total_entries") or 0),
        "num_shards": int(validation.get("num_shards") or 0),
        "datasets": validation.get("datasets", {}),
        "validation_status": validation.get("status"),
        "validation_errors": validation.get("errors", []),
        "validation": dict(validation),
        "manifest_path": str(manifest_path),
    }
    if extraction_env_name is not None:
        report["extraction_env_name"] = extraction_env_name
    return report


def _route_action(route: str, model_alias: str) -> str:
    if route == "accept_existing_stage0":
        return f"accept existing Stage 0 cache for {model_alias}"
    if route == "accept_existing_separate_env":
        return f"accept existing separate-env cache for {model_alias}"
    if route == "extract_separate_env":
        return f"extract full cache in separate environment for {model_alias}"
    return f"extract full cache in main environment for {model_alias}"


def _route_command(route: str, model_alias: str) -> str:
    if route == "extract_separate_env":
        env_name = "model-specific-separate-env"
    else:
        env_name = "mind-py311"
    return f"<pending script integration> {model_alias} --env {env_name}"


def _require_exact_panel_order(aliases: Sequence[str]) -> None:
    expected = list(_required_model_aliases())
    if list(aliases) != expected:
        raise ValueError(f"model aliases must match final panel order: expected={expected!r} actual={list(aliases)!r}")
    if len(set(aliases)) != len(aliases):
        raise ValueError("model aliases must be unique")


def _entry_key(entry: Mapping[str, object]) -> tuple[str, str, str, str, str] | None:
    values = (
        _optional_text(entry.get("model_alias")),
        _optional_text(entry.get("dataset_name")),
        _optional_text(entry.get("subset")),
        _optional_text(entry.get("split")),
        _optional_text(entry.get("sample_id")),
    )
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _entry_question(entry: Mapping[str, object], *, sidecar: Mapping[str, object]) -> QuestionRecord | None:
    dataset_name = _optional_text(entry.get("source_dataset")) or _optional_text(entry.get("dataset_name"))
    subset = _optional_text(entry.get("subset"))
    sample_id = _optional_text(entry.get("sample_id"))
    question = _optional_text(entry.get("question"))
    if dataset_name is None or subset is None or sample_id is None or question is None:
        return None
    return dataset_name, subset, sample_id, question, _optional_text(sidecar.get("source_records_path"))


def _dataset_key(dataset_name: object | None, subset: object | None) -> str | None:
    dataset_text = _optional_text(dataset_name)
    subset_text = _optional_text(subset)
    if dataset_text is None or subset_text is None:
        return None
    return f"{dataset_text}/{subset_text}"


def _question_preservation_report(question_records: Sequence[QuestionRecord]) -> dict[str, object]:
    errors: list[str] = []
    questions_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    examples_by_key: dict[tuple[str, str, str], str] = {}
    source_cache: dict[Path, tuple[dict[str, str], list[str]]] = {}
    source_paths_used: set[str] = set()
    checked = 0

    for dataset_name, subset, sample_id, question, _source_path in question_records:
        key = (dataset_name, subset, sample_id)
        questions_by_key[key].add(question)
        examples_by_key.setdefault(key, question)

    for key, questions in sorted(questions_by_key.items()):
        if len(questions) > 1:
            dataset_name, subset, sample_id = key
            errors.append(
                f"{dataset_name}/{subset} sample_id {sample_id} has multiple question texts"
            )

    for dataset_name, subset, sample_id, question, source_path in question_records:
        resolved_path = _resolve_source_records_path(dataset_name, subset, source_path)
        if resolved_path is None:
            continue
        source_paths_used.add(str(resolved_path))
        if resolved_path not in source_cache:
            source_cache[resolved_path] = _load_source_record_questions(resolved_path)
        source_questions, load_errors = source_cache[resolved_path]
        for error in load_errors:
            errors.append(error)
        if load_errors:
            continue
        checked += 1
        expected = source_questions.get(sample_id)
        if expected is None:
            errors.append(
                f"{dataset_name}/{subset} sample_id {sample_id} is missing from source records {resolved_path}"
            )
        elif expected != question:
            errors.append(
                f"{dataset_name}/{subset} sample_id {sample_id} question does not match source records"
            )

    return {
        "status": "failed" if errors else "passed",
        "num_sample_ids": len(questions_by_key),
        "num_records_checked": checked,
        "source_paths": sorted(source_paths_used),
        "examples": [
            {
                "dataset_name": dataset_name,
                "subset": subset,
                "sample_id": sample_id,
                "question": question,
            }
            for (dataset_name, subset, sample_id), question in sorted(examples_by_key.items())[:5]
        ],
        "errors": errors,
    }


def _resolve_source_records_path(dataset_name: str, subset: str, source_path: str | None) -> Path | None:
    if source_path is not None:
        path = Path(source_path)
        if path.is_absolute():
            return path
        return (_repo_root() / path).resolve()
    return None


def _default_source_records_path(dataset_name: str | None, subset: str | None) -> str | None:
    if dataset_name is None or subset is None:
        return None
    candidates = [
        Path("outputs/stage0/normalized") / dataset_name / f"{subset}.jsonl",
        Path("data") / dataset_name / f"{subset}.jsonl",
    ]
    for candidate in candidates:
        if (_repo_root() / candidate).is_file():
            return str(candidate)
    return None


def _load_source_record_questions(path: Path) -> tuple[dict[str, str], list[str]]:
    if not path.is_file():
        return {}, [f"source records path does not exist: {path}"]
    questions: dict[str, str] = {}
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as error:
                    errors.append(f"{path}:{line_number}: source record is not valid JSON: {error}")
                    continue
                if not isinstance(row, Mapping):
                    errors.append(f"{path}:{line_number}: source record must be a JSON object")
                    continue
                sample_id = _optional_text(row.get("sample_id"))
                question = _optional_text(row.get("question"))
                if sample_id is None or question is None:
                    errors.append(f"{path}:{line_number}: source record missing sample_id or question")
                    continue
                if sample_id in questions and questions[sample_id] != question:
                    errors.append(f"{path}:{line_number}: duplicate source sample_id {sample_id} has different question")
                    continue
                questions[sample_id] = question
    except OSError as error:
        return {}, [f"failed to read source records path {path}: {error}"]
    return questions, errors


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _normalize_full_cache_status(status: object, *, route: object) -> str:
    value = str(status or "").strip()
    route_text = str(route or "").strip()
    if value == "extracted_default_env":
        return "extracted_main_env"
    if value == "needs_extraction":
        if route_text in {"accept_existing_separate_env", "extract_separate_env"}:
            return "needs_extraction_separate_env"
        return "failed_extraction"
    return value or "failed_extraction"


def _required_model_aliases() -> tuple[str, ...]:
    from mind.models.registry import REQUIRED_MODEL_ALIASES

    return tuple(REQUIRED_MODEL_ALIASES)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_MODEL_METADATA_CACHE: dict[str, dict[str, object]] | None = None
_FULL_CACHE_PANEL_DEFAULTS_CACHE: dict[str, object] | None = None


def _model_metadata(model_alias: str) -> Mapping[str, object]:
    global _MODEL_METADATA_CACHE
    if _MODEL_METADATA_CACHE is None:
        _MODEL_METADATA_CACHE = _load_model_metadata()
    return _MODEL_METADATA_CACHE.get(model_alias, {"alias": model_alias, "name": model_alias})


def _load_model_metadata() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    asset_config = _read_yaml_mapping(_repo_root() / "configs/assets/model_assets.yaml")
    for row in _mapping_rows(asset_config.get("models")):
        alias = _optional_text(row.get("alias")) or _optional_text(row.get("name"))
        if alias is None:
            continue
        rows[alias] = dict(row)

    for alias in set(_required_model_aliases()) | set(rows):
        row = rows.setdefault(alias, {"alias": alias})
        config_path = _optional_text(row.get("model_config_path")) or model_config_path(alias)
        model_config = _read_yaml_mapping(_repo_root() / config_path)
        for key, value in model_config.items():
            row.setdefault(key, value)
        row.setdefault("alias", alias)
        row.setdefault("name", alias)
    return rows


def _full_cache_panel_defaults() -> Mapping[str, object]:
    global _FULL_CACHE_PANEL_DEFAULTS_CACHE
    if _FULL_CACHE_PANEL_DEFAULTS_CACHE is None:
        panel = _read_yaml_mapping(_repo_root() / "configs/full_cache/model_panel.yaml")
        _FULL_CACHE_PANEL_DEFAULTS_CACHE = {
            "logit_source": panel.get("logit_source"),
            "token_index": panel.get("token_index"),
            "dtype": panel.get("dtype"),
        }
    return _FULL_CACHE_PANEL_DEFAULTS_CACHE


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _has_nonempty_field(mapping: Mapping[str, object], field: str) -> bool:
    if field not in mapping:
        return False
    value = mapping[field]
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _safe_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int_list(value: object | None) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            result.append(int(item))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return result
