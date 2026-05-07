"""v2 Stage 0 cache validation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

REQUIRED_ENTRY_FIELDS = (
    "sample_id",
    "image_id",
    "image_path",
    "question",
    "label",
    "object_name",
    "answer_text",
    "parsed_answer",
    "first_token_logits",
)

REQUIRED_SIDECAR_FIELDS = (
    "stage",
    "cache_type",
    "model_name",
    "model_id",
    "model_family",
    "dataset_name",
    "subset",
    "split",
    "total_layers",
    "selected_layers",
    "num_selected_layers",
    "hidden_dim",
    "token_index",
    "max_new_tokens",
    "dtype",
    "num_entries",
    "script",
    "git_commit",
    "created_at_utc",
    "records_path",
    "image_root",
)

DuplicateKey = tuple[str, str, str]


class CacheValidationError(ValueError):
    """Raised when a v2 Stage 0 cache fails validation."""

    def __init__(self, manifest: Mapping[str, object]) -> None:
        self.manifest = manifest
        errors = manifest.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, str) and errors:
            message = "; ".join(str(error) for error in errors[:3])
        else:
            message = "Stage 0 cache validation failed"
        super().__init__(message)


def validate_stage0_cache(
    cache_root: Path | str,
    *,
    output: Path | str | None = None,
    dataset_name: str | None = None,
    split: str | None = None,
    model_name: str | None = None,
    raise_on_error: bool = True,
) -> dict[str, object]:
    """Validate all ``*.pt`` shards under a v2 Stage 0 cache root.

    The validator accepts only the list-of-dicts Stage 0 shard format. It
    writes a manifest when ``output`` is supplied, then raises by default if
    any validation error was found.
    """

    manifest = build_stage0_cache_manifest(
        cache_root,
        dataset_name=dataset_name,
        split=split,
        model_name=model_name,
    )
    if output is not None:
        write_cache_manifest(manifest, output)
    if raise_on_error and manifest["status"] != "passed":
        raise CacheValidationError(manifest)
    return manifest


def build_stage0_cache_manifest(
    cache_root: Path | str,
    *,
    dataset_name: str | None = None,
    split: str | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    cache_root_path = Path(cache_root)
    shards: list[dict[str, object]] = []
    errors: list[str] = []
    duplicate_counter: Counter[DuplicateKey] = Counter()
    total_entries = 0

    if not cache_root_path.exists():
        errors.append(f"Cache root does not exist: {cache_root_path}")
    elif not cache_root_path.is_dir():
        errors.append(f"Cache root is not a directory: {cache_root_path}")
    else:
        shard_paths = sorted(path for path in cache_root_path.rglob("*.pt") if path.is_file())
        if not shard_paths:
            errors.append(f"No .pt cache shards found under {cache_root_path}")
        for shard_path in shard_paths:
            shard, keys = _validate_shard(
                shard_path,
                expected_dataset_name=dataset_name,
                expected_split=split,
                expected_model_name=model_name,
            )
            shards.append(shard)
            total_entries += int(shard["num_entries"] or 0)
            duplicate_counter.update(keys)

    duplicate_keys = sorted(key for key, count in duplicate_counter.items() if count > 1)
    for key in duplicate_keys:
        errors.append(
            "duplicate cache key "
            f"dataset_name={key[0]} split={key[1]} sample_id={key[2]}"
        )

    for shard in shards:
        for error in shard["errors"]:  # type: ignore[index]
            errors.append(f"{shard['path']}: {error}")

    manifest: dict[str, object] = {
        "status": "failed" if errors else "passed",
        "cache_root": str(cache_root_path),
        "shards": shards,
        "total_entries": total_entries,
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "errors": errors,
    }
    if dataset_name is not None:
        manifest["dataset_name"] = dataset_name
    if split is not None:
        manifest["split"] = split
    if model_name is not None:
        manifest["model_name"] = model_name
    return manifest


def write_cache_manifest(manifest: Mapping[str, object], output: Path | str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _validate_shard(
    shard_path: Path,
    *,
    expected_dataset_name: str | None,
    expected_split: str | None,
    expected_model_name: str | None,
) -> tuple[dict[str, object], list[DuplicateKey]]:
    import torch

    errors: list[str] = []
    keys: list[DuplicateKey] = []
    sidecar_path = Path(str(shard_path) + ".json")
    sidecar, sidecar_errors = _load_sidecar(sidecar_path)
    errors.extend(sidecar_errors)
    errors.extend(
        _expected_metadata_errors(
            sidecar,
            dataset_name=expected_dataset_name,
            split=expected_split,
            model_name=expected_model_name,
        )
    )

    payload = None
    load_failed = False
    try:
        payload = torch.load(shard_path, weights_only=False)
    except Exception as error:  # pragma: no cover - exact torch error is environment-specific.
        load_failed = True
        errors.append(f"torch.load failed: {error}")

    entries: list[Mapping[str, object]] = []
    if not load_failed:
        if not isinstance(payload, list):
            errors.append("payload must be a list of dicts for Stage 0 validation")
        else:
            for index, item in enumerate(payload):
                if isinstance(item, Mapping):
                    entries.append(item)
                else:
                    errors.append(f"entry {index} is not a dict")

    total_layers = _optional_int(sidecar.get("total_layers"))
    selected_layers = _int_list(sidecar.get("selected_layers"))
    num_selected_layers = _optional_int(sidecar.get("num_selected_layers"))
    sidecar_hidden_dim = _optional_int(sidecar.get("hidden_dim"))

    errors.extend(_layer_metadata_errors(total_layers, selected_layers, num_selected_layers))
    if "num_entries" in sidecar and _optional_int(sidecar.get("num_entries")) != len(entries):
        errors.append(
            f"sidecar num_entries={sidecar.get('num_entries')} does not match payload length {len(entries)}"
        )

    observed_hidden_dim: int | None = None
    for index, entry in enumerate(entries):
        entry_errors, hidden_dim, key = _validate_entry(
            entry,
            index=index,
            sidecar=sidecar,
            selected_layers=selected_layers,
            num_selected_layers=num_selected_layers,
        )
        errors.extend(entry_errors)
        if hidden_dim is not None:
            if observed_hidden_dim is None:
                observed_hidden_dim = hidden_dim
            elif observed_hidden_dim != hidden_dim:
                errors.append(
                    f"entry {index} hidden dim {hidden_dim} does not match shard hidden dim "
                    f"{observed_hidden_dim}"
                )
        if key is not None:
            keys.append(key)

    if (
        sidecar_hidden_dim is not None
        and observed_hidden_dim is not None
        and sidecar_hidden_dim != observed_hidden_dim
    ):
        errors.append(
            f"sidecar hidden_dim={sidecar_hidden_dim} does not match observed hidden dim "
            f"{observed_hidden_dim}"
        )

    shard = {
        "path": str(shard_path),
        "sidecar_path": str(sidecar_path),
        "num_entries": len(entries),
        "hidden_dim": observed_hidden_dim,
        "total_layers": total_layers,
        "selected_layers": selected_layers,
        "errors": errors,
    }
    return shard, keys


def _load_sidecar(sidecar_path: Path) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    if not sidecar_path.exists():
        return {}, [f"missing sidecar metadata: {sidecar_path}"]
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {}, [f"sidecar is not valid JSON: {error}"]
    if not isinstance(payload, dict):
        return {}, ["sidecar metadata must be a JSON object"]

    sidecar = dict(payload)
    missing = [field for field in REQUIRED_SIDECAR_FIELDS if field not in sidecar]
    if missing:
        errors.append("sidecar missing required metadata: " + ", ".join(missing))
    return sidecar, errors


def _expected_metadata_errors(
    sidecar: Mapping[str, object],
    *,
    dataset_name: str | None,
    split: str | None,
    model_name: str | None,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "dataset_name": dataset_name,
        "split": split,
        "model_name": model_name,
    }
    for key, value in expected.items():
        if value is not None and _optional_text(sidecar.get(key)) != value:
            errors.append(f"sidecar {key} does not match expected {key}: {value}")
    return errors


def _layer_metadata_errors(
    total_layers: int | None,
    selected_layers: list[int] | None,
    num_selected_layers: int | None,
) -> list[str]:
    errors: list[str] = []
    if total_layers is None:
        return errors
    expected = list(range(total_layers))
    if selected_layers is not None and selected_layers != expected:
        errors.append(
            "selected_layers must contain every layer in order for Stage 0 full-layer cache: "
            f"expected {expected}, got {selected_layers}"
        )
    if num_selected_layers is not None and num_selected_layers != total_layers:
        errors.append(
            "num_selected_layers must equal total_layers for Stage 0 full-layer cache: "
            f"num_selected_layers={num_selected_layers} total_layers={total_layers}"
        )
    return errors


def _validate_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    sidecar: Mapping[str, object],
    selected_layers: list[int] | None,
    num_selected_layers: int | None,
) -> tuple[list[str], int | None, DuplicateKey | None]:
    import torch

    errors: list[str] = []
    missing_fields = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
    if missing_fields:
        errors.append(f"entry {index} missing required fields: {', '.join(missing_fields)}")

    layer_vectors = entry.get("layer_vectors")
    hidden_dim: int | None = None
    if not isinstance(layer_vectors, torch.Tensor):
        errors.append(f"entry {index} missing tensor layer_vectors")
    else:
        if layer_vectors.ndim != 2:
            errors.append(f"entry {index} layer_vectors.ndim must be 2, got {layer_vectors.ndim}")
        else:
            hidden_dim = int(layer_vectors.shape[1])
            expected_layers = _entry_num_selected_layers(
                entry,
                selected_layers=selected_layers,
                num_selected_layers=num_selected_layers,
            )
            if expected_layers is not None and int(layer_vectors.shape[0]) != expected_layers:
                errors.append(
                    f"entry {index} layer_vectors.shape[0]={int(layer_vectors.shape[0])} "
                    f"does not match num_selected_layers={expected_layers}"
                )
        if not torch.isfinite(layer_vectors).all().item():
            errors.append(f"entry {index} layer_vectors must be finite")

    entry_selected_layers = _int_list(entry.get("selected_layers"))
    if (
        selected_layers is not None
        and entry_selected_layers is not None
        and entry_selected_layers != selected_layers
    ):
        errors.append(
            f"entry {index} selected_layers {entry_selected_layers} does not match sidecar "
            f"selected_layers {selected_layers}"
        )

    logits = entry.get("first_token_logits")
    if "first_token_logits" in entry:
        if isinstance(logits, torch.Tensor):
            if not torch.isfinite(logits).all().item():
                errors.append(f"entry {index} first_token_logits must be finite")
        else:
            errors.append(f"entry {index} first_token_logits must be a tensor")

    key = _duplicate_key(entry, sidecar)
    if key is None and "sample_id" in entry:
        errors.append(f"entry {index} is missing dataset_name or split metadata for duplicate check")
    return errors, hidden_dim, key


def _entry_num_selected_layers(
    entry: Mapping[str, object],
    *,
    selected_layers: list[int] | None,
    num_selected_layers: int | None,
) -> int | None:
    if num_selected_layers is not None:
        return num_selected_layers
    if selected_layers is not None:
        return len(selected_layers)
    entry_selected_layers = _int_list(entry.get("selected_layers"))
    if entry_selected_layers is not None:
        return len(entry_selected_layers)
    return None


def _duplicate_key(
    entry: Mapping[str, object],
    sidecar: Mapping[str, object],
) -> DuplicateKey | None:
    dataset_name = _optional_text(entry.get("dataset_name")) or _optional_text(
        sidecar.get("dataset_name")
    )
    split = _optional_text(entry.get("split")) or _optional_text(sidecar.get("split"))
    sample_id = _optional_text(entry.get("sample_id"))
    if dataset_name is None or split is None or sample_id is None:
        return None
    return (dataset_name, split, sample_id)


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
