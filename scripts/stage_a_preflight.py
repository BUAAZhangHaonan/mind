#!/usr/bin/env python3
"""Validate Stage 0 outputs before Stage A starts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.trajectory.stage_a_population import iter_cache_shards


REQUIRED_STAGE0_MODELS = ("qwen3-vl-8b", "internvl3.5-8b")
REQUIRED_STAGE0_DATASET_SUBSETS = {
    "pope": ("popular", "random", "adversarial"),
    "repope": ("popular", "random", "adversarial"),
    "dash-b": ("all",),
}

MatrixKey = tuple[str, str, str]


def run_preflight(
    *,
    stage0_root: Path | str = Path("outputs/stage0"),
    output_root: Path | str = Path("outputs/stageA"),
    required_models: Sequence[str] = REQUIRED_STAGE0_MODELS,
    required_dataset_subsets: Mapping[str, Sequence[str]] = REQUIRED_STAGE0_DATASET_SUBSETS,
    write_outputs: bool = True,
) -> dict[str, object]:
    """Run the Stage A preflight gate and write the acceptance artifact."""

    stage0_root = Path(stage0_root)
    output_root = Path(output_root)
    issues: list[str] = []
    expected_keys = _expected_matrix_keys(required_models, required_dataset_subsets)

    summary_path = stage0_root / "manifests" / "stage0_summary.json"
    cache_manifest_path = stage0_root / "manifests" / "cache_manifest.json"
    cache_label_balance_path = stage0_root / "audit" / "cache_label_balance.csv"

    summary = _read_json_object(summary_path, issues, "stage0_summary.json")
    cache_manifest = _read_json_object(cache_manifest_path, issues, "cache_manifest.json")

    _validate_stage0_summary(summary, expected_keys, issues)
    _validate_cache_manifest(cache_manifest, issues)
    _validate_cache_label_balance(cache_label_balance_path, expected_keys, issues)
    _validate_cache_directories(stage0_root, expected_keys, issues)
    shard_validation = _validate_required_shard_payloads(stage0_root, cache_manifest, expected_keys, issues)

    streamed_counts = _stream_cache_counts(
        stage0_root,
        cache_manifest,
        expected_keys=expected_keys,
        issues=issues,
    )
    _validate_streamed_counts(summary, streamed_counts, expected_keys, issues)

    result: dict[str, object] = {
        "stage": "stage_a_preflight",
        "status": "failed" if issues else "passed",
        "stage0_root": str(stage0_root),
        "output_root": str(output_root),
        "required_stage0_matrix": [_matrix_row(key) for key in expected_keys],
        "shard_validation": shard_validation,
        "streamed_cache_counts": {
            _format_key(key): streamed_counts.get(key, 0)
            for key in expected_keys
        },
        "issues": issues,
    }
    if write_outputs:
        result["outputs"] = _write_preflight_outputs(result, output_root)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageA"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validation without writing Stage A preflight artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_preflight(
        stage0_root=args.stage0_root,
        output_root=args.output_root,
        write_outputs=not args.dry_run,
    )
    _print_summary(result, dry_run=args.dry_run)
    return 0 if result["status"] == "passed" else 2


def _read_json_object(path: Path, issues: list[str], label: str) -> dict[str, object]:
    if not path.exists():
        issues.append(f"missing {label}: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        issues.append(f"{label} is not valid JSON: {error}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{label} must be a JSON object: {path}")
        return {}
    return payload


def _validate_stage0_summary(
    summary: Mapping[str, object],
    expected_keys: Sequence[MatrixKey],
    issues: list[str],
) -> None:
    if not summary:
        return
    status = str(summary.get("status", "")).strip()
    if status != "passed":
        issues.append(f"stage0_summary status is not passed: {status or '<missing>'}")
    if not str(summary.get("git_commit", "")).strip():
        issues.append("stage0_summary git_commit is missing")

    blocking = summary.get("blocking_issues", [])
    if isinstance(blocking, Sequence) and not isinstance(blocking, (str, bytes)) and blocking:
        issues.append("stage0_summary blocking_issues present: " + "; ".join(map(str, blocking[:3])))

    required_rows = _matrix_rows_by_key(summary.get("required_cache_matrix", []))
    completed_rows = _matrix_rows_by_key(summary.get("completed_cache_matrix", []))
    missing_rows = _matrix_rows_by_key(summary.get("missing_cache_matrix", []))
    raw_missing = summary.get("missing_cache_matrix", [])
    if isinstance(raw_missing, Sequence) and not isinstance(raw_missing, (str, bytes)) and raw_missing:
        issues.append("stage0_summary missing_cache_matrix is not empty")

    missing_required = [key for key in expected_keys if key not in required_rows]
    if missing_required:
        issues.append("required_cache_matrix missing: " + _format_keys(missing_required))

    missing_completed = [key for key in expected_keys if key not in completed_rows]
    if missing_completed:
        issues.append("completed_cache_matrix missing: " + _format_keys(missing_completed))

    unexpected_missing = [key for key in expected_keys if key in missing_rows]
    if unexpected_missing:
        issues.append("missing_cache_matrix contains required entries: " + _format_keys(unexpected_missing))

    for key in expected_keys:
        required_row = required_rows.get(key)
        completed_row = completed_rows.get(key)
        if required_row is None or completed_row is None:
            continue
        expected = _optional_int(required_row.get("expected_num_records"))
        observed = _optional_int(completed_row.get("cache_num_entries"))
        if expected is not None and observed is not None and observed != expected:
            issues.append(
                f"cache matrix count mismatch for {_format_key(key)}: expected {expected} got {observed}"
            )


def _validate_cache_manifest(cache_manifest: Mapping[str, object], issues: list[str]) -> None:
    if not cache_manifest:
        return
    status = str(cache_manifest.get("status", "")).strip()
    if status != "passed":
        issues.append(f"cache_manifest status is not passed: {status or '<missing>'}")
    errors = cache_manifest.get("errors", [])
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) and errors:
        issues.append("cache_manifest errors present: " + "; ".join(map(str, errors[:3])))


def _validate_cache_label_balance(
    path: Path,
    expected_keys: Sequence[MatrixKey],
    issues: list[str],
) -> None:
    if not path.exists():
        issues.append(f"missing cache_label_balance.csv: {path}")
        return
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        issues.append(f"could not read cache_label_balance.csv: {error}")
        return
    observed = {
        key
        for row in rows
        if (key := _matrix_key(row)) is not None
    }
    missing = [key for key in expected_keys if key not in observed]
    if missing:
        issues.append("cache_label_balance.csv missing rows: " + _format_keys(missing))


def _validate_cache_directories(
    stage0_root: Path,
    expected_keys: Sequence[MatrixKey],
    issues: list[str],
) -> None:
    for model_name, dataset_name, subset in expected_keys:
        cache_dir = stage0_root / "cache" / model_name / dataset_name / subset
        if not cache_dir.is_dir():
            issues.append(f"missing cache directory: {cache_dir}")


def _stream_cache_counts(
    stage0_root: Path,
    cache_manifest: Mapping[str, object],
    *,
    expected_keys: Sequence[MatrixKey],
    issues: list[str],
) -> Counter[MatrixKey]:
    expected = set(expected_keys)
    counts: Counter[MatrixKey] = Counter()
    if not cache_manifest:
        return counts

    import torch

    try:
        shards = list(iter_cache_shards(stage0_root, cache_manifest))
    except ValueError as error:
        issues.append(str(error))
        return counts

    for shard_path, shard in shards:
        if not shard_path.exists():
            issues.append(f"missing cache shard: {shard_path}")
            continue
        try:
            payload = torch.load(shard_path, weights_only=False)
        except Exception as error:  # pragma: no cover - exact torch error varies.
            issues.append(f"torch.load failed for {shard_path}: {error}")
            continue
        for entry in _iter_cache_payload_entries(payload):
            row = dict(entry)
            _fill_entry_metadata(row, shard)
            key = _matrix_key(row)
            if key in expected:
                counts[key] += 1
    return counts


def _validate_required_shard_payloads(
    stage0_root: Path,
    cache_manifest: Mapping[str, object],
    expected_keys: Sequence[MatrixKey],
    issues: list[str],
) -> dict[str, object]:
    import torch

    shards_by_key: dict[MatrixKey, list[tuple[Path, Mapping[str, object]]]] = {
        key: [] for key in expected_keys
    }
    try:
        shard_pairs = list(iter_cache_shards(stage0_root, cache_manifest))
    except ValueError as error:
        issues.append(str(error))
        return {"status": "failed", "validated": {}}

    for shard_path, shard in shard_pairs:
        key = _matrix_key(shard)
        if key in shards_by_key:
            shards_by_key[key].append((shard_path, shard))

    hidden_dims: dict[tuple[str, str], int] = {}
    validated: dict[str, dict[str, object]] = {}
    for key in expected_keys:
        candidates = shards_by_key.get(key, [])
        if not candidates:
            issues.append(f"no shard listed for required cache matrix row: {_format_key(key)}")
            continue
        shard_path, shard = candidates[0]
        shard_issues: list[str] = []
        try:
            payload = torch.load(shard_path, weights_only=False)
        except Exception as error:  # pragma: no cover - exact torch error varies.
            issues.append(f"torch.load failed for required shard {shard_path}: {error}")
            shard_issues.append(f"torch.load failed: {error}")
            validated[_format_key(key)] = {"path": str(shard_path), "issues": shard_issues}
            continue

        if not isinstance(payload, list):
            shard_issues.append("payload is not a list")
        elif not payload:
            shard_issues.append("payload is empty")
        else:
            total_layers = _optional_int(shard.get("total_layers"))
            selected_layers = _int_list(shard.get("selected_layers"))
            sidecar_hidden_dim = _optional_int(shard.get("hidden_dim"))
            shard_issues.extend(_layer_metadata_issues(total_layers, selected_layers))
            for index, entry in enumerate(payload):
                if not isinstance(entry, Mapping):
                    shard_issues.append(f"entry {index} is not a dict")
                    continue
                entry_issues, observed_hidden_dim = _validate_cache_entry(
                    entry,
                    index=index,
                    total_layers=total_layers,
                    sidecar_selected_layers=selected_layers,
                )
                shard_issues.extend(entry_issues)
                if observed_hidden_dim is not None:
                    model_dataset = (key[0], key[1])
                    expected_hidden_dim = hidden_dims.get(model_dataset)
                    if expected_hidden_dim is None:
                        hidden_dims[model_dataset] = observed_hidden_dim
                    elif expected_hidden_dim != observed_hidden_dim:
                        shard_issues.append(
                            f"hidden_dim {observed_hidden_dim} inconsistent for "
                            f"{model_dataset[0]}/{model_dataset[1]} expected {expected_hidden_dim}"
                        )
                    if sidecar_hidden_dim is not None and sidecar_hidden_dim != observed_hidden_dim:
                        shard_issues.append(
                            f"sidecar hidden_dim={sidecar_hidden_dim} does not match entry hidden_dim="
                            f"{observed_hidden_dim}"
                        )

        if shard_issues:
            issues.extend(f"{shard_path}: {issue}" for issue in shard_issues)
        validated[_format_key(key)] = {
            "path": str(shard_path),
            "num_entries_inspected": len(payload) if isinstance(payload, list) else 0,
            "issues": shard_issues,
        }
    return {
        "status": "failed" if any(item["issues"] for item in validated.values()) else "passed",
        "validated": validated,
        "hidden_dims": {
            f"{model_name}/{dataset_name}": hidden_dim
            for (model_name, dataset_name), hidden_dim in sorted(hidden_dims.items())
        },
    }


def _validate_streamed_counts(
    summary: Mapping[str, object],
    streamed_counts: Counter[MatrixKey],
    expected_keys: Sequence[MatrixKey],
    issues: list[str],
) -> None:
    required_rows = _matrix_rows_by_key(summary.get("required_cache_matrix", []))
    for key in expected_keys:
        required_row = required_rows.get(key)
        if required_row is None:
            continue
        expected = _optional_int(required_row.get("expected_num_records"))
        if expected is None:
            continue
        observed = streamed_counts.get(key, 0)
        if observed != expected:
            issues.append(
                f"streamed cache count mismatch for {_format_key(key)}: expected {expected} got {observed}"
            )


def _write_preflight_outputs(result: Mapping[str, object], output_root: Path) -> dict[str, str]:
    audit_path = output_root / "audit" / "stage0_acceptance.json"
    manifest_path = output_root / "manifests" / "stage_a_preflight.json"
    outputs = {
        "stage0_acceptance": str(audit_path),
        "stage_a_preflight": str(manifest_path),
    }
    payload = dict(result)
    payload["outputs"] = outputs
    _write_json(audit_path, payload)
    _write_json(manifest_path, payload)
    return outputs


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _expected_matrix_keys(
    required_models: Sequence[str],
    required_dataset_subsets: Mapping[str, Sequence[str]],
) -> list[MatrixKey]:
    return [
        (model_name, dataset_name, subset)
        for model_name in required_models
        for dataset_name, subsets in required_dataset_subsets.items()
        for subset in subsets
    ]


def _matrix_rows_by_key(rows: object) -> dict[MatrixKey, Mapping[str, object]]:
    result: dict[MatrixKey, Mapping[str, object]] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return result
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = _matrix_key(row)
        if key is not None:
            result[key] = row
    return result


def _matrix_key(row: Mapping[str, object]) -> MatrixKey | None:
    model_name = _text(row.get("model_name"))
    dataset_name = _text(row.get("dataset_name") or row.get("source_dataset"))
    subset = _text(row.get("subset") or row.get("split"))
    if model_name is None or dataset_name is None or subset is None:
        return None
    return (model_name, dataset_name, subset)


def _matrix_row(key: MatrixKey) -> dict[str, str]:
    model_name, dataset_name, subset = key
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "subset": subset,
    }


def _iter_cache_payload_entries(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if not isinstance(payload, Mapping):
        return
    for key in ("entries", "records", "samples"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield item
            return
    for sample_id, value in payload.items():
        if isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("sample_id", sample_id)
            yield row


def _fill_entry_metadata(row: dict[str, object], shard: Mapping[str, object]) -> None:
    for field in ("model_name", "dataset_name", "source_dataset", "subset", "split"):
        value = shard.get(field)
        if value is not None and row.get(field) in (None, ""):
            row[field] = value


def _optional_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int_list(value: object | None) -> list[int] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    if not isinstance(value, Sequence):
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


def _layer_metadata_issues(total_layers: int | None, selected_layers: list[int] | None) -> list[str]:
    issues: list[str] = []
    if total_layers is None:
        issues.append("total_layers is missing")
        return issues
    expected = list(range(total_layers))
    if selected_layers != expected:
        issues.append(
            "selected_layers must be contiguous full-layer indices "
            f"0 through {total_layers - 1}"
        )
    return issues


def _validate_cache_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    total_layers: int | None,
    sidecar_selected_layers: list[int] | None,
) -> tuple[list[str], int | None]:
    import torch

    issues: list[str] = []
    required_fields = (
        "layer_vectors",
        "selected_layers",
        "first_token_logits",
        "parsed_answer",
        "label",
        "object_name",
        "sample_id",
        "image_id",
        "image_path",
        "question",
    )
    for field in required_fields:
        if field not in entry:
            issues.append(f"entry {index} missing {field}")

    hidden_dim: int | None = None
    layer_vectors = entry.get("layer_vectors")
    if not isinstance(layer_vectors, torch.Tensor):
        issues.append(f"entry {index} layer_vectors is not a tensor")
    elif layer_vectors.ndim != 2:
        issues.append(f"entry {index} layer_vectors.ndim is {layer_vectors.ndim}, expected 2")
    else:
        hidden_dim = int(layer_vectors.shape[1])
        if total_layers is not None and int(layer_vectors.shape[0]) != total_layers:
            issues.append(
                f"entry {index} layer_vectors.shape[0]={int(layer_vectors.shape[0])} "
                f"does not equal total_layers={total_layers}"
            )

    entry_selected_layers = _int_list(entry.get("selected_layers"))
    selected_layers = entry_selected_layers if entry_selected_layers is not None else sidecar_selected_layers
    issues.extend(_layer_metadata_issues(total_layers, selected_layers))
    return issues, hidden_dim
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_key(key: MatrixKey) -> str:
    return "/".join(key)


def _format_keys(keys: Sequence[MatrixKey]) -> str:
    return ", ".join(_format_key(key) for key in keys)


def _print_summary(result: Mapping[str, object], *, dry_run: bool) -> None:
    print("Stage A preflight complete")
    print(f"status={result['status']}")
    print(f"stage0_root={result['stage0_root']}")
    print(f"output_root={result['output_root']}")
    print(f"issues={len(result['issues'])}")  # type: ignore[arg-type]
    print(f"dry_run={str(dry_run).lower()}")


if __name__ == "__main__":
    raise SystemExit(main())
