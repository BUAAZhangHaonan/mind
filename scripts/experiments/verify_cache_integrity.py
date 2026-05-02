#!/usr/bin/env python3
"""Verify prefill cache shards and optional sidecar metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.types import resolve_torch_dtype


SHARD_PATTERN = re.compile(r"^shard-\d{5}\.pt$")


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_shards(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [
        path
        for path in sorted(root.rglob("shard-*.pt"))
        if SHARD_PATTERN.match(path.name)
    ]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def prefill_cache_sidecar_path(path: str | Path) -> Path:
    shard_path = Path(path)
    return shard_path.with_suffix(shard_path.suffix + ".json")


def _sidecar_int_field(
    sidecar: dict[str, object],
    field: str,
    *,
    default: int | None,
    sidecar_path: Path,
    errors: list[str],
) -> int | None:
    raw_value = sidecar.get(field, default)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        errors.append(f"{sidecar_path}: {field} is not an integer: {raw_value!r}")
        return None


def verify_sidecar(path: Path, *, record_count: int, errors: list[str]) -> None:
    sidecar_path = prefill_cache_sidecar_path(path)
    if not sidecar_path.exists():
        return
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{sidecar_path}: unreadable sidecar: {exc}")
        return
    if not isinstance(sidecar, dict):
        errors.append(f"{sidecar_path}: sidecar payload is not a JSON object")
        return

    num_entries = _sidecar_int_field(
        sidecar,
        "num_entries",
        default=-1,
        sidecar_path=sidecar_path,
        errors=errors,
    )
    if num_entries is not None and num_entries != record_count:
        errors.append(
            f"{sidecar_path}: num_entries {sidecar.get('num_entries')} "
            f"does not match shard count {record_count}"
        )
    actual_file_bytes = _sidecar_int_field(
        sidecar,
        "actual_file_bytes",
        default=None,
        sidecar_path=sidecar_path,
        errors=errors,
    )
    if actual_file_bytes is not None and actual_file_bytes != path.stat().st_size:
        errors.append(f"{sidecar_path}: actual_file_bytes does not match {path}")
    expected_hash = sidecar.get("content_sha256")
    if expected_hash and expected_hash != content_sha256(path):
        errors.append(f"{sidecar_path}: content_sha256 does not match {path}")
    config = sidecar.get("config")
    expected_config_hash = sidecar.get("config_hash")
    if isinstance(config, dict) and expected_config_hash:
        actual = config_hash(config)
        if actual != expected_config_hash:
            errors.append(f"{sidecar_path}: config_hash does not match config payload")


def verify_entry(
    entry: object,
    *,
    shard_path: Path,
    entry_index: int,
    expected_dtype: torch.dtype,
    selected_layer_count: int | None,
    errors: list[str],
) -> None:
    prefix = f"{shard_path} entry {entry_index}"
    if not isinstance(entry, dict):
        errors.append(f"{prefix}: entry is not a dict")
        return
    layer_vectors = entry.get("layer_vectors")
    if not isinstance(layer_vectors, torch.Tensor):
        errors.append(f"{prefix}: missing tensor field layer_vectors")
        return
    if layer_vectors.ndim != 2:
        errors.append(f"{prefix}: layer_vectors shape {tuple(layer_vectors.shape)} is not 2D")
    if layer_vectors.dtype != expected_dtype:
        errors.append(
            f"{prefix}: layer_vectors dtype {_dtype_name(layer_vectors.dtype)} "
            f"!= {_dtype_name(expected_dtype)}"
        )
    selected_layers = entry.get("selected_layers")
    if not isinstance(selected_layers, list):
        errors.append(f"{prefix}: selected_layers is not a list")
    else:
        if layer_vectors.ndim >= 1 and len(selected_layers) != int(layer_vectors.shape[0]):
            errors.append(
                f"{prefix}: selected_layers length {len(selected_layers)} "
                f"!= layer_vectors rows {int(layer_vectors.shape[0])}"
            )
        if selected_layer_count is not None and len(selected_layers) != selected_layer_count:
            errors.append(
                f"{prefix}: selected_layers length {len(selected_layers)} "
                f"!= expected {selected_layer_count}"
            )

def verify_shard(
    path: Path,
    *,
    expected_dtype: torch.dtype,
    selected_layer_count: int | None,
    require_sidecar: bool,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    try:
        payload = torch.load(path, weights_only=False)
    except Exception as exc:
        return 0, [f"{path}: unreadable shard: {exc}"]

    if not isinstance(payload, list):
        return 0, [f"{path}: shard payload is not a list"]
    if not payload:
        errors.append(f"{path}: shard contains no records")
    for index, entry in enumerate(payload):
        verify_entry(
            entry,
            shard_path=path,
            entry_index=index,
            expected_dtype=expected_dtype,
            selected_layer_count=selected_layer_count,
            errors=errors,
        )
    if require_sidecar and not prefill_cache_sidecar_path(path).exists():
        errors.append(f"{path}: missing required sidecar {prefill_cache_sidecar_path(path)}")
    verify_sidecar(path, record_count=len(payload), errors=errors)
    return len(payload), errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--selected-layers", type=int, default=None)
    parser.add_argument("--require-sidecars", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shards = iter_shards(args.cache_root)
    if not shards:
        print(f"No shards found under {args.cache_root}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    total_records = 0
    expected_dtype = resolve_torch_dtype(args.dtype)
    for shard in shards:
        count, errors = verify_shard(
            shard,
            expected_dtype=expected_dtype,
            selected_layer_count=args.selected_layers,
            require_sidecar=args.require_sidecars,
        )
        total_records += count
        all_errors.extend(errors)

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        print(
            f"FAILED: {len(all_errors)} error(s), {len(shards)} shard(s), {total_records} record(s)",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {len(shards)} shard(s), {total_records} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
