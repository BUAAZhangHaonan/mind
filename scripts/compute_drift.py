#!/usr/bin/env python3
"""Compute drift features from cached hidden-state shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.drift import (
    build_drift_features,
    calibrate_drift_curve,
    compute_drift_curves_batched,
)
from mind.evaluation import compute_object_hallucination_label
from mind.manifolds import resolve_reference_scope_key
from mind.utils import output_root_lock


def build_feature_output_path(
    *,
    output_root: Path,
    experiment_name: str,
    split: str,
) -> Path:
    return output_root / experiment_name / f"{split}.parquet"


def load_reference_bank(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, torch.Tensor]] = {}
    model_root = reference_root / model_name
    for layer_path in model_root.glob("*/layer-*.pt"):
        object_name = layer_path.parent.name
        if bank_scope in {"object", "shuffled_object"} and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        layer_index = int(layer_path.stem.split("-")[-1])
        bank.setdefault(object_name, {})[layer_index] = torch.load(layer_path, weights_only=True)
    return bank


def load_reference_stats(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, dict[str, float]]]:
    stats_map: dict[str, dict[int, dict[str, float]]] = {}
    model_root = reference_root / model_name
    for stats_path in model_root.glob("*/stats.pt"):
        object_name = stats_path.parent.name
        if bank_scope in {"object", "shuffled_object"} and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        payload = torch.load(stats_path, weights_only=True)
        stats_map[object_name] = {
            int(layer_index): {str(key): float(value) for key, value in layer_stats.items()}
            for layer_index, layer_stats in payload.items()
        }
    return stats_map


def load_cache_entries(cache_path: Path) -> list[dict[str, object]]:
    if cache_path.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=True))
        return entries
    return list(torch.load(cache_path, weights_only=True))


def _format_missing_reference_coverage(missing_entries: list[dict[str, object]]) -> str:
    preview = ", ".join(
        f"{entry['sample_id']}[{entry['reason']}]"
        for entry in missing_entries[:5]
    )
    if len(missing_entries) > 5:
        preview += f", ... (+{len(missing_entries) - 5} more)"
    return f"Missing reference coverage for {len(missing_entries)} cache entries: {preview}"


def build_feature_frame(
    *,
    cache_entries: list[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_stats: dict[str, dict[int, dict[str, float]]],
    bank_scope: str = "object",
    batch_size: int = 32,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    missing_entries: list[dict[str, object]] = []
    prepared_entries: list[dict[str, object]] = []
    grouped_indices: dict[tuple[tuple[int, ...], str], list[int]] = {}
    for entry_index, entry in enumerate(cache_entries):
        selected_layers = [int(layer) for layer in entry["selected_layers"]]
        object_name = str(entry["object_name"])
        bank_key = resolve_reference_scope_key(object_name, bank_scope)
        if bank_key not in reference_bank:
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": f"missing bank:{bank_key}"}
            )
            continue
        if bank_key not in reference_stats:
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": f"missing stats:{bank_key}"}
            )
            continue
        missing_bank_layers = [
            layer_index for layer_index in selected_layers if layer_index not in reference_bank[bank_key]
        ]
        missing_stats_layers = [
            layer_index for layer_index in selected_layers if layer_index not in reference_stats[bank_key]
        ]
        if missing_bank_layers or missing_stats_layers:
            reason_parts = []
            if missing_bank_layers:
                reason_parts.append(f"missing bank layers:{missing_bank_layers}")
            if missing_stats_layers:
                reason_parts.append(f"missing stats layers:{missing_stats_layers}")
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": "; ".join(reason_parts)}
            )
            continue
        prepared_entries.append(
            {
                "entry_index": entry_index,
                "entry": entry,
                "selected_layers": selected_layers,
                "object_name": object_name,
                "bank_key": bank_key,
            }
        )
        grouped_indices.setdefault((tuple(selected_layers), bank_key), []).append(len(prepared_entries) - 1)

    if missing_entries:
        raise ValueError(_format_missing_reference_coverage(missing_entries))

    raw_curves: list[np.ndarray | None] = [None for _ in prepared_entries]
    for (selected_layers_key, bank_key), group_indices in grouped_indices.items():
        selected_layers = list(selected_layers_key)
        for start in range(0, len(group_indices), batch_size):
            batch_indices = group_indices[start : start + batch_size]
            layer_vectors_batch = torch.stack(
                [
                    prepared_entries[prepared_index]["entry"]["layer_vectors"]
                    for prepared_index in batch_indices
                ],
                dim=0,
            )
            batch_curves = compute_drift_curves_batched(
                layer_vectors_batch=layer_vectors_batch,
                selected_layers=selected_layers,
                object_name=str(prepared_entries[batch_indices[0]]["object_name"]),
                reference_bank=reference_bank,
                bank_scope=bank_scope,
                bank_key=bank_key,
                batch_size=batch_size,
            )
            for batch_offset, prepared_index in enumerate(batch_indices):
                raw_curves[prepared_index] = batch_curves[batch_offset]

    rows: list[dict[str, object]] = []
    for prepared_index, prepared in enumerate(prepared_entries):
        entry = prepared["entry"]
        selected_layers = prepared["selected_layers"]
        object_name = str(prepared["object_name"])
        bank_key = str(prepared["bank_key"])
        raw_curve = raw_curves[prepared_index]
        if raw_curve is None:
            raise RuntimeError(f"Missing drift curve for {entry['sample_id']}")
        calibrated_curve = calibrate_drift_curve(
            raw_curve,
            selected_layers=selected_layers,
            layer_stats=reference_stats[bank_key],
        )
        features = build_drift_features(
            raw_curve=raw_curve,
            calibrated_curve=calibrated_curve,
        )
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "image_id": int(entry.get("image_id", -1)),
                "ground_truth_label": int(entry["label"]),
                "answer_label": -1 if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
                "label": compute_object_hallucination_label(
                    ground_truth_label=int(entry["label"]),
                    answer_label=(
                        None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
                    ),
                ),
                "subset": entry["subset"],
                "object_name": object_name,
                **features,
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--bank-scope", choices=["object", "shared", "shuffled_object"], default="object")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = build_feature_output_path(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        split=args.split,
    )
    if args.cache_path is None or args.reference_root is None or not args.model_name:
        print(output_path)
        return 0

    cache_entries = load_cache_entries(args.cache_path)
    reference_bank = load_reference_bank(args.reference_root, args.model_name, bank_scope=args.bank_scope)
    reference_stats = load_reference_stats(args.reference_root, args.model_name, bank_scope=args.bank_scope)
    with output_root_lock(
        output_path.parent,
        command=f"compute_drift:{args.experiment_name}:{args.split}:{args.bank_scope}",
    ):
        frame = build_feature_frame(
            cache_entries=cache_entries,
            reference_bank=reference_bank,
            reference_stats=reference_stats,
            bank_scope=args.bank_scope,
            batch_size=args.batch_size,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
