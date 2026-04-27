#!/usr/bin/env python3
"""Build flattened hidden-state linear-probe features with CUDA reshaping."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd
import torch


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.evaluation.baselines import load_cache_entries  # noqa: E402
from mind.evaluation.metrics import compute_object_hallucination_label  # noqa: E402
from mind.utils import output_root_lock  # noqa: E402


def validate_cuda_visible_devices() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != "0":
        raise ValueError(f"CUDA_VISIBLE_DEVICES must be '0' for this script, found {visible_devices!r}.")


def resolve_cuda_device(device_name: str | torch.device = "cuda") -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("linear-probe feature construction requires CUDA.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")
    if device.index not in (None, 0):
        raise ValueError("Use GPU 0 only.")
    return torch.device("cuda:0")


def metadata_row_from_entry(entry: dict[str, object]) -> dict[str, object]:
    answer_label = None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
    return {
        "sample_id": str(entry["sample_id"]),
        "image_id": int(entry.get("image_id", -1)),
        "ground_truth_label": int(entry["label"]),
        "answer_label": -1 if answer_label is None else int(answer_label),
        "label": compute_object_hallucination_label(
            ground_truth_label=int(entry["label"]),
            answer_label=answer_label,
        ),
        "subset": str(entry["subset"]),
        "object_name": str(entry["object_name"]),
    }


def apply_label_overrides(frame: pd.DataFrame, overrides_path: Path | None) -> pd.DataFrame:
    if overrides_path is None:
        return frame
    if not overrides_path.exists():
        raise FileNotFoundError(f"Label override file does not exist: {overrides_path}")
    overrides = pd.read_parquet(overrides_path) if overrides_path.suffix == ".parquet" else pd.read_csv(overrides_path)
    required = ["sample_id", "ground_truth_label", "answer_label", "label"]
    missing = [column for column in required if column not in overrides.columns]
    if missing:
        raise ValueError(f"Label override file {overrides_path} is missing columns: {missing}")
    override_frame = overrides[required].copy()
    override_frame["sample_id"] = override_frame["sample_id"].astype(str)
    merged = frame.drop(columns=["ground_truth_label", "answer_label", "label"], errors="ignore").merge(
        override_frame,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged["label"].isna().any():
        missing_ids = merged.loc[merged["label"].isna(), "sample_id"].head().tolist()
        raise ValueError(f"Label overrides do not cover all feature rows. Missing examples: {missing_ids}")
    for column in ["ground_truth_label", "answer_label", "label"]:
        merged[column] = merged[column].astype(int)
    metadata_order = [
        column
        for column in ["sample_id", "image_id", "ground_truth_label", "answer_label", "label", "subset", "object_name"]
        if column in merged.columns
    ]
    feature_columns = [column for column in merged.columns if column not in metadata_order]
    return merged[metadata_order + feature_columns]


def build_linear_probe_feature_frame(
    *,
    cache_entries: Sequence[dict[str, object]],
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    limit_rows: int | None = None,
    label_overrides: Path | None = None,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = resolve_cuda_device(device)
    entries = list(cache_entries[:limit_rows] if limit_rows is not None else cache_entries)
    if not entries:
        return pd.DataFrame()

    first = torch.as_tensor(entries[0]["layer_vectors"], dtype=torch.float32)
    flat_dim = int(first.numel())
    rows: list[dict[str, object]] = []
    feature_frames: list[pd.DataFrame] = []

    with torch.no_grad():
        for start in range(0, len(entries), int(batch_size)):
            batch_entries = entries[start : start + int(batch_size)]
            layer_batch = torch.stack(
                [torch.as_tensor(entry["layer_vectors"], dtype=torch.float32) for entry in batch_entries],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            flattened = layer_batch.reshape(layer_batch.shape[0], -1)
            if int(flattened.shape[1]) != flat_dim:
                raise ValueError("All linear-probe entries must share the same flattened hidden-state size.")
            feature_frames.append(
                pd.DataFrame(
                    flattened.detach().cpu().numpy(),
                    columns=[f"hidden_{index}" for index in range(flat_dim)],
                )
            )
            rows.extend(metadata_row_from_entry(entry) for entry in batch_entries)

    feature_frame = pd.concat(feature_frames, ignore_index=True)
    frame = pd.concat([pd.DataFrame(rows).reset_index(drop=True), feature_frame], axis=1)
    return apply_label_overrides(frame, label_overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--label-overrides", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    validate_cuda_visible_devices()
    args = build_parser().parse_args(argv)
    cache_entries = load_cache_entries(args.cache_path)
    frame = build_linear_probe_feature_frame(
        cache_entries=cache_entries,
        batch_size=args.batch_size,
        device=args.device,
        limit_rows=args.limit_rows,
        label_overrides=args.label_overrides,
    )
    with output_root_lock(args.output_path.parent, command=f"linear_probe:{args.model_name}"):
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output_path, index=False)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
