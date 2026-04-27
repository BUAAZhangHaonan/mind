#!/usr/bin/env python3
"""Build query-local kNN bank features on CUDA."""

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
from mind.geometry.local_neighborhood import compute_local_features, select_local_references  # noqa: E402
from mind.utils import output_root_lock  # noqa: E402


FEATURE_CURVES = (
    "local_pca_residual",
    "centroid_angular_distance",
    "mean_angular_distance",
    "std_angular_distance",
)


def validate_cuda_visible_devices() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != "0":
        raise ValueError(f"CUDA_VISIBLE_DEVICES must be '0' for this script, found {visible_devices!r}.")


def resolve_cuda_device(device_name: str | torch.device = "cuda") -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("query-local feature construction requires CUDA.")
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


def load_pooled_layer(
    pooled_bank_root: Path,
    model_name: str,
    layer_index: int,
    *,
    device: torch.device,
    cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    layer_index = int(layer_index)
    if layer_index not in cache:
        path = pooled_bank_root / model_name / f"layer-{layer_index:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing pooled bank layer: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{path} must contain a rank-2 tensor.")
        cache[layer_index] = tensor.to(device=device, dtype=torch.float32)
    return cache[layer_index]


def _slope(values: torch.Tensor) -> torch.Tensor:
    x = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    return (x_centered * y_centered).sum() / x_centered.square().sum().clamp_min(1e-8)


def _curve_features(name: str, values: torch.Tensor, *, primary: bool) -> dict[str, float]:
    if values.device.type != "cuda":
        raise ValueError("curve values must be CUDA tensors.")
    prefix = "raw_drift" if primary else f"{name}_drift"
    cpu_values = values.detach().cpu()
    features = {
        f"{prefix}_{index}": float(cpu_values[index])
        for index in range(int(cpu_values.numel()))
    }
    summary_prefix = "cal" if primary else name
    features[f"{summary_prefix}_mean_drift"] = float(values.mean().detach().cpu())
    features[f"{summary_prefix}_max_drift"] = float(values.max().detach().cpu())
    features[f"{summary_prefix}_final_drift"] = float(values[-1].detach().cpu())
    features[f"{summary_prefix}_drift_slope"] = float(_slope(values).detach().cpu())
    features[f"{summary_prefix}_drift_variance"] = float(values.var(unbiased=False).detach().cpu())
    if primary:
        features["raw_max_drift"] = float(values.max().detach().cpu())
        features["raw_mean_drift"] = float(values.mean().detach().cpu())
        features["raw_peak_layer_index"] = float(values.argmax().detach().cpu())
    return features


def build_query_local_feature_frame(
    *,
    cache_entries: Sequence[dict[str, object]],
    pooled_bank_root: Path,
    model_name: str,
    k_neighbors: int = 30,
    reference_chunk_size: int = 16_384,
    device: str | torch.device = "cuda",
    limit_rows: int | None = None,
    label_overrides: Path | None = None,
) -> pd.DataFrame:
    device = resolve_cuda_device(device)
    entries = list(cache_entries[:limit_rows] if limit_rows is not None else cache_entries)
    pooled_cache: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for entry in entries:
            selected_layers = [int(layer) for layer in entry["selected_layers"]]
            layer_vectors = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32)
            curves = {name: [] for name in FEATURE_CURVES}
            for offset, layer_index in enumerate(selected_layers):
                query = layer_vectors[offset].to(device=device, dtype=torch.float32)
                pooled_layer = load_pooled_layer(
                    pooled_bank_root,
                    model_name,
                    layer_index,
                    device=device,
                    cache=pooled_cache,
                )
                _indices, local_references = select_local_references(
                    query,
                    pooled_layer,
                    k=k_neighbors,
                    reference_chunk_size=reference_chunk_size,
                )
                local_features = compute_local_features(query, local_references)
                for name in FEATURE_CURVES:
                    curves[name].append(local_features[name])

            row = metadata_row_from_entry(entry)
            for name in FEATURE_CURVES:
                values = torch.stack(curves[name]).to(device=device, dtype=torch.float32)
                row.update(_curve_features(name, values, primary=(name == "local_pca_residual")))
            rows.append(row)
    return apply_label_overrides(pd.DataFrame(rows), label_overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--pooled-bank-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, help="Accepted for orchestration; rows are streamed.")
    parser.add_argument("--reference-chunk-size", type=int, default=16_384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--label-overrides", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    validate_cuda_visible_devices()
    args = build_parser().parse_args(argv)
    cache_entries = load_cache_entries(args.cache_path)
    frame = build_query_local_feature_frame(
        cache_entries=cache_entries,
        pooled_bank_root=args.pooled_bank_root,
        model_name=args.model_name,
        k_neighbors=args.k_neighbors,
        reference_chunk_size=args.reference_chunk_size,
        device=args.device,
        limit_rows=args.limit_rows,
        label_overrides=args.label_overrides,
    )
    with output_root_lock(args.output_path.parent, command=f"query_local:{args.model_name}"):
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output_path, index=False)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
