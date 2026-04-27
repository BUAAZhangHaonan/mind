#!/usr/bin/env python3
"""Build CUDA drift features from cached hidden states and reference banks."""

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

from mind.evaluation.baselines import (  # noqa: E402
    load_cache_entries,
    load_reference_bank,
    load_reference_stats,
)
from mind.evaluation.metrics import compute_object_hallucination_label  # noqa: E402
from mind.manifolds import resolve_reference_scope_key  # noqa: E402
from mind.utils import output_root_lock  # noqa: E402


METADATA_COLUMNS = (
    "sample_id",
    "image_id",
    "ground_truth_label",
    "answer_label",
    "label",
    "subset",
    "object_name",
)
CURVE_TYPES = ("local_pca", "no_manifold")
DEFAULT_REFERENCE_CHUNK_SIZE = 16_384


def validate_cuda_visible_devices() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != "0":
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be set to '0' for this GPU round script "
            f"(found {visible_devices!r})."
        )


def resolve_cuda_device(device_name: str | torch.device = "cuda") -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("CUDA drift feature construction requires a CUDA device.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA drift feature construction requires torch.cuda.is_available().")
    if device.index not in (None, 0):
        raise ValueError("Use GPU 0 only for this experiment.")
    return torch.device("cuda:0")


def build_feature_output_path(
    *,
    output_path: Path | None,
    output_root: Path | None,
    experiment_name: str | None,
    split: str | None,
) -> Path:
    if output_path is not None:
        return output_path
    if output_root is None or not experiment_name or not split:
        raise ValueError("--output-path or all of --output-root, --experiment-name, and --split are required.")
    return output_root / experiment_name / f"{split}.parquet"


def _format_missing_reference_coverage(missing_entries: list[dict[str, object]]) -> str:
    preview = ", ".join(
        f"{entry['sample_id']}[{entry['reason']}]"
        for entry in missing_entries[:5]
    )
    if len(missing_entries) > 5:
        preview += f", ... (+{len(missing_entries) - 5} more)"
    return f"Missing reference coverage for {len(missing_entries)} cache entries: {preview}"


def _metadata_row_from_entry(entry: dict[str, object]) -> dict[str, object]:
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


def _validate_cuda_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA.")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point.")


def _euclidean_topk_gpu(
    *,
    queries: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    reference_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_cuda_matrix("queries", queries)
    _validate_cuda_matrix("reference", reference)
    if queries.shape[1] != reference.shape[1]:
        raise ValueError("queries and reference must have the same feature dimension.")
    if k < 1:
        raise ValueError("k must be positive.")
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be positive.")
    neighbor_count = min(int(k), int(reference.shape[0]))
    if neighbor_count < 1:
        raise ValueError("reference must contain at least one row.")

    query_norm = queries.square().sum(dim=1, keepdim=True)
    best_values: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, int(reference.shape[0]), int(reference_chunk_size)):
        stop = min(start + int(reference_chunk_size), int(reference.shape[0]))
        reference_chunk = reference[start:stop]
        reference_norm = reference_chunk.square().sum(dim=1).unsqueeze(0)
        distance_squares = (
            query_norm
            + reference_norm
            - 2.0 * (queries @ reference_chunk.T)
        ).clamp_min(0.0)
        local_k = min(neighbor_count, int(reference_chunk.shape[0]))
        local_values, local_indices = torch.topk(
            distance_squares,
            k=local_k,
            largest=False,
            sorted=True,
            dim=1,
        )
        local_indices = local_indices + start
        if best_values is None:
            best_values = local_values
            best_indices = local_indices
            continue
        merged_values = torch.cat((best_values, local_values), dim=1)
        merged_indices = torch.cat((best_indices, local_indices), dim=1)
        best_values, merge_indices = torch.topk(
            merged_values,
            k=neighbor_count,
            largest=False,
            sorted=True,
            dim=1,
        )
        best_indices = torch.gather(merged_indices, dim=1, index=merge_indices)

    if best_values is None or best_indices is None:
        raise RuntimeError("failed to compute top-k neighbors.")
    return best_values, best_indices


def _neighbor_residuals_gpu(
    *,
    queries: torch.Tensor,
    reference: torch.Tensor,
    k_neighbors: int,
    reference_chunk_size: int,
) -> torch.Tensor:
    distance_squares, _ = _euclidean_topk_gpu(
        queries=queries,
        reference=reference,
        k=k_neighbors,
        reference_chunk_size=reference_chunk_size,
    )
    distances = distance_squares.clamp_min(0.0).sqrt()
    return distances[:, 0] / distances.mean(dim=1).clamp_min(1e-8)


def _local_pca_residuals_gpu(
    *,
    queries: torch.Tensor,
    reference: torch.Tensor,
    k_neighbors: int,
    variance_threshold: float,
    max_components: int,
    reference_chunk_size: int,
) -> torch.Tensor:
    _, neighbor_indices = _euclidean_topk_gpu(
        queries=queries,
        reference=reference,
        k=k_neighbors,
        reference_chunk_size=reference_chunk_size,
    )
    neighbors = reference[neighbor_indices]
    means = neighbors.mean(dim=1)
    centered_neighbors = neighbors - means.unsqueeze(1)
    centered_queries = queries - means
    radii = torch.norm(centered_neighbors, dim=2).mean(dim=1).clamp_min(1e-8)

    if neighbors.shape[1] < 2:
        return torch.norm(centered_queries, dim=1) / radii

    gram = centered_neighbors @ centered_neighbors.transpose(1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    variances = torch.flip(eigenvalues.clamp_min(0.0), dims=(1,))
    left_vectors = torch.flip(eigenvectors, dims=(2,))
    variance_ratio = variances / variances.sum(dim=1, keepdim=True).clamp_min(1e-8)
    cumulative = torch.cumsum(variance_ratio, dim=1)
    component_counts = (cumulative < float(variance_threshold)).sum(dim=1) + 1
    component_counts = component_counts.clamp(max=min(int(max_components), int(left_vectors.shape[2])))

    neighbor_query_dot = torch.bmm(centered_neighbors, centered_queries.unsqueeze(2)).squeeze(2)
    singular_values = variances.sqrt().clamp_min(1e-8)
    coefficients = (
        torch.bmm(left_vectors.transpose(1, 2), neighbor_query_dot.unsqueeze(2)).squeeze(2)
        / singular_values
    )
    component_mask = (
        torch.arange(coefficients.shape[1], device=queries.device).unsqueeze(0)
        < component_counts.unsqueeze(1)
    )
    projection_norm_square = coefficients.square().masked_fill(~component_mask, 0.0).sum(dim=1)
    query_norm_square = centered_queries.square().sum(dim=1)
    residual_norm = (query_norm_square - projection_norm_square).clamp_min(0.0).sqrt()
    return residual_norm / radii


def _calibration_tensors(
    *,
    selected_layers: Sequence[int],
    layer_stats: dict[int, dict[str, float]],
    mean_key: str,
    std_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[float] = []
    stds: list[float] = []
    for layer_index in selected_layers:
        stats = layer_stats[int(layer_index)]
        means.append(float(stats[mean_key]))
        stds.append(max(float(stats[std_key]), 1e-8))
    return (
        torch.tensor(means, device=device, dtype=torch.float32).unsqueeze(0),
        torch.tensor(stds, device=device, dtype=torch.float32).unsqueeze(0),
    )


def calibrate_curves_gpu(
    *,
    raw_curves: torch.Tensor,
    selected_layers: Sequence[int],
    layer_stats: dict[int, dict[str, float]],
    curve_type: str,
) -> torch.Tensor:
    if raw_curves.device.type != "cuda":
        raise ValueError("raw_curves must be on CUDA.")
    if curve_type == "local_pca":
        mean_key = "residual_mean"
        std_key = "residual_std"
    elif curve_type == "no_manifold":
        mean_key = "neighbor_residual_mean"
        std_key = "neighbor_residual_std"
    else:
        raise ValueError(f"Unsupported curve_type: {curve_type}")
    means, stds = _calibration_tensors(
        selected_layers=selected_layers,
        layer_stats=layer_stats,
        mean_key=mean_key,
        std_key=std_key,
        device=raw_curves.device,
    )
    return (raw_curves - means) / stds


def _calibrated_summary_gpu(calibrated_curves: torch.Tensor) -> dict[str, torch.Tensor]:
    if calibrated_curves.device.type != "cuda":
        raise ValueError("calibrated_curves must be on CUDA.")
    layer_count = int(calibrated_curves.shape[1])
    x = torch.arange(layer_count, device=calibrated_curves.device, dtype=torch.float32)
    x_centered = x - x.mean()
    y_centered = calibrated_curves - calibrated_curves.mean(dim=1, keepdim=True)
    slope = (y_centered * x_centered.unsqueeze(0)).sum(dim=1) / x_centered.square().sum().clamp_min(1e-8)
    return {
        "cal_mean_drift": calibrated_curves.mean(dim=1),
        "cal_max_drift": calibrated_curves.max(dim=1).values,
        "cal_final_drift": calibrated_curves[:, -1],
        "cal_drift_slope": slope,
        "cal_drift_variance": calibrated_curves.var(dim=1, unbiased=False),
    }


def build_prompt_full_curve_features_gpu(
    *,
    raw_curve: torch.Tensor,
    calibrated_curve: torch.Tensor,
) -> dict[str, float]:
    if raw_curve.device.type != "cuda" or calibrated_curve.device.type != "cuda":
        raise ValueError("raw_curve and calibrated_curve must be CUDA tensors.")
    if raw_curve.ndim != 1 or calibrated_curve.ndim != 1:
        raise ValueError("raw_curve and calibrated_curve must be rank 1.")
    if raw_curve.shape[0] != calibrated_curve.shape[0]:
        raise ValueError("raw_curve and calibrated_curve must align.")
    summary = _calibrated_summary_gpu(calibrated_curve.unsqueeze(0))
    features = {
        f"raw_drift_{index}": float(value)
        for index, value in enumerate(raw_curve.detach().cpu().tolist())
    }
    features.update(
        {
            name: float(values.detach().cpu()[0])
            for name, values in summary.items()
        }
    )
    return features


def _feature_rows_from_curves_gpu(
    *,
    raw_curves: torch.Tensor,
    calibrated_curves: torch.Tensor,
) -> list[dict[str, float]]:
    if raw_curves.device.type != "cuda" or calibrated_curves.device.type != "cuda":
        raise ValueError("raw_curves and calibrated_curves must be CUDA tensors.")
    summary = _calibrated_summary_gpu(calibrated_curves)
    raw_cpu = raw_curves.detach().cpu()
    summary_cpu = {name: values.detach().cpu() for name, values in summary.items()}
    rows: list[dict[str, float]] = []
    for row_index in range(int(raw_cpu.shape[0])):
        raw_row = raw_cpu[row_index]
        row = {
            f"raw_drift_{column_index}": float(raw_row[column_index])
            for column_index in range(int(raw_cpu.shape[1]))
        }
        row["raw_max_drift"] = float(raw_row.max())
        row["raw_mean_drift"] = float(raw_row.mean())
        row["raw_peak_layer_index"] = float(raw_row.argmax())
        row.update(
            {
                name: float(values[row_index])
                for name, values in summary_cpu.items()
            }
        )
        rows.append(row)
    return rows


def _prepare_entries(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_stats: dict[str, dict[int, dict[str, float]]],
    bank_scope: str,
) -> tuple[list[dict[str, object]], dict[tuple[tuple[int, ...], str], list[int]]]:
    missing_entries: list[dict[str, object]] = []
    prepared_entries: list[dict[str, object]] = []
    grouped_indices: dict[tuple[tuple[int, ...], str], list[int]] = {}
    for entry in cache_entries:
        selected_layers = tuple(int(layer) for layer in entry["selected_layers"])
        object_name = str(entry["object_name"])
        bank_key = resolve_reference_scope_key(object_name, bank_scope)
        if bank_key not in reference_bank:
            missing_entries.append({"sample_id": entry["sample_id"], "reason": f"missing bank:{bank_key}"})
            continue
        if bank_key not in reference_stats:
            missing_entries.append({"sample_id": entry["sample_id"], "reason": f"missing stats:{bank_key}"})
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
            missing_entries.append({"sample_id": entry["sample_id"], "reason": "; ".join(reason_parts)})
            continue
        prepared_entries.append(
            {
                "entry": entry,
                "selected_layers": selected_layers,
                "bank_key": bank_key,
            }
        )
        grouped_indices.setdefault((selected_layers, bank_key), []).append(len(prepared_entries) - 1)

    if missing_entries:
        raise ValueError(_format_missing_reference_coverage(missing_entries))
    return prepared_entries, grouped_indices


def _reference_to_cuda(
    reference_cache: dict[tuple[str, int], torch.Tensor],
    *,
    reference_bank: dict[str, dict[int, torch.Tensor]],
    bank_key: str,
    layer_index: int,
    device: torch.device,
) -> torch.Tensor:
    cache_key = (bank_key, int(layer_index))
    if cache_key not in reference_cache:
        reference = reference_bank[bank_key][int(layer_index)]
        if reference.ndim != 2:
            raise ValueError(f"reference bank {bank_key}/layer-{layer_index} must be rank 2.")
        reference_cache[cache_key] = reference.to(device=device, dtype=torch.float32)
    return reference_cache[cache_key]


def _compute_curves_gpu(
    *,
    layer_vectors_batch: torch.Tensor,
    selected_layers: Sequence[int],
    bank_key: str,
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_cache: dict[tuple[str, int], torch.Tensor],
    curve_type: str,
    k_neighbors: int,
    variance_threshold: float,
    max_components: int,
    reference_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if layer_vectors_batch.ndim != 3:
        raise ValueError("layer_vectors_batch must be rank 3.")
    if layer_vectors_batch.shape[1] != len(selected_layers):
        raise ValueError("layer_vectors_batch and selected_layers must align.")
    queries_by_layer = layer_vectors_batch.to(device=device, dtype=torch.float32)
    curves = torch.empty(
        (int(queries_by_layer.shape[0]), len(selected_layers)),
        device=device,
        dtype=torch.float32,
    )
    for offset, layer_index in enumerate(selected_layers):
        reference = _reference_to_cuda(
            reference_cache,
            reference_bank=reference_bank,
            bank_key=bank_key,
            layer_index=int(layer_index),
            device=device,
        )
        queries = queries_by_layer[:, offset, :]
        if curve_type == "local_pca":
            curves[:, offset] = _local_pca_residuals_gpu(
                queries=queries,
                reference=reference,
                k_neighbors=k_neighbors,
                variance_threshold=variance_threshold,
                max_components=max_components,
                reference_chunk_size=reference_chunk_size,
            )
        elif curve_type == "no_manifold":
            curves[:, offset] = _neighbor_residuals_gpu(
                queries=queries,
                reference=reference,
                k_neighbors=k_neighbors,
                reference_chunk_size=reference_chunk_size,
            )
        else:
            raise ValueError(f"Unsupported curve_type: {curve_type}")
    return curves


def build_gpu_feature_frame(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_stats: dict[str, dict[int, dict[str, float]]],
    bank_scope: str = "object",
    curve_type: str = "local_pca",
    batch_size: int = 32,
    k_neighbors: int = 32,
    variance_threshold: float = 0.9,
    max_components: int = 32,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    device: str | torch.device = "cuda",
    label_overrides: Path | None = None,
) -> pd.DataFrame:
    if curve_type not in CURVE_TYPES:
        raise ValueError(f"Unsupported curve_type: {curve_type}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = resolve_cuda_device(device)
    prepared_entries, grouped_indices = _prepare_entries(
        cache_entries=cache_entries,
        reference_bank=reference_bank,
        reference_stats=reference_stats,
        bank_scope=bank_scope,
    )

    rows_by_index: list[dict[str, object] | None] = [None for _ in prepared_entries]
    reference_cache: dict[tuple[str, int], torch.Tensor] = {}
    with torch.no_grad():
        for (selected_layers_key, bank_key), group_indices in grouped_indices.items():
            selected_layers = list(selected_layers_key)
            layer_stats = reference_stats[bank_key]
            for start in range(0, len(group_indices), int(batch_size)):
                batch_indices = group_indices[start : start + int(batch_size)]
                layer_vectors_batch = torch.stack(
                    [
                        torch.as_tensor(prepared_entries[index]["entry"]["layer_vectors"], dtype=torch.float32)
                        for index in batch_indices
                    ],
                    dim=0,
                )
                raw_curves = _compute_curves_gpu(
                    layer_vectors_batch=layer_vectors_batch,
                    selected_layers=selected_layers,
                    bank_key=str(bank_key),
                    reference_bank=reference_bank,
                    reference_cache=reference_cache,
                    curve_type=curve_type,
                    k_neighbors=k_neighbors,
                    variance_threshold=variance_threshold,
                    max_components=max_components,
                    reference_chunk_size=reference_chunk_size,
                    device=device,
                )
                calibrated_curves = calibrate_curves_gpu(
                    raw_curves=raw_curves,
                    selected_layers=selected_layers,
                    layer_stats=layer_stats,
                    curve_type=curve_type,
                )
                feature_rows = _feature_rows_from_curves_gpu(
                    raw_curves=raw_curves,
                    calibrated_curves=calibrated_curves,
                )
                for batch_offset, prepared_index in enumerate(batch_indices):
                    entry = prepared_entries[prepared_index]["entry"]
                    rows_by_index[prepared_index] = {
                        **_metadata_row_from_entry(entry),
                        **feature_rows[batch_offset],
                    }

    rows = [row for row in rows_by_index if row is not None]
    if len(rows) != len(prepared_entries):
        raise RuntimeError("internal error: some prepared entries did not receive feature rows.")
    return apply_label_overrides(pd.DataFrame(rows), label_overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--bank-scope", choices=["object", "shared", "shuffled_object"], default="object")
    parser.add_argument("--curve-type", choices=CURVE_TYPES, default="local_pca")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--variance-threshold", type=float, default=0.9)
    parser.add_argument("--max-components", type=int, default=32)
    parser.add_argument("--reference-chunk-size", type=int, default=DEFAULT_REFERENCE_CHUNK_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--label-overrides", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    validate_cuda_visible_devices()
    args = build_parser().parse_args(argv)
    output_path = build_feature_output_path(
        output_path=args.output_path,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        split=args.split,
    )
    device = resolve_cuda_device(args.device)
    cache_entries = load_cache_entries(args.cache_path)
    if args.limit_rows is not None:
        cache_entries = cache_entries[: int(args.limit_rows)]
    reference_bank = load_reference_bank(args.reference_root, args.model_name, bank_scope=args.bank_scope)
    reference_stats = load_reference_stats(args.reference_root, args.model_name, bank_scope=args.bank_scope)

    try:
        with output_root_lock(
            output_path.parent,
            command=f"build_gpu_drift_features:{args.model_name}:{args.bank_scope}:{args.curve_type}",
        ):
            frame = build_gpu_feature_frame(
                cache_entries=cache_entries,
                reference_bank=reference_bank,
                reference_stats=reference_stats,
                bank_scope=args.bank_scope,
                curve_type=args.curve_type,
                batch_size=args.batch_size,
                k_neighbors=args.k_neighbors,
                variance_threshold=args.variance_threshold,
                max_components=args.max_components,
                reference_chunk_size=args.reference_chunk_size,
                device=device,
                label_overrides=args.label_overrides,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output_path, index=False)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise RuntimeError(
                "CUDA out of memory while building drift features. Re-run with a smaller --batch-size "
                "or --reference-chunk-size; this script does not switch vector math to CPU."
            ) from error
        raise
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
