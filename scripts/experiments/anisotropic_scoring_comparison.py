#!/usr/bin/env python3
"""Run Sub-task 2 local anisotropic scoring with locked radius-ball neighbors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import NamedTuple, Sequence

import pandas as pd
import torch


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.geometry.gpu_anisotropic import (  # noqa: E402
    DEFAULT_EPS,
    DEFAULT_RANK_CAP,
    DEFAULT_REFERENCE_CHUNK_SIZE,
    VARIANT_NAMES,
    compute_multi_variant_scores_for_radius_ball_gpu,
)
from mind.geometry.gpu_distances import batch_angular_distance  # noqa: E402
from mind.geometry.neighbor_selection import DEFAULT_RADIUS_MARGIN  # noqa: E402
from mind.utils import output_root_lock  # noqa: E402


OUTPUT_ROOT = Path("outputs/subtask2_anisotropic")
DEFAULT_MAIN_MARKDOWN = OUTPUT_ROOT / "tables/subtask2_anisotropic_comparison.md"
DEFAULT_MAIN_CSV = OUTPUT_ROOT / "tables/subtask2_anisotropic_comparison.csv"
DEFAULT_HELDOUT_MARKDOWN = OUTPUT_ROOT / "tables/subtask2_heldout_transfer.md"
DEFAULT_HELDOUT_CSV = OUTPUT_ROOT / "tables/subtask2_heldout_transfer.csv"
DEFAULT_ANALYSIS = OUTPUT_ROOT / "review/subtask2_anisotropic_analysis.md"
DEFAULT_BASELINE_CSV = Path("docs/tables/experiment_query_local_bank.csv")
DEFAULT_HELDOUT_BASELINE_CSV = Path("docs/tables/table3_transfer_controls.csv")

METRIC_COLUMNS = (
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "pr_auc",
    "pr_auc_ci_lower",
    "pr_auc_ci_upper",
)
CALIBRATED_SUMMARY_COLUMNS = (
    "cal_mean_drift",
    "cal_max_drift",
    "cal_final_drift",
    "cal_drift_slope",
    "cal_drift_variance",
)
MAIN_TABLE_METHODS = (
    "radius_ball_isotropic",
    "diag_maha",
    "lowrank_maha",
    "full_maha_shrink",
    "no_manifold",
    "linear_probe",
)
ANISOTROPIC_METHODS = ("diag_maha", "lowrank_maha", "full_maha_shrink")
BASELINE_METHODS = ("no_manifold", "linear_probe", "query_local_k30")
HELDOUT_TABLE_METHODS = ("best_anisotropic", "no_manifold", "linear_probe")
DEFAULT_RANDOM_STATE = 13
HELDOUT_RANDOM_STATE_DEFAULTS = {
    "qwen3-vl-8b": 5,
    "internvl3.5-8b": 1,
    "internvl3-5-8b": 1,
    "llava-onevision-7b": 0,
    "molmo-7b-d-0924": 1,
}


class ArtifactPaths(NamedTuple):
    feature_path: Path
    metrics_path: Path
    predictions_path: Path


class FeatureBuildResult(NamedTuple):
    frames: dict[str, pd.DataFrame]
    support_floor_trigger_count: int
    total_query_layer_count: int


def _load_neighbor_selection_module():
    path = Path(__file__).resolve().parent / "neighbor_selection_comparison.py"
    spec = importlib.util.spec_from_file_location("neighbor_selection_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gpu_detector_module():
    path = Path(__file__).resolve().parent / "train_gpu_detector.py"
    spec = importlib.util.spec_from_file_location("train_gpu_detector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_NEIGHBOR = _load_neighbor_selection_module()
metadata_row_from_entry = _NEIGHBOR.metadata_row_from_entry
load_pooled_reference_layers = _NEIGHBOR.load_pooled_reference_layers
feature_columns = _NEIGHBOR.feature_columns


def _validate_positive_int(name: str, value: int) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be positive.")


def validate_cuda_visible_devices() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != "0":
        raise ValueError(f"run requires CUDA_VISIBLE_DEVICES=0, found {visible_devices!r}.")


def resolve_cuda_device(device_name: str | torch.device = "cuda:0") -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("Sub-task 2 run requires a CUDA device.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")
    if device.index not in (None, 0):
        raise ValueError("Use GPU 0 only.")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def parse_variants(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(VARIANT_NAMES)
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [variant for variant in variants if variant not in VARIANT_NAMES]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}")
    if not variants:
        raise ValueError("Provide at least one variant or 'all'.")
    if len(set(variants)) != len(variants):
        raise ValueError("Variants must not contain duplicates.")
    return variants


def parse_split_strategies(value: str) -> tuple[str, ...]:
    allowed = {"image_grouped", "object_heldout"}
    strategies = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [strategy for strategy in strategies if strategy not in allowed]
    if invalid:
        raise ValueError(f"Unsupported split strategies: {invalid}")
    if not strategies:
        raise ValueError("Provide at least one split strategy.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("Split strategies must not contain duplicates.")
    return strategies


def resolve_random_state(model_name: str, split_strategy: str, requested: str | int) -> int:
    requested_text = str(requested).strip().lower()
    if requested_text != "auto":
        try:
            return int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported --random-state value: {requested!r}. Use an integer or auto.") from exc
    if split_strategy == "object_heldout":
        raw_model_key = str(model_name).strip().lower()
        return HELDOUT_RANDOM_STATE_DEFAULTS.get(
            raw_model_key,
            HELDOUT_RANDOM_STATE_DEFAULTS.get(_model_key(model_name), DEFAULT_RANDOM_STATE),
        )
    return DEFAULT_RANDOM_STATE


def _parse_expected_curve_length(value: str | None) -> int | None:
    if value is None or value.strip().lower() in {"auto", "none", "0"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("--expected-curve-length must be positive, auto, none, or 0.")
    return parsed


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


MODEL_KEY_ALIASES = {
    "internvl3-5-8b": "internvl3.5-8b",
}


def _model_key(value: object) -> str:
    return MODEL_KEY_ALIASES.get(_slugify(value), _slugify(value))


def _model_display(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    return MODEL_KEY_ALIASES.get(_slugify(text), text)


def _model_path_slug(value: object) -> str:
    return _slugify(value)


def _benchmark_key(row_or_value: object) -> str:
    if isinstance(row_or_value, dict):
        explicit = row_or_value.get("benchmark_key")
        if explicit is not None and str(explicit).strip():
            return _slugify(explicit)
        value = row_or_value.get("benchmark", "")
    else:
        value = row_or_value
    slug = _slugify(value)
    if slug in {"pope-popular", "popular"}:
        return "popular"
    if slug in {"dash-b", "dashb"}:
        return "dash-b"
    return slug


def _benchmark_display(benchmark: str) -> str:
    key = _benchmark_key(benchmark)
    if key == "popular":
        return "POPE popular"
    if key == "dash-b":
        return "DASH-B"
    return str(benchmark)


def build_artifact_paths(
    *,
    output_root: Path,
    model_name: str,
    benchmark: str,
    split_strategy: str,
    variant: str,
) -> ArtifactPaths:
    model_slug = _model_path_slug(model_name)
    benchmark_slug = _slugify(benchmark)
    return ArtifactPaths(
        feature_path=output_root / "features" / model_slug / benchmark_slug / split_strategy / f"{variant}.parquet",
        metrics_path=output_root / "metrics" / model_slug / benchmark_slug / split_strategy / f"{variant}.json",
        predictions_path=output_root / "predictions" / model_slug / benchmark_slug / split_strategy / f"{variant}.parquet",
    )


def build_radii_path(*, output_root: Path, model_name: str, benchmark: str) -> Path:
    return output_root / "radii" / _model_path_slug(model_name) / _slugify(benchmark) / "radii.json"


def build_summary_path(*, output_root: Path, model_name: str, benchmark: str) -> Path:
    return output_root / "summaries" / _model_path_slug(model_name) / _slugify(benchmark) / "metrics.csv"


def _metric_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "missing"
    if str(row.get("status", "ok")) != "ok":
        reason = str(row.get("reason", "")).strip()
        return f"{row.get('status')}: {reason}".rstrip()
    return (
        f"ROC-AUC {float(row['roc_auc']):.4f} "
        f"[{float(row['roc_auc_ci_lower']):.4f}, {float(row['roc_auc_ci_upper']):.4f}]; "
        f"PR-AUC {float(row['pr_auc']):.4f} "
        f"[{float(row['pr_auc_ci_lower']):.4f}, {float(row['pr_auc_ci_upper']):.4f}]"
    )


def _metric_from_text_cell(cell: object) -> dict[str, float] | None:
    if cell is None or pd.isna(cell):
        return None
    match = re.search(
        r"ROC(?:-AUC)?\s+([0-9.]+)\s+\[([0-9.]+),\s*([0-9.]+)\];\s*"
        r"PR(?:-AUC)?\s+([0-9.]+)\s+\[([0-9.]+),\s*([0-9.]+)\]",
        str(cell),
    )
    if not match:
        return None
    values = [float(item) for item in match.groups()]
    return {
        "roc_auc": values[0],
        "roc_auc_ci_lower": values[1],
        "roc_auc_ci_upper": values[2],
        "pr_auc": values[3],
        "pr_auc_ci_lower": values[4],
        "pr_auc_ci_upper": values[5],
    }


def _read_csv_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if path is None:
            continue
        resolved_path = _resolve_existing_csv_path(path)
        if resolved_path is None:
            continue
        try:
            frame = pd.read_csv(resolved_path)
        except pd.errors.EmptyDataError:
            continue
        rows.extend(frame.where(pd.notna(frame), None).to_dict(orient="records"))
    return rows


def _resolve_existing_csv_path(path: Path) -> Path | None:
    if path.exists():
        return path
    for candidate in _summary_csv_path_candidates(path):
        if candidate.exists():
            return candidate
    return None


def _summary_csv_path_candidates(path: Path) -> list[Path]:
    if path.name != "metrics.csv":
        return []
    parts = list(path.parts)
    try:
        summary_index = len(parts) - 1 - list(reversed(parts)).index("summaries")
    except ValueError:
        return []
    if summary_index + 3 >= len(parts):
        return []

    model_part = parts[summary_index + 1]
    benchmark_part = parts[summary_index + 2]
    model_candidates = _unique_strings(
        (
            model_part,
            _model_path_slug(model_part),
            _model_path_slug(_model_display(model_part)),
        )
    )
    benchmark_candidates = _unique_strings(
        (
            benchmark_part,
            _benchmark_key(benchmark_part),
            _slugify(benchmark_part),
        )
    )

    candidates: list[Path] = []
    for model_slug in model_candidates:
        for benchmark_slug in benchmark_candidates:
            candidate_parts = list(parts)
            candidate_parts[summary_index + 1] = model_slug
            candidate_parts[summary_index + 2] = benchmark_slug
            candidate = Path(*candidate_parts)
            if candidate != path:
                candidates.append(candidate)
    return candidates


def _unique_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _with_keys(row: dict[str, object]) -> dict[str, object]:
    updated = dict(row)
    updated["model"] = _model_display(updated.get("model", ""))
    updated["model_key"] = _model_key(updated.get("model", ""))
    updated["benchmark_key"] = _benchmark_key(updated)
    return updated


def load_baseline_rows(path: Path, *, methods: Sequence[str] = BASELINE_METHODS) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    allowed = set(methods)
    for row in _read_csv_rows((path,)):
        if str(row.get("method", "")) not in allowed:
            continue
        rows.append(_with_keys(row))
    return rows


def load_heldout_baseline_rows(
    path: Path,
    *,
    methods: Sequence[str] = ("no_manifold", "linear_probe"),
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    allowed = set(methods)
    for row in _read_csv_rows((path,)):
        method = str(row.get("method", ""))
        if method not in allowed:
            continue
        metrics = _metric_from_text_cell(row.get("object_heldout"))
        if metrics is None:
            continue
        out = {
            "model": row.get("model", ""),
            "benchmark": row.get("benchmark", "POPE popular"),
            "benchmark_key": _benchmark_key(row),
            "method": method,
            "status": "ok",
            "split_strategy": "object_heldout",
        }
        out.update(metrics)
        rows.append(_with_keys(out))
    return rows


def load_heldout_baseline_metric_rows(
    paths: Sequence[Path],
    *,
    methods: Sequence[str] = BASELINE_METHODS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    allowed = set(methods)
    for row in _read_csv_rows(paths):
        method = str(row.get("method", ""))
        if method not in allowed:
            continue
        if str(row.get("split_strategy", "object_heldout")) != "object_heldout":
            continue
        out = dict(row)
        out.setdefault("benchmark", "POPE popular")
        out.setdefault("status", "ok")
        out["split_strategy"] = "object_heldout"
        rows.append(_with_keys(out))
    return rows


def _curve_to_feature_row(curve: torch.Tensor, *, expected_curve_length: int | None) -> dict[str, float]:
    if expected_curve_length is not None and int(curve.numel()) != int(expected_curve_length):
        raise ValueError(f"raw_plus_full_curve expected {expected_curve_length} raw values, got {curve.numel()}.")
    if curve.numel() == 0:
        raise ValueError("raw_plus_full_curve requires at least one value.")
    x = torch.arange(curve.numel(), device=curve.device, dtype=curve.dtype)
    x_centered = x - x.mean()
    y_centered = curve - curve.mean()
    slope = (x_centered * y_centered).sum() / x_centered.square().sum().clamp_min(1e-8)
    row = {f"raw_drift_{index}": float(value.detach().cpu()) for index, value in enumerate(curve)}
    row.update(
        {
            "cal_mean_drift": float(curve.mean().detach().cpu()),
            "cal_max_drift": float(curve.max().detach().cpu()),
            "cal_final_drift": float(curve[-1].detach().cpu()),
            "cal_drift_slope": float(slope.detach().cpu()),
            "cal_drift_variance": float(curve.var(unbiased=False).detach().cpu()),
        }
    )
    return row


def _round_radius_outward(radius: torch.Tensor) -> torch.Tensor:
    radius_f = radius.detach().to(dtype=torch.float32)
    return torch.nextafter(
        radius_f,
        torch.tensor(float("inf"), dtype=radius_f.dtype, device=radius_f.device),
    )


def _layer_query_vectors(
    *,
    cache_entries: Sequence[dict[str, object]],
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for entry in cache_entries:
        selected_layers = [int(value) for value in entry["selected_layers"]]
        if int(layer) not in selected_layers:
            continue
        offset = selected_layers.index(int(layer))
        layer_vectors = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32, device=device)
        vectors.append(layer_vectors[offset])
    if not vectors:
        raise ValueError(f"No query vectors found for layer {layer}.")
    return torch.stack(vectors, dim=0)


def tune_radius_for_target_mean_count_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    target_count: int,
    query_chunk_size: int = 512,
    reference_chunk_size: int = DEFAULT_REFERENCE_CHUNK_SIZE,
    binary_steps: int = 32,
) -> torch.Tensor:
    """Tune radius from zero so the mean radius-ball count targets ``target_count``."""
    _validate_positive_int("target_count", target_count)
    _validate_positive_int("query_chunk_size", query_chunk_size)
    _validate_positive_int("reference_chunk_size", reference_chunk_size)
    _validate_positive_int("binary_steps", binary_steps)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("query and reference must be 2D matrices.")
    if query.shape[0] == 0:
        raise ValueError("query must contain at least one row.")
    if reference.shape[0] == 0:
        raise ValueError("reference must contain at least one row.")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same feature dimension.")
    distances = batch_angular_distance(
        query.to(dtype=torch.float32),
        reference.to(dtype=torch.float32),
        query_chunk_size=query_chunk_size,
        reference_chunk_size=reference_chunk_size,
    ).detach()
    low = torch.zeros((), device=distances.device, dtype=torch.float32)
    high = distances.max().to(dtype=torch.float32)
    target = float(min(int(target_count), int(reference.shape[0])))
    for _step in range(int(binary_steps)):
        midpoint = (low + high) / 2.0
        mean_count = (distances <= midpoint).sum(dim=1).to(dtype=torch.float32).mean()
        if float(mean_count.detach().cpu()) < target:
            low = midpoint
        else:
            high = midpoint
    return _round_radius_outward(high)


def tune_subtask2_radius_ball_layer_radii_gpu(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_layers: dict[int, torch.Tensor],
    device: torch.device,
    target_count: int,
    reference_chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Tune Sub-task 2 layer radii by true target mean count, allowing empty balls."""
    selected_layers = sorted({int(layer) for entry in cache_entries for layer in entry["selected_layers"]})
    radii: dict[int, torch.Tensor] = {}
    for layer in selected_layers:
        if layer not in reference_layers:
            raise KeyError(f"Missing reference layer: {layer}")
        query = _layer_query_vectors(cache_entries=cache_entries, layer=layer, device=device)
        radii[layer] = tune_radius_for_target_mean_count_gpu(
            query,
            reference_layers[layer],
            target_count=target_count,
            reference_chunk_size=reference_chunk_size,
        )
    return radii


def _raw_radius_neighbor_count_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor,
    reference_chunk_size: int,
    radius_margin: float,
) -> torch.Tensor:
    distances = batch_angular_distance(
        query.to(dtype=torch.float32),
        reference.to(dtype=torch.float32),
        query_chunk_size=1,
        reference_chunk_size=reference_chunk_size,
    )
    radius_value = radius.to(device=query.device, dtype=torch.float32) + float(radius_margin)
    return (distances <= radius_value).sum(dim=1).to(dtype=torch.float32)


def build_variant_feature_frames(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_layers: dict[int, torch.Tensor],
    layer_radii: dict[int, torch.Tensor],
    variants: Sequence[str],
    device: torch.device,
    eps: float,
    rank_cap: int,
    reference_chunk_size: int,
    radius_margin: float,
    min_neighbors: int,
    expected_curve_length: int | None,
) -> FeatureBuildResult:
    rows_by_variant: dict[str, list[dict[str, object]]] = {variant: [] for variant in variants}
    support_floor_trigger_count = 0
    total_query_layer_count = 0
    with torch.inference_mode():
        for entry in cache_entries:
            selected_layers = [int(layer) for layer in entry["selected_layers"]]
            layer_vectors = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32, device=device)
            if len(selected_layers) != int(layer_vectors.shape[0]):
                raise ValueError("selected_layers must align with layer_vectors rows")
            curves: dict[str, list[torch.Tensor]] = {variant: [] for variant in variants}
            counts: list[torch.Tensor] = []
            for offset, layer in enumerate(selected_layers):
                if layer not in reference_layers:
                    raise KeyError(f"Missing reference layer: {layer}")
                if layer not in layer_radii:
                    raise KeyError(f"Missing tuned radius for layer: {layer}")
                raw_count = _raw_radius_neighbor_count_gpu(
                    layer_vectors[offset : offset + 1],
                    reference_layers[layer],
                    radius=layer_radii[layer],
                    reference_chunk_size=reference_chunk_size,
                    radius_margin=radius_margin,
                ).squeeze(0)
                total_query_layer_count += 1
                if float(raw_count.detach().cpu()) < float(min_neighbors):
                    support_floor_trigger_count += 1
                results = compute_multi_variant_scores_for_radius_ball_gpu(
                    layer_vectors[offset : offset + 1],
                    reference_layers[layer],
                    radius=layer_radii[layer],
                    variants=variants,
                    eps=eps,
                    rank_cap=rank_cap,
                    reference_chunk_size=reference_chunk_size,
                    radius_margin=radius_margin,
                    min_neighbors=min_neighbors,
                )
                first_result = results[variants[0]]
                if first_result.neighbor_counts is not None:
                    counts.append(first_result.neighbor_counts.squeeze(0))
                for variant in variants:
                    curves[variant].append(results[variant].values.squeeze(0))

            count_tensor = torch.stack(counts).to(dtype=torch.float32) if counts else torch.empty(0, device=device)
            for variant in variants:
                curve = torch.stack(curves[variant]).to(dtype=torch.float32)
                row = metadata_row_from_entry(dict(entry))
                row.update(_curve_to_feature_row(curve, expected_curve_length=expected_curve_length))
                row["n_raw_features"] = int(curve.numel())
                if count_tensor.numel() > 0:
                    row["mean_neighbor_count"] = float(count_tensor.mean().detach().cpu())
                    row["min_neighbor_count"] = float(count_tensor.min().detach().cpu())
                    row["max_neighbor_count"] = float(count_tensor.max().detach().cpu())
                rows_by_variant[variant].append(row)
    return FeatureBuildResult(
        frames={variant: pd.DataFrame(rows) for variant, rows in rows_by_variant.items()},
        support_floor_trigger_count=support_floor_trigger_count,
        total_query_layer_count=total_query_layer_count,
    )


def save_radii_json(
    *,
    path: Path,
    model_name: str,
    benchmark: str,
    layer_radii: dict[int, torch.Tensor],
    target_count: int,
    min_neighbors: int,
    reference_chunk_size: int,
    radius_margin: float,
    n_rows: int,
    limit_rows: int | None,
    support_floor_trigger_count: int | None = None,
    total_query_layer_count: int | None = None,
) -> None:
    payload = {
        "model": model_name,
        "benchmark": benchmark,
        "benchmark_key": _benchmark_key(benchmark),
        "radius_source": "subtask2_mean_count_radius_ball",
        "target_count": int(target_count),
        "support_floor_min_neighbors": int(min_neighbors),
        "reference_chunk_size": int(reference_chunk_size),
        "radius_margin": float(radius_margin),
        "n_tuning_rows": int(n_rows),
        "limit_rows": None if limit_rows is None else int(limit_rows),
        "radii_by_layer": {
            str(int(layer)): float(radius.detach().cpu()) for layer, radius in sorted(layer_radii.items())
        },
    }
    if support_floor_trigger_count is not None:
        payload["support_floor_trigger_count"] = int(support_floor_trigger_count)
    if total_query_layer_count is not None:
        payload["total_query_layer_count"] = int(total_query_layer_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_frame_gpu(
    frame: pd.DataFrame,
    *,
    split_strategy: str,
    device: torch.device,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: str | int,
    max_iter: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    detector = _load_gpu_detector_module()
    metrics, predictions, _states = detector.evaluate_frame(
        frame,
        columns=feature_columns(frame),
        split_strategy=split_strategy,
        device=device,
        random_state=random_state,
        num_folds=num_folds,
        max_iter=max_iter,
        bootstrap_resamples=bootstrap_resamples,
    )
    return metrics, predictions


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_comparison(
    *,
    cache_path: Path,
    pooled_bank_root: Path,
    model_name: str,
    benchmark: str,
    variants: Sequence[str],
    split_strategies: Sequence[str],
    output_root: Path,
    device: torch.device,
    target_count: int,
    min_neighbors: int,
    reference_chunk_size: int,
    radius_margin: float,
    limit_rows: int | None,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
    expected_curve_length: int | None,
    eps: float,
    rank_cap: int,
) -> tuple[list[dict[str, object]], Path]:
    from mind.evaluation.baselines import load_cache_entries

    entries = load_cache_entries(
        cache_path,
        keep_fields={
            "sample_id",
            "image_id",
            "label",
            "parsed_answer",
            "subset",
            "object_name",
            "selected_layers",
            "layer_vectors",
        },
    )
    if not entries:
        raise ValueError(f"No eval cache entries found: {cache_path}")
    working_entries = entries[:limit_rows] if limit_rows is not None else entries
    if not working_entries:
        raise ValueError("--limit-rows selected zero rows.")
    selected_layers = sorted({int(layer) for entry in working_entries for layer in entry["selected_layers"]})
    reference_layers = load_pooled_reference_layers(pooled_bank_root, model_name, selected_layers, device=device)
    layer_radii = tune_subtask2_radius_ball_layer_radii_gpu(
        cache_entries=working_entries,
        reference_layers=reference_layers,
        device=device,
        target_count=target_count,
        reference_chunk_size=reference_chunk_size,
    )

    feature_result = build_variant_feature_frames(
        cache_entries=working_entries,
        reference_layers=reference_layers,
        layer_radii=layer_radii,
        variants=variants,
        device=device,
        eps=eps,
        rank_cap=rank_cap,
        reference_chunk_size=reference_chunk_size,
        radius_margin=radius_margin,
        min_neighbors=min_neighbors,
        expected_curve_length=expected_curve_length,
    )
    frames = feature_result.frames
    radii_path = build_radii_path(output_root=output_root, model_name=model_name, benchmark=benchmark)
    save_radii_json(
        path=radii_path,
        model_name=model_name,
        benchmark=benchmark,
        layer_radii=layer_radii,
        target_count=target_count,
        min_neighbors=min_neighbors,
        reference_chunk_size=reference_chunk_size,
        radius_margin=radius_margin,
        n_rows=len(working_entries),
        limit_rows=limit_rows,
        support_floor_trigger_count=feature_result.support_floor_trigger_count,
        total_query_layer_count=feature_result.total_query_layer_count,
    )

    rows: list[dict[str, object]] = []
    for split_strategy in split_strategies:
        for variant, frame in frames.items():
            paths = build_artifact_paths(
                output_root=output_root,
                model_name=model_name,
                benchmark=benchmark,
                split_strategy=split_strategy,
                variant=variant,
            )
            paths.feature_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(paths.feature_path, index=False)
            metrics, predictions = evaluate_frame_gpu(
                frame,
                split_strategy=split_strategy,
                device=device,
                bootstrap_resamples=bootstrap_resamples,
                num_folds=num_folds,
                random_state=resolve_random_state(model_name, split_strategy, random_state),
                max_iter=max_iter,
            )
            paths.predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(paths.predictions_path, index=False)
            metric_payload: dict[str, object] = {
                "model": model_name,
                "benchmark": _benchmark_display(benchmark),
                "benchmark_key": _benchmark_key(benchmark),
                "method": variant,
                "status": "ok",
                "split_strategy": split_strategy,
                "n_rows": int(len(frame)),
                "feature_source": str(paths.feature_path),
                "predictions_path": str(paths.predictions_path),
                "radii_path": str(radii_path),
                "target_count": int(target_count),
                "support_floor_min_neighbors": int(min_neighbors),
                "support_floor_trigger_count": int(feature_result.support_floor_trigger_count),
                "total_query_layer_count": int(feature_result.total_query_layer_count),
                "radius_margin": float(radius_margin),
                "expected_curve_length": expected_curve_length,
            }
            metric_payload.update(metrics)
            _write_json(paths.metrics_path, metric_payload)
            row = dict(metric_payload)
            row["metrics_path"] = str(paths.metrics_path)
            rows.append(row)

    summary_path = build_summary_path(output_root=output_root, model_name=model_name, benchmark=benchmark)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return rows, summary_path


def _validate_metric_rows(rows: Sequence[dict[str, object]]) -> None:
    for index, row in enumerate(rows):
        if str(row.get("status", "ok")) != "ok":
            continue
        for column in METRIC_COLUMNS:
            value = row.get(column)
            if value is None or pd.isna(value):
                raise ValueError(f"Metric row {index + 1} is missing {column}.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Metric row {index + 1} has non-finite {column}: {value!r}.")


def _sort_group_keys(rows: Sequence[dict[str, object]]) -> list[tuple[str, str]]:
    return sorted({(str(row.get("model", "")), str(row.get("benchmark", ""))) for row in rows})


def _row_lookup(rows: Sequence[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        keyed = _with_keys(dict(row))
        lookup[(str(keyed["model_key"]), str(keyed["benchmark_key"]), str(keyed["method"]))] = keyed
    return lookup


def _main_table_rows(
    metric_rows: Sequence[dict[str, object]],
    baseline_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    image_rows = [
        _with_keys(dict(row))
        for row in metric_rows
        if str(row.get("split_strategy", "image_grouped")) == "image_grouped"
    ]
    baseline_lookup = _row_lookup(baseline_rows)
    metric_lookup = _row_lookup(image_rows)
    output: list[dict[str, object]] = []
    for model, benchmark in _sort_group_keys(image_rows):
        context = _with_keys({"model": model, "benchmark": benchmark})
        model_key = str(context["model_key"])
        benchmark_key = str(context["benchmark_key"])
        row: dict[str, object] = {"model": model, "benchmark": benchmark}
        for method in MAIN_TABLE_METHODS:
            source = metric_lookup.get((model_key, benchmark_key, method))
            if source is None:
                source = baseline_lookup.get((model_key, benchmark_key, method))
            row[method] = _metric_cell(source)
        output.append(row)
    return output


def _select_best_anisotropic_by_dashb(metric_rows: Sequence[dict[str, object]]) -> str | None:
    totals: dict[str, list[float]] = {method: [] for method in ANISOTROPIC_METHODS}
    for row in metric_rows:
        keyed = _with_keys(dict(row))
        method = str(keyed.get("method", ""))
        if method not in totals:
            continue
        if str(keyed.get("split_strategy", "image_grouped")) != "image_grouped":
            continue
        if str(keyed.get("benchmark_key")) != "dash-b":
            continue
        if str(keyed.get("status", "ok")) != "ok":
            continue
        totals[method].append(float(keyed["pr_auc"]))
    ranked = [
        (sum(values) / len(values), method)
        for method, values in totals.items()
        if values
    ]
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def _heldout_table_rows(
    heldout_rows: Sequence[dict[str, object]],
    heldout_baseline_rows: Sequence[dict[str, object]],
    *,
    best_anisotropic: str | None,
) -> list[dict[str, object]]:
    heldout_lookup = _row_lookup(
        [
            _with_keys(dict(row))
            for row in heldout_rows
            if str(row.get("split_strategy", "object_heldout")) == "object_heldout"
        ]
    )
    baseline_lookup = _row_lookup(heldout_baseline_rows)
    model_rows = sorted({(str(_with_keys(dict(row))["model_key"]), str(row.get("model", ""))) for row in heldout_rows})
    output: list[dict[str, object]] = []
    for model_key, model in model_rows:
        selected_method = best_anisotropic
        if selected_method is None or (model_key, "popular", selected_method) not in heldout_lookup:
            candidates = [
                row for row in heldout_rows
                if _model_key(row.get("model", "")) == model_key
                and str(row.get("method", "")) in ANISOTROPIC_METHODS
                and str(row.get("split_strategy", "object_heldout")) == "object_heldout"
            ]
            candidates = sorted(candidates, key=lambda row: float(row.get("pr_auc", 0.0)), reverse=True)
            selected_method = str(candidates[0]["method"]) if candidates else None
        best_row = None
        if selected_method is not None:
            best_row = heldout_lookup.get((model_key, "popular", selected_method))
        row = {
            "model": model,
            "best_anisotropic": _metric_cell(best_row),
            "no_manifold": _metric_cell(baseline_lookup.get((model_key, "popular", "no_manifold"))),
            "linear_probe": _metric_cell(baseline_lookup.get((model_key, "popular", "linear_probe"))),
        }
        output.append(row)
    return output


def _write_markdown_table(path: Path, *, title: str, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_analysis_stub(
    path: Path,
    *,
    metric_rows: Sequence[dict[str, object]],
    best_anisotropic: str | None,
) -> None:
    image_rows = [
        row for row in metric_rows
        if str(row.get("split_strategy", "image_grouped")) == "image_grouped"
        and str(row.get("status", "ok")) == "ok"
    ]
    means: dict[str, float] = {}
    for method in VARIANT_NAMES:
        values = [float(row["pr_auc"]) for row in image_rows if str(row.get("method")) == method]
        if values:
            means[method] = sum(values) / len(values)
    lines = [
        "# Subtask 2 Anisotropic Analysis Draft",
        "",
        "## Decision Gate",
        "",
        f"- Best DASH-B anisotropic variant: {best_anisotropic or 'missing'}.",
    ]
    for method, value in sorted(means.items()):
        lines.append(f"- Mean image-grouped PR-AUC for {method}: {value:.4f}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_formatted_outputs(
    *,
    metrics_csv: Path,
    heldout_csv: Path | None,
    baseline_csv: Path,
    heldout_baseline_csv: Path,
    heldout_baseline_metrics_csvs: Sequence[Path] = (),
    markdown_path: Path,
    csv_path: Path,
    heldout_markdown_path: Path,
    heldout_csv_path: Path,
    analysis_path: Path,
) -> dict[str, int | str | None]:
    metric_rows = [_with_keys(row) for row in _read_csv_rows((metrics_csv,))]
    heldout_rows = [_with_keys(row) for row in _read_csv_rows((heldout_csv,))] if heldout_csv else []
    baseline_rows = load_baseline_rows(baseline_csv, methods=BASELINE_METHODS)
    heldout_baseline_rows = [
        *load_heldout_baseline_rows(heldout_baseline_csv),
        *load_heldout_baseline_metric_rows(heldout_baseline_metrics_csvs),
    ]
    _validate_metric_rows(metric_rows)
    _validate_metric_rows(heldout_rows)
    _validate_metric_rows(heldout_baseline_rows)

    main_rows = _main_table_rows(metric_rows, baseline_rows)
    best_anisotropic = _select_best_anisotropic_by_dashb(metric_rows)
    heldout_table_rows = _heldout_table_rows(
        heldout_rows,
        heldout_baseline_rows,
        best_anisotropic=best_anisotropic,
    )

    main_columns = ("model", "benchmark", *MAIN_TABLE_METHODS)
    heldout_columns = ("model", *HELDOUT_TABLE_METHODS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(main_rows, columns=main_columns).to_csv(csv_path, index=False)
    heldout_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(heldout_table_rows, columns=heldout_columns).to_csv(heldout_csv_path, index=False)
    _write_markdown_table(
        markdown_path,
        title="Subtask 2 Anisotropic Scoring Comparison",
        rows=main_rows,
        columns=main_columns,
    )
    _write_markdown_table(
        heldout_markdown_path,
        title="Subtask 2 Held-out Transfer",
        rows=heldout_table_rows,
        columns=heldout_columns,
    )
    _write_analysis_stub(analysis_path, metric_rows=metric_rows, best_anisotropic=best_anisotropic)
    return {
        "main_rows": len(main_rows),
        "heldout_rows": len(heldout_table_rows),
        "best_anisotropic": best_anisotropic,
    }


def write_formatted_outputs_from_many(
    *,
    metrics_csvs: Sequence[Path],
    heldout_csvs: Sequence[Path],
    baseline_csv: Path,
    heldout_baseline_csv: Path,
    heldout_baseline_metrics_csvs: Sequence[Path] = (),
    markdown_path: Path,
    csv_path: Path,
    heldout_markdown_path: Path,
    heldout_csv_path: Path,
    analysis_path: Path,
) -> dict[str, int | str | None]:
    combined_dir = csv_path.parent
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_metrics = combined_dir / ".anisotropic_metrics_combined.tmp.csv"
    combined_heldout = combined_dir / ".anisotropic_heldout_combined.tmp.csv"
    metric_rows = _read_csv_rows(metrics_csvs)
    if not metric_rows:
        raise ValueError("No metric rows were found in --metrics-csv inputs.")
    heldout_rows = _read_csv_rows(heldout_csvs)
    pd.DataFrame(metric_rows).to_csv(combined_metrics, index=False)
    heldout_arg: Path | None = None
    if heldout_rows:
        pd.DataFrame(heldout_rows).to_csv(combined_heldout, index=False)
        heldout_arg = combined_heldout
    try:
        return write_formatted_outputs(
            metrics_csv=combined_metrics,
            heldout_csv=heldout_arg,
            baseline_csv=baseline_csv,
            heldout_baseline_csv=heldout_baseline_csv,
            heldout_baseline_metrics_csvs=heldout_baseline_metrics_csvs,
            markdown_path=markdown_path,
            csv_path=csv_path,
            heldout_markdown_path=heldout_markdown_path,
            heldout_csv_path=heldout_csv_path,
            analysis_path=analysis_path,
        )
    finally:
        combined_metrics.unlink(missing_ok=True)
        combined_heldout.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Variants: {', '.join(VARIANT_NAMES)}",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run Sub-task 2 scoring and evaluation on CUDA GPU 0.")
    run_parser.add_argument("--cache-path", type=Path, required=True)
    run_parser.add_argument("--pooled-bank-root", type=Path, required=True)
    run_parser.add_argument("--model-name", required=True)
    run_parser.add_argument("--benchmark", required=True)
    run_parser.add_argument("--variants", default="all", help=f"Comma-separated variants or all: {', '.join(VARIANT_NAMES)}")
    run_parser.add_argument("--split-strategies", default="image_grouped")
    run_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--target-count", type=int, default=30)
    run_parser.add_argument("--min-neighbors", type=int, default=2)
    run_parser.add_argument("--reference-chunk-size", type=int, default=DEFAULT_REFERENCE_CHUNK_SIZE)
    run_parser.add_argument("--radius-margin", type=float, default=DEFAULT_RADIUS_MARGIN)
    run_parser.add_argument("--limit-rows", type=int, default=None)
    run_parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    run_parser.add_argument("--num-folds", type=int, default=5)
    run_parser.add_argument("--random-state", default="auto")
    run_parser.add_argument("--max-iter", type=int, default=100)
    run_parser.add_argument("--expected-curve-length", default="auto")
    run_parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    run_parser.add_argument("--rank-cap", type=int, default=DEFAULT_RANK_CAP)

    format_parser = subparsers.add_parser("format", help="Format saved metric CSVs without reading caches.")
    format_parser.add_argument("--metrics-csv", type=Path, action="append", required=True)
    format_parser.add_argument("--heldout-csv", type=Path, action="append", default=[])
    format_parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    format_parser.add_argument("--heldout-baseline-csv", type=Path, default=DEFAULT_HELDOUT_BASELINE_CSV)
    format_parser.add_argument("--heldout-baseline-metrics-csv", type=Path, action="append", default=[])
    format_parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MAIN_MARKDOWN)
    format_parser.add_argument("--csv-path", type=Path, default=DEFAULT_MAIN_CSV)
    format_parser.add_argument("--heldout-markdown-path", type=Path, default=DEFAULT_HELDOUT_MARKDOWN)
    format_parser.add_argument("--heldout-csv-path", type=Path, default=DEFAULT_HELDOUT_CSV)
    format_parser.add_argument("--analysis-path", type=Path, default=DEFAULT_ANALYSIS)

    smoke_parser = subparsers.add_parser("smoke", help="Print planned artifact paths without CUDA work.")
    smoke_parser.add_argument("--model-name", required=True)
    smoke_parser.add_argument("--benchmark", required=True)
    smoke_parser.add_argument("--variant", default="diag_maha", choices=VARIANT_NAMES)
    smoke_parser.add_argument("--split-strategy", default="image_grouped", choices=("image_grouped", "object_heldout"))
    smoke_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "run":
            validate_cuda_visible_devices()
            device = resolve_cuda_device(args.device)
            with output_root_lock(args.output_root, command="anisotropic_scoring_comparison"):
                rows, summary_path = run_comparison(
                    cache_path=args.cache_path,
                    pooled_bank_root=args.pooled_bank_root,
                    model_name=args.model_name,
                    benchmark=args.benchmark,
                    variants=parse_variants(args.variants),
                    split_strategies=parse_split_strategies(args.split_strategies),
                    output_root=args.output_root,
                    device=device,
                    target_count=args.target_count,
                    min_neighbors=args.min_neighbors,
                    reference_chunk_size=args.reference_chunk_size,
                    radius_margin=args.radius_margin,
                    limit_rows=args.limit_rows,
                    bootstrap_resamples=args.bootstrap_resamples,
                    num_folds=args.num_folds,
                    random_state=args.random_state,
                    max_iter=args.max_iter,
                    expected_curve_length=_parse_expected_curve_length(args.expected_curve_length),
                    eps=args.eps,
                    rank_cap=args.rank_cap,
                )
            print(json.dumps({"rows": len(rows), "summary_csv": str(summary_path)}, sort_keys=True))
            return 0
        if args.command == "format":
            with output_root_lock(args.csv_path.parent, command="anisotropic_scoring_comparison_format"):
                output = write_formatted_outputs_from_many(
                    metrics_csvs=args.metrics_csv,
                    heldout_csvs=args.heldout_csv,
                    baseline_csv=args.baseline_csv,
                    heldout_baseline_csv=args.heldout_baseline_csv,
                    heldout_baseline_metrics_csvs=args.heldout_baseline_metrics_csv,
                    markdown_path=args.markdown_path,
                    csv_path=args.csv_path,
                    heldout_markdown_path=args.heldout_markdown_path,
                    heldout_csv_path=args.heldout_csv_path,
                    analysis_path=args.analysis_path,
                )
            print(json.dumps(output, sort_keys=True))
            return 0
        if args.command == "smoke":
            paths = build_artifact_paths(
                output_root=args.output_root,
                model_name=args.model_name,
                benchmark=args.benchmark,
                split_strategy=args.split_strategy,
                variant=args.variant,
            )
            print(
                json.dumps(
                    {
                        "feature_path": str(paths.feature_path),
                        "metrics_path": str(paths.metrics_path),
                        "predictions_path": str(paths.predictions_path),
                        "radii_path": str(build_radii_path(output_root=args.output_root, model_name=args.model_name, benchmark=args.benchmark)),
                    },
                    sort_keys=True,
                )
            )
            return 0
        raise ValueError(f"Unsupported command: {args.command}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
