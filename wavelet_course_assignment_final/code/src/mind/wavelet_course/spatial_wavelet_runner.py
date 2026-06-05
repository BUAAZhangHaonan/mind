"""Runner for the spatial hidden-dimension wavelet supplement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .common_sequence_models import train_sequence_model
from .metrics import METRIC_NAMES, evaluate_validation_test
from .population import WaveletPopulation
from .spatial_wavelet_features import SpatialWaveletConfig, spatial_dwt_stat28_sequence_batch
from .utils import SPLIT_NAMES


SPATIAL_EXPERIMENT_NAME = "F_spatial_dwt_db2_l2_universal_soft_stat28_sequence_lstm_projected"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 200
DEFAULT_QUICK_MAX_EPOCHS = 5
DEFAULT_PATIENCE = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SEQUENCE_MODEL = "lstm_projected"
SPATIAL_METRICS_FILENAME = "spatial_hidden_wavelet_metrics.csv"
SPATIAL_SUMMARY_FILENAME = "spatial_hidden_wavelet_summary.md"

CSV_FIELDS = (
    "name",
    "status",
    "failure_reason",
    "quick_run",
    "device",
    "wavelet",
    "level",
    "threshold",
    "sequence_model",
    "learning_rate",
    "max_epochs",
    "patience",
    "batch_size",
    "feature_shape",
    "train_shape",
    "validation_shape",
    "test_shape",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_pos",
    "val_pos",
    "test_pos",
    "pr_auc",
    "average_precision",
    "roc_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "fpr_at_95pct_tpr",
    "best_val_threshold",
    "threshold_selection_reason",
    "best_epoch",
    "best_validation_pr_auc",
    "epochs_ran",
    "early_stopped",
    "converged",
    "max_epoch_reached",
    "feature_seconds",
    "train_eval_seconds",
    "total_seconds",
)


@dataclass(frozen=True)
class SpatialRunSpec:
    name: str = SPATIAL_EXPERIMENT_NAME
    wavelet: str = "db2"
    level: int = 2
    threshold: str = "universal_soft"
    sequence_model: str = DEFAULT_SEQUENCE_MODEL
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    feature_batch_size: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class SpatialSplitArrays:
    features: np.ndarray
    labels: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def run_spatial_hidden_wavelet_experiment(
    *,
    config: Mapping[str, object],
    preflight: Mapping[str, object],
    output_root: Path | str,
    reports_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the fixed spatial hidden-dim DWT stat28 sequence experiment."""

    started_at = time.perf_counter()
    report_dir = Path(reports_dir) if reports_dir is not None else Path(output_root) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / SPATIAL_METRICS_FILENAME
    summary_path = report_dir / SPATIAL_SUMMARY_FILENAME

    quick_run = bool(config.get("quick_run", False))
    spec = spatial_run_spec_from_config(config, quick_run=quick_run)
    population = _population_from_preflight(preflight)
    entries, labels = _population_entries_and_labels(population)
    if quick_run:
        entries, labels = _limit_entries_for_quick(entries, labels, config)
    split_indices = _split_indices_by_name([str(entry.get("wavelet_split", "")) for entry in entries])
    _require_two_classes(labels[split_indices["train"]], split_name="train")

    feature_start = time.perf_counter()
    feature_config = SpatialWaveletConfig(
        wavelet=spec.wavelet,
        level=spec.level,
        threshold=spec.threshold,
        expected_num_layers=int(config.get("expected_num_layers", 36)),
        expected_hidden_dim=int(config.get("expected_hidden_dim", 4096)),
        epsilon=float(config.get("epsilon", 1e-12)),
    )
    features = _build_spatial_features(entries, feature_config, batch_size=spec.feature_batch_size)
    _raise_if_non_finite(features, name="spatial hidden features")
    feature_seconds = time.perf_counter() - feature_start
    split_arrays = _split_features(features, labels, split_indices=split_indices)

    train_start = time.perf_counter()
    result = train_sequence_model(
        spec.sequence_model,
        split_arrays.train_x,
        split_arrays.train_y,
        split_arrays.validation_x,
        split_arrays.validation_y,
        test_x=split_arrays.test_x,
        device=str(config.get("device", "cuda:0")),
        batch_size=spec.batch_size,
        max_epochs=spec.max_epochs,
        patience=spec.patience,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        seed=int(config.get("seed", 0)),
        hidden_dim=spec.hidden_dim,
        dropout=spec.dropout,
    )
    train_eval_seconds = time.perf_counter() - train_start
    evaluation = evaluate_validation_test(
        split_arrays.validation_y,
        result.scores.validation,
        split_arrays.test_y,
        result.scores.test,
    )
    row = _metric_row(
        spec,
        config=config,
        quick_run=quick_run,
        split_arrays=split_arrays,
        evaluation=evaluation,
        result=result,
        feature_seconds=feature_seconds,
        train_eval_seconds=train_eval_seconds,
        total_seconds=time.perf_counter() - started_at,
    )
    _write_metrics_csv([row], metrics_path)
    summary = _write_summary(row, summary_path)
    return {
        "status": "success",
        "name": spec.name,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "summary_snippet": summary,
        "feature_shape": _shape_text(features.shape),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def spatial_run_spec_from_config(config: Mapping[str, object], *, quick_run: bool) -> SpatialRunSpec:
    section = dict(config.get("spatial_hidden_wavelet", {}) or {})
    sequence_section = _sequence_model_section(config, DEFAULT_SEQUENCE_MODEL)
    batch_size = int(section.get("batch_size", sequence_section.get("batch_size", DEFAULT_BATCH_SIZE)))
    max_epochs = int(section.get("max_epochs", DEFAULT_MAX_EPOCHS))
    if quick_run:
        max_epochs = min(max_epochs, int(section.get("quick_max_epochs", DEFAULT_QUICK_MAX_EPOCHS)))
    return SpatialRunSpec(
        name=str(section.get("name", SPATIAL_EXPERIMENT_NAME)),
        wavelet=str(section.get("wavelet", "db2")),
        level=int(section.get("level", 2)),
        threshold=str(section.get("threshold", "universal_soft")),
        sequence_model=str(section.get("sequence_model", DEFAULT_SEQUENCE_MODEL)),
        learning_rate=float(section.get("learning_rate", DEFAULT_LEARNING_RATE)),
        max_epochs=max_epochs,
        patience=int(section.get("patience", DEFAULT_PATIENCE)),
        batch_size=batch_size,
        feature_batch_size=int(section.get("feature_batch_size", 128)),
        hidden_dim=int(section.get("hidden_dim", sequence_section.get("hidden_dim", 128))),
        dropout=float(section.get("dropout", sequence_section.get("dropout", 0.1))),
        weight_decay=float(section.get("weight_decay", sequence_section.get("weight_decay", 1e-4))),
    )


def _sequence_model_section(config: Mapping[str, object], name: str) -> dict[str, object]:
    value = config.get("sequence_models")
    if not isinstance(value, Mapping):
        return {}
    section = value.get(name)
    if not isinstance(section, Mapping):
        return {}
    return dict(section)


def _population_from_preflight(preflight: Mapping[str, object]) -> WaveletPopulation:
    population = preflight.get("population")
    if not isinstance(population, WaveletPopulation):
        raise ValueError("preflight must contain a WaveletPopulation at key 'population'")
    return population


def _population_entries_and_labels(population: WaveletPopulation) -> tuple[list[dict[str, object]], np.ndarray]:
    entries = [dict(entry) for entry in population.primary_entries]
    labels = np.asarray(population.labels, dtype=np.int64)
    if not entries:
        raise ValueError("WaveletPopulation.primary_entries must not be empty")
    if labels.ndim != 1 or labels.shape[0] != len(entries):
        raise ValueError("WaveletPopulation labels must match primary_entries")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("WaveletPopulation labels must contain only 0/1 values")
    for index, entry in enumerate(entries):
        if "layer_vectors" not in entry:
            raise ValueError(f"entry {index} is missing layer_vectors")
    return entries, labels


def _limit_entries_for_quick(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], np.ndarray]:
    quick = dict(config.get("quick", {}) or {})
    shared_limit = _first_int(quick, ("max_samples_per_split", "max_samples"))
    train_limit = _first_int(quick, ("max_train_samples",)) or shared_limit
    validation_limit = _first_int(quick, ("max_validation_samples", "max_val_samples")) or shared_limit or train_limit
    test_limit = _first_int(quick, ("max_test_samples",)) or shared_limit or train_limit
    limits = {
        "train": train_limit,
        "validation": validation_limit,
        "test": test_limit,
    }
    keep: set[int] = set()
    for split in SPLIT_NAMES:
        split_indices = [index for index, entry in enumerate(entries) if str(entry.get("wavelet_split", "")) == split]
        limit = limits[split]
        if limit is None:
            keep.update(split_indices)
        else:
            keep.update(_stratified_limit_indices(split_indices, labels, int(limit)))
    if not keep:
        raise ValueError("quick mode sample limits removed all population rows")
    limited_entries: list[dict[str, object]] = []
    limited_labels: list[int] = []
    for index, entry in enumerate(entries):
        if index in keep:
            limited_entries.append(dict(entry))
            limited_labels.append(int(labels[index]))
    return limited_entries, np.asarray(limited_labels, dtype=np.int64)


def _stratified_limit_indices(
    split_indices: Sequence[int],
    labels: np.ndarray,
    limit: int,
) -> list[int]:
    indices = [int(index) for index in split_indices]
    if limit < 0:
        raise ValueError("quick sample limits must be non-negative")
    if len(indices) <= limit:
        return indices
    if limit == 0:
        return []
    selected: set[int] = set()
    for label in (0, 1):
        first = next((index for index in indices if int(labels[index]) == label), None)
        if first is not None and len(selected) < limit:
            selected.add(first)
    for index in indices:
        if len(selected) >= limit:
            break
        selected.add(index)
    return [index for index in indices if index in selected]


def _first_int(mapping: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or str(value).strip() == "":
            continue
        return int(value)
    return None


def _build_spatial_features(
    entries: Sequence[Mapping[str, object]],
    feature_config: SpatialWaveletConfig,
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    parts: list[np.ndarray] = []
    for start in range(0, len(entries), int(batch_size)):
        stop = min(start + int(batch_size), len(entries))
        batch = np.stack([np.asarray(entry["layer_vectors"], dtype=np.float32) for entry in entries[start:stop]], axis=0)
        parts.append(spatial_dwt_stat28_sequence_batch(batch, feature_config))
    features = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    expected_shape = (len(entries), int(feature_config.expected_num_layers), 28)
    if features.shape != expected_shape:
        raise ValueError(f"spatial hidden feature shape {features.shape} != {expected_shape}")
    _raise_if_non_finite(features, name="spatial hidden features")
    return features


def _split_indices_by_name(split_names: Sequence[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        indices = np.asarray([index for index, value in enumerate(split_names) if value == split], dtype=np.int64)
        if indices.shape[0] == 0:
            raise ValueError(f"missing required split: {split}")
        arrays[split] = indices
    unknown = sorted(set(split_names) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"unknown split names in spatial entries: {unknown}")
    return arrays


def _split_features(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    split_indices: Mapping[str, np.ndarray],
) -> SpatialSplitArrays:
    return SpatialSplitArrays(
        features=features,
        labels=labels,
        train_x=features[split_indices["train"]],
        train_y=labels[split_indices["train"]],
        validation_x=features[split_indices["validation"]],
        validation_y=labels[split_indices["validation"]],
        test_x=features[split_indices["test"]],
        test_y=labels[split_indices["test"]],
    )


def _require_two_classes(labels: np.ndarray, *, split_name: str) -> None:
    if np.unique(np.asarray(labels, dtype=np.int64)).shape[0] < 2:
        raise RuntimeError(f"{split_name} split must contain two classes")


def _metric_row(
    spec: SpatialRunSpec,
    *,
    config: Mapping[str, object],
    quick_run: bool,
    split_arrays: SpatialSplitArrays,
    evaluation: Mapping[str, object],
    result: Any,
    feature_seconds: float,
    train_eval_seconds: float,
    total_seconds: float,
) -> dict[str, object]:
    test_metrics = dict(evaluation["test"])  # type: ignore[index]
    row = {
        "name": spec.name,
        "status": "success",
        "failure_reason": "",
        "quick_run": quick_run,
        "device": str(config.get("device", "cuda:0")),
        "wavelet": spec.wavelet,
        "level": spec.level,
        "threshold": spec.threshold,
        "sequence_model": spec.sequence_model,
        "learning_rate": spec.learning_rate,
        "max_epochs": result.max_epochs,
        "patience": result.patience,
        "batch_size": spec.batch_size,
        "feature_shape": _shape_text(split_arrays.features.shape),
        "train_shape": _shape_text(split_arrays.train_x.shape),
        "validation_shape": _shape_text(split_arrays.validation_x.shape),
        "test_shape": _shape_text(split_arrays.test_x.shape),
        "train_samples": int(split_arrays.train_y.shape[0]),
        "val_samples": int(split_arrays.validation_y.shape[0]),
        "test_samples": int(split_arrays.test_y.shape[0]),
        "train_pos": int(np.sum(split_arrays.train_y == 1)),
        "val_pos": int(np.sum(split_arrays.validation_y == 1)),
        "test_pos": int(np.sum(split_arrays.test_y == 1)),
        "pr_auc": test_metrics["pr_auc"],
        "average_precision": test_metrics["average_precision"],
        "roc_auc": test_metrics["roc_auc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "tpr_at_1pct_fpr": test_metrics["tpr_at_1pct_fpr"],
        "fpr_at_95pct_tpr": test_metrics["fpr_at_95pct_tpr"],
        "best_val_threshold": evaluation["threshold"],
        "threshold_selection_reason": evaluation.get("threshold_selection_reason", ""),
        "best_epoch": result.best_epoch,
        "best_validation_pr_auc": result.best_validation_pr_auc,
        "epochs_ran": result.epochs_ran,
        "early_stopped": result.early_stopped,
        "converged": result.converged,
        "max_epoch_reached": result.max_epoch_reached,
        "feature_seconds": feature_seconds,
        "train_eval_seconds": train_eval_seconds,
        "total_seconds": total_seconds,
    }
    for metric_name in METRIC_NAMES:
        row.setdefault(metric_name, test_metrics.get(metric_name, ""))
    return row


def _write_metrics_csv(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return path


def _write_summary(row: Mapping[str, object], path: Path) -> str:
    lines = [
        "# Spatial Hidden Wavelet Supplement",
        "",
        f"- name: {row['name']}",
        f"- status: {row['status']}",
        f"- feature_shape: {row['feature_shape']}",
        f"- pr_auc: {row['pr_auc']}",
        f"- roc_auc: {row['roc_auc']}",
        f"- test_f1: {row['test_f1']}",
        f"- max_epochs: {row['max_epochs']}",
    ]
    snippet = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snippet, encoding="utf-8")
    return snippet


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def _raise_if_non_finite(values: np.ndarray, *, name: str) -> None:
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError(f"{name} contains NaN or Inf")
