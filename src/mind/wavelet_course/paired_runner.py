"""Full paired-wavelet v2 runner.

This module is the hook loaded by ``scripts/wavelet_course_v2_run.py``. It
keeps the paired-grid contract strict: every configured readout emits one
Teacher row and one Ours row, including failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .common_classifiers import (
    STATIC_CLASSIFIERS,
    XGBOOST_ALIASES,
    XGBOOST_NOT_INSTALLED,
    XGBoostNotInstalledError,
    train_static_classifier,
)
from .common_feature_protocols import features_for_protocol
from .common_sequence_models import SEQUENCE_MODELS, train_sequence_model
from .metrics import METRIC_NAMES, evaluate_validation_test
from .paired_config import PAIR_BLOCKS, PAIR_SOURCES, PairSpec, pair_spec_from_mapping
from .paired_grid import assert_paired_grid_complete, build_paired_grid
from .ours_wavelet_features import NO_CANDIDATES, YES_CANDIDATES, resolve_yes_no_token_ids
from .paired_reporting import LONG_FIELD_ORDER, write_paired_reports
from .population import WaveletPopulation, population_key
from .signal_builders import ours_semantic_trace_signal, teacher_hidden_dim_signal
from .utils import SPLIT_NAMES


DEFAULT_STATIC_CLASSIFIERS = ("logreg",)
DEFAULT_SEQUENCE_EPOCHS_QUICK = 3
DEFAULT_STATIC_MAX_ITER = 20000
DEFAULT_TREE_ESTIMATORS = 1000
DEFAULT_XGBOOST_ESTIMATORS = 5000
DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS = 100
DEFAULT_SEQUENCE_MAX_EPOCHS = 200
DEFAULT_SEQUENCE_PATIENCE = 20
DEFAULT_FEATURE_MEMMAP_MIN_BYTES = 512 * 1024 * 1024
YES_NO_TRACE_SOURCE = "final_broadcast"
SAMPLE_GRID_FIELDS = (
    "row_index",
    "population_key",
    "image_id",
    "subset",
    "split",
    "label",
    "row_order_hash",
)
METRIC_LEDGER_EXTRA_FIELDS = (
    "yes_no_trace_source",
    "sequence_model",
    "learning_rate",
    "max_epochs",
    "patience",
    "window_mode",
    "threshold_selection_reason",
    "best_epoch",
    "best_validation_pr_auc",
    "best_params",
    "epochs_ran",
    "early_stopped",
    "converged",
    "max_epoch_reached",
    "training_curve_csv",
    "training_curve_json",
)
FEATURE_MANIFEST_FIELDS = (
    "run_id",
    "model_name",
    "dataset_name",
    "subset_scope",
    "seed",
    "quick_run",
    "block",
    "pair_id",
    "row_id",
    "source",
    "signal_builder",
    "yes_no_trace_source",
    "transform",
    "feature_protocol",
    "wavelet",
    "level",
    "threshold",
    "classifier",
    "sequence_model",
    "window_mode",
    "window_strategy",
    "window_size",
    "stride",
    "mode",
    "cwt_scales",
    "feature_kind",
    "feature_shape",
    "feature_seconds",
    "train_shape",
    "validation_shape",
    "test_shape",
    "train_samples",
    "validation_samples",
    "test_samples",
    "status",
    "failure_reason",
)


@dataclass(frozen=True)
class ReadoutSpec:
    kind: str
    train_name: str
    classifier_name: str
    config_suffix: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitFeatureArrays:
    feature_kind: str
    feature_shape: tuple[int, ...]
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_storage: str = "memory"
    feature_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleGridContract:
    rows: tuple[dict[str, object], ...]
    row_order_hash: str
    path: str = ""


def run_paired_wavelet_experiment(
    *,
    config: Mapping[str, object],
    preflight: Mapping[str, object],
    output_root: Path | str,
    audit_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    features_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run paired v2 feature extraction, readouts, metrics, and reports."""

    started_at = time.perf_counter()
    del cache_dir

    root = Path(output_root)
    audit_path = Path(audit_dir) if audit_dir is not None else root / "audit"
    feature_dir = Path(features_dir) if features_dir is not None else root / "features"
    report_dir = Path(reports_dir) if reports_dir is not None else root / "reports"
    curve_dir = report_dir / "training_curves"
    audit_path.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    quick_run = bool(config.get("quick_run", False))
    paired = _paired_config(config)
    configured_pairs = _resolve_pairs(config, quick_run=False)
    pairs = _resolve_pairs(config, quick_run=quick_run)
    expected_blocks = _ordered_blocks(row.block for row in pairs)
    expected_pair_ids = tuple(dict.fromkeys(row.pair_id for row in pairs))
    configured_blocks = _ordered_blocks(row.block for row in configured_pairs)
    configured_pair_ids = tuple(dict.fromkeys(row.pair_id for row in configured_pairs))
    population = _require_population(preflight)
    configured_sample_grid = _configured_sample_grid_contract_from_preflight(preflight, population)
    configured_sample_grid = _write_sample_grid_contract(
        configured_sample_grid,
        audit_path / "configured_sample_grid.csv",
    )
    entries, labels = _population_entries_and_labels(population, config, quick_run=quick_run)
    sample_grid = _build_sample_grid_contract(entries, labels)
    _validate_selected_sample_grid_against_configured(sample_grid, configured_sample_grid)
    sample_grid = _write_sample_grid_contract(sample_grid, audit_path / "selected_sample_grid.csv")
    _validate_entries_against_sample_grid(entries, labels, sample_grid)
    configured_readouts = _readout_specs(config, quick_run=False)
    readouts = _readout_specs(config, quick_run=quick_run)
    if not readouts:
        raise ValueError("no paired readouts are enabled")
    configured_metric_pair_ids = _expected_metric_pair_ids(configured_pairs, configured_readouts, config)
    expected_pair_ids = _expected_metric_pair_ids(pairs, readouts, config)

    run_id = str(paired.get("run_id", "paired_wavelet_v2"))
    ledger_path = report_dir / "metrics_ledger.csv"
    _remove_stale_run_artifacts(report_dir, feature_dir, curve_dir)
    _reset_metric_ledger(ledger_path)
    rows: list[dict[str, object]] = []
    feature_manifest: list[dict[str, object]] = []
    for pair in pairs:
        static_specs = _static_specs_for_pair(pair, readouts)
        sequence_specs = _sequence_specs_for_pair(pair, readouts)
        unavailable_static_specs = _unavailable_static_specs(static_specs, config)
        for spec in unavailable_static_specs:
            _record_metric_row(
                rows,
                ledger_path,
                _failure_metric_row(
                    _base_metric_row(pair, spec, config, run_id=run_id, quick_run=quick_run),
                    XGBOOST_NOT_INSTALLED,
                    feature_seconds="",
                    train_eval_seconds="",
                    total_seconds="",
                )
            )
        if unavailable_static_specs:
            unavailable_ids = {id(spec) for spec in unavailable_static_specs}
            static_specs = [spec for spec in static_specs if id(spec) not in unavailable_ids]
        if static_specs:
            feature_start = time.perf_counter()
            feature_result = _build_split_features(
                pair,
                config,
                entries,
                labels,
                sample_grid=sample_grid,
                feature_kind="static",
                feature_dir=feature_dir,
            )
            feature_seconds = time.perf_counter() - feature_start
            feature_manifest.append(
                _feature_manifest_row(
                    pair,
                    config,
                    run_id=run_id,
                    quick_run=quick_run,
                    feature_result=feature_result,
                    feature_seconds=feature_seconds,
                    feature_kind="static",
                )
            )
            for spec in static_specs:
                _record_metric_row(
                    rows,
                    ledger_path,
                    _run_static_readout(
                        pair,
                        spec,
                        config,
                        run_id=run_id,
                        quick_run=quick_run,
                        feature_result=feature_result,
                        feature_seconds=feature_seconds,
                    )
                )
            _cleanup_split_feature_arrays(feature_result)
        if sequence_specs:
            feature_start = time.perf_counter()
            feature_result = _build_split_features(
                pair,
                config,
                entries,
                labels,
                sample_grid=sample_grid,
                feature_kind="sequence",
                feature_dir=feature_dir,
            )
            feature_seconds = time.perf_counter() - feature_start
            feature_manifest.append(
                _feature_manifest_row(
                    pair,
                    config,
                    run_id=run_id,
                    quick_run=quick_run,
                    feature_result=feature_result,
                    feature_seconds=feature_seconds,
                    feature_kind="sequence",
                )
            )
            for spec in sequence_specs:
                _record_metric_row(
                    rows,
                    ledger_path,
                    _run_sequence_readout(
                        pair,
                        spec,
                        config,
                        run_id=run_id,
                        quick_run=quick_run,
                        feature_result=feature_result,
                        feature_seconds=feature_seconds,
                        curve_dir=curve_dir,
                    )
                )
            _cleanup_split_feature_arrays(feature_result)

    manifest_path = _write_feature_shape_manifest(
        feature_manifest,
        feature_dir / "feature_shape_manifest.csv",
    )
    ledger_rows = _read_metric_ledger(ledger_path)
    report_preflight = _preflight_with_run_artifacts(
        preflight,
        sample_grid=sample_grid,
        configured_sample_grid=configured_sample_grid,
        metrics_ledger_path=ledger_path,
    )
    report_paths = write_paired_reports(
        ledger_rows,
        report_dir,
        config=config,
        preflight=report_preflight,
        metrics_ledger_path=ledger_path,
        expected_sources=PAIR_SOURCES,
        expected_blocks=expected_blocks,
        expected_pair_ids=expected_pair_ids,
    )
    success_rows = sum(1 for row in ledger_rows if row.get("status") == "success")
    failure_rows = len(ledger_rows) - success_rows
    return {
        "status": "success",
        "training_started": True,
        "quick_run": quick_run,
        "run_id": run_id,
        "configured_grid_rows": len(configured_metric_pair_ids) * len(PAIR_SOURCES),
        "configured_grid_pair_ids": len(configured_metric_pair_ids),
        "configured_base_grid_rows": len(configured_pairs),
        "configured_base_grid_pair_ids": len(configured_pair_ids),
        "configured_grid_blocks": list(configured_blocks),
        "configured_sample_grid_rows": len(configured_sample_grid.rows),
        "configured_sample_grid_row_order_hash": configured_sample_grid.row_order_hash,
        "configured_sample_grid_path": configured_sample_grid.path,
        "selected_run_grid_rows": len(expected_pair_ids) * len(PAIR_SOURCES),
        "selected_run_grid_pair_ids": len(expected_pair_ids),
        "selected_run_grid_blocks": list(expected_blocks),
        "pair_rows": len(expected_pair_ids) * len(PAIR_SOURCES),
        "pair_ids": len(expected_pair_ids),
        "sample_grid_rows": len(sample_grid.rows),
        "sample_grid_row_order_hash": sample_grid.row_order_hash,
        "sample_grid_path": sample_grid.path,
        "metrics_long_rows": len(ledger_rows),
        "success_rows": success_rows,
        "failure_rows": failure_rows,
        "metrics_ledger": str(ledger_path),
        "feature_shape_manifest": str(manifest_path),
        "training_curves_dir": str(curve_dir),
        "report_paths": {key: str(value) for key, value in report_paths.items()},
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def run_paired_experiment(**kwargs: Any) -> dict[str, object]:
    """Alias used by the v2 CLI hook discovery."""

    return run_paired_wavelet_experiment(**kwargs)


def run_experiment(**kwargs: Any) -> dict[str, object]:
    """Alias used by the v2 CLI hook discovery."""

    return run_paired_wavelet_experiment(**kwargs)


def run(**kwargs: Any) -> dict[str, object]:
    """Alias used by the v2 CLI hook discovery."""

    return run_paired_wavelet_experiment(**kwargs)


def _record_metric_row(
    rows: list[dict[str, object]],
    ledger_path: Path,
    row: Mapping[str, object],
) -> None:
    completed = _complete_metric_row(dict(row))
    rows.append(completed)
    _append_metric_ledger_row(ledger_path, completed)


def _reset_metric_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def _remove_stale_run_artifacts(report_dir: Path, feature_dir: Path, curve_dir: Path) -> None:
    """Remove stale success/failure artifacts before a new v2 run starts."""

    for path in (
        report_dir / "metrics_long.csv",
        report_dir / "metrics_wide_paired.csv",
        report_dir / "best_by_block.csv",
        report_dir / "pairwise_winrate.csv",
        report_dir / "failure_report.csv",
        report_dir / "summary.md",
        report_dir / "full_run_status.json",
        report_dir / "metrics_ledger.csv",
        feature_dir / "feature_shape_manifest.csv",
    ):
        if path.exists():
            path.unlink()
    if curve_dir.exists():
        _clear_directory(curve_dir)
    array_dir = feature_dir / "feature_arrays"
    if array_dir.exists():
        _clear_directory(array_dir)


def _clear_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _clear_directory(child)
            child.rmdir()
        else:
            child.unlink()


def _append_metric_ledger_row(path: Path, row: Mapping[str, object]) -> None:
    rows = _read_metric_ledger(path)
    rows.append(dict(row))
    _write_metric_ledger(path, rows)


def _read_metric_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_metric_ledger(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ledger_fieldnames(rows)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_parent_dir(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _ledger_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    fields: list[str] = []
    for field in (*LONG_FIELD_ORDER, *METRIC_LEDGER_EXTRA_FIELDS):
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    return fields


def _fsync_parent_dir(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _run_static_readout(
    pair: PairSpec,
    spec: ReadoutSpec,
    config: Mapping[str, object],
    *,
    run_id: str,
    quick_run: bool,
    feature_result: SplitFeatureArrays | Exception,
    feature_seconds: float,
) -> dict[str, object]:
    train_start = time.perf_counter()
    base = _base_metric_row(pair, spec, config, run_id=run_id, quick_run=quick_run)
    if isinstance(feature_result, Exception):
        return _failure_metric_row(
            base,
            _failure_reason(feature_result),
            feature_seconds=feature_seconds,
            train_eval_seconds="",
            total_seconds=feature_seconds,
        )
    base.update(_split_count_fields(feature_result))
    base["feature_shape"] = _shape_text(feature_result.feature_shape)
    try:
        _require_two_classes(feature_result.train_y, name="train_y")
        train_start = time.perf_counter()
        result = train_static_classifier(
            spec.train_name,
            feature_result.train_x,
            feature_result.train_y,
            validation_x=feature_result.validation_x,
            validation_y=feature_result.validation_y,
            test_x=feature_result.test_x,
            random_state=int(config.get("seed", 0)),
            max_iter=int(spec.params.get("max_iter", DEFAULT_STATIC_MAX_ITER)),
            n_estimators=int(spec.params.get("n_estimators", DEFAULT_TREE_ESTIMATORS)),
            n_jobs=int(spec.params.get("n_jobs", 1)),
            allow_missing_xgboost=bool(config.get("allow_no_xgboost", True)),
            model_params=dict(spec.params.get("model_params", {})),
        )
        train_eval_seconds = time.perf_counter() - train_start
        if result.status != "success" or result.scores is None:
            reason = result.failure_reason or "static_training_failed"
            return _failure_metric_row(
                base,
                reason,
                feature_seconds=feature_seconds,
                train_eval_seconds=train_eval_seconds,
                total_seconds=feature_seconds + train_eval_seconds,
            )
        base["best_params"] = json.dumps(result.best_params, ensure_ascii=True, sort_keys=True)
        if result.best_validation_pr_auc is not None:
            base["best_validation_pr_auc"] = result.best_validation_pr_auc
        return _success_metric_row(
            base,
            labels=feature_result,
            validation_scores=result.scores.validation,
            test_scores=result.scores.test,
            feature_seconds=feature_seconds,
            train_eval_seconds=train_eval_seconds,
            total_seconds=feature_seconds + train_eval_seconds,
        )
    except XGBoostNotInstalledError:
        if spec.train_name == "xgboost" and not bool(config.get("allow_no_xgboost", True)):
            raise
        return _failure_metric_row(
            base,
            XGBOOST_NOT_INSTALLED,
            feature_seconds=feature_seconds,
            train_eval_seconds=time.perf_counter() - train_start,
            total_seconds=feature_seconds + (time.perf_counter() - train_start),
        )
    except Exception as error:
        return _failure_metric_row(
            base,
            _failure_reason(error),
            feature_seconds=feature_seconds,
            train_eval_seconds=time.perf_counter() - train_start,
            total_seconds=feature_seconds + (time.perf_counter() - train_start),
        )


def _run_sequence_readout(
    pair: PairSpec,
    spec: ReadoutSpec,
    config: Mapping[str, object],
    *,
    run_id: str,
    quick_run: bool,
    feature_result: SplitFeatureArrays | Exception,
    feature_seconds: float,
    curve_dir: Path,
) -> dict[str, object]:
    train_start = time.perf_counter()
    base = _base_metric_row(pair, spec, config, run_id=run_id, quick_run=quick_run)
    if isinstance(feature_result, Exception):
        return _failure_metric_row(
            base,
            _failure_reason(feature_result),
            feature_seconds=feature_seconds,
            train_eval_seconds="",
            total_seconds=feature_seconds,
        )
    base.update(_split_count_fields(feature_result))
    base["feature_shape"] = _shape_text(feature_result.feature_shape)
    try:
        _require_two_classes(feature_result.train_y, name="train_y")
        train_start = time.perf_counter()
        result = train_sequence_model(
            spec.train_name,
            feature_result.train_x,
            feature_result.train_y,
            feature_result.validation_x,
            feature_result.validation_y,
            test_x=feature_result.test_x,
            device=_optional_text(config.get("device")),
            batch_size=int(spec.params.get("batch_size", 32)),
            max_epochs=int(spec.params.get("max_epochs", DEFAULT_SEQUENCE_MAX_EPOCHS)),
            patience=int(spec.params.get("patience", DEFAULT_SEQUENCE_PATIENCE)),
            learning_rate=float(spec.params.get("learning_rate", 1e-3)),
            weight_decay=float(spec.params.get("weight_decay", 1e-4)),
            seed=int(config.get("seed", 0)),
            hidden_dim=int(spec.params.get("hidden_dim", 128)),
            dropout=float(spec.params.get("dropout", 0.1)),
        )
        train_eval_seconds = time.perf_counter() - train_start
        curve_paths = _write_training_curve(
            result.training_curve,
            curve_dir,
            config_name=str(base["config_name"]),
            metadata={
                "run_id": run_id,
                "pair_id": base["pair_id"],
                "row_id": base["row_id"],
                "source": pair.source,
                "classifier": spec.classifier_name,
                "learning_rate": result.learning_rate,
                "max_epochs": result.max_epochs,
                "patience": result.patience,
                "best_epoch": result.best_epoch,
                "best_validation_pr_auc": result.best_validation_pr_auc,
                "epochs_ran": result.epochs_ran,
                "early_stopped": result.early_stopped,
                "converged": result.converged,
                "max_epoch_reached": result.max_epoch_reached,
                "quick_run": quick_run,
            },
        )
        base["learning_rate"] = result.learning_rate
        base["max_epochs"] = result.max_epochs
        base["patience"] = result.patience
        base["best_epoch"] = result.best_epoch
        base["best_validation_pr_auc"] = result.best_validation_pr_auc
        base["epochs_ran"] = result.epochs_ran
        base["early_stopped"] = result.early_stopped
        base["converged"] = result.converged
        base["max_epoch_reached"] = result.max_epoch_reached
        base["training_curve_csv"] = str(curve_paths["csv"])
        base["training_curve_json"] = str(curve_paths["json"])
        return _success_metric_row(
            base,
            labels=feature_result,
            validation_scores=result.scores.validation,
            test_scores=result.scores.test,
            feature_seconds=feature_seconds,
            train_eval_seconds=train_eval_seconds,
            total_seconds=feature_seconds + train_eval_seconds,
        )
    except Exception as error:
        return _failure_metric_row(
            base,
            _failure_reason(error),
            feature_seconds=feature_seconds,
            train_eval_seconds=time.perf_counter() - train_start,
            total_seconds=feature_seconds + (time.perf_counter() - train_start),
        )


def _build_split_features(
    pair: PairSpec,
    config: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    *,
    sample_grid: SampleGridContract,
    feature_kind: str,
    feature_dir: Path,
) -> SplitFeatureArrays | Exception:
    try:
        _validate_entries_against_sample_grid(entries, labels, sample_grid)
        if not entries:
            raise ValueError(f"{pair.row_id}: cannot build features for an empty population")
        split_names = [str(entry.get("wavelet_split", "")) for entry in entries]
        split_indices = _split_indices_by_name(split_names)

        first_row = _feature_row_for_entry(pair, config, entries[0], feature_kind=feature_kind)
        row_shape = tuple(int(dim) for dim in first_row.shape)
        _validate_feature_row(first_row, feature_kind=feature_kind, context=pair.row_id)

        feature_shape = (len(entries), *row_shape)
        use_memmap = _should_memmap_features(
            feature_shape,
            config=config,
        )
        arrays = _allocate_split_feature_arrays(
            pair,
            feature_dir=feature_dir,
            split_indices=split_indices,
            row_shape=row_shape,
            use_memmap=use_memmap,
        )
        label_arrays = {
            split: np.asarray(labels[indices], dtype=np.int64)
            for split, indices in split_indices.items()
        }
        write_positions = {split: 0 for split in SPLIT_NAMES}

        for row_index, entry in enumerate(entries):
            row = first_row if row_index == 0 else _feature_row_for_entry(
                pair,
                config,
                entry,
                feature_kind=feature_kind,
            )
            row = np.asarray(row, dtype=np.float32)
            if tuple(row.shape) != row_shape:
                raise ValueError(
                    f"{pair.row_id}: feature shape changed at row {row_index}; "
                    f"expected {row_shape}, got {tuple(row.shape)}"
                )
            _validate_feature_row(row, feature_kind=feature_kind, context=pair.row_id)
            split = str(entry.get("wavelet_split", ""))
            position = write_positions[split]
            arrays[split][position] = row
            write_positions[split] = position + 1

        for split, expected_count in ((split, len(split_indices[split])) for split in SPLIT_NAMES):
            if write_positions[split] != expected_count:
                raise ValueError(
                    f"{pair.row_id}: split {split!r} write count mismatch; "
                    f"wrote={write_positions[split]} expected={expected_count}"
                )
            _flush_memmap(arrays[split])
            _validate_split_feature_array(
                arrays[split],
                feature_kind=feature_kind,
                expected_samples=expected_count,
                context=f"{pair.row_id} {split}",
                finite_already_checked=True,
            )

        return SplitFeatureArrays(
            feature_kind=feature_kind,
            feature_shape=feature_shape,
            train_x=arrays["train"],
            train_y=label_arrays["train"],
            validation_x=arrays["validation"],
            validation_y=label_arrays["validation"],
            test_x=arrays["test"],
            test_y=label_arrays["test"],
            feature_storage="memmap" if use_memmap else "memory",
            feature_paths=tuple(
                _memmap_path(arrays[split])
                for split in SPLIT_NAMES
                if _memmap_path(arrays[split])
            ),
        )
    except Exception as error:
        return error


def _feature_row_for_entry(
    pair: PairSpec,
    config: Mapping[str, object],
    entry: Mapping[str, object],
    *,
    feature_kind: str,
) -> np.ndarray:
    signal = _build_signal(pair, entry, config)
    if feature_kind == "static":
        row = features_for_protocol(signal, pair, epsilon=_epsilon(config))
    elif feature_kind == "sequence":
        row = _sequence_features_for_signal(signal, pair)
    else:
        raise ValueError(f"unsupported feature_kind={feature_kind!r}")
    return np.asarray(row, dtype=np.float32)


def _split_indices_by_name(split_names: Sequence[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        indices = np.asarray(
            [index for index, value in enumerate(split_names) if value == split],
            dtype=np.int64,
        )
        if indices.shape[0] == 0:
            raise ValueError(f"missing required splits: {[split]}")
        arrays[split] = indices
    unknown = sorted(set(split_names) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"unknown split names in feature entries: {unknown}")
    return arrays


def _should_memmap_features(
    feature_shape: Sequence[int],
    *,
    config: Mapping[str, object],
) -> bool:
    min_bytes = int(config.get("feature_memmap_min_bytes", DEFAULT_FEATURE_MEMMAP_MIN_BYTES))
    if min_bytes < 0:
        raise ValueError("feature_memmap_min_bytes must be non-negative")
    total_values = int(np.prod(np.asarray(tuple(feature_shape), dtype=np.int64)))
    total_bytes = total_values * np.dtype(np.float32).itemsize
    return total_bytes >= min_bytes


def _allocate_split_feature_arrays(
    pair: PairSpec,
    *,
    feature_dir: Path,
    split_indices: Mapping[str, np.ndarray],
    row_shape: Sequence[int],
    use_memmap: bool,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        shape = (int(split_indices[split].shape[0]), *tuple(int(dim) for dim in row_shape))
        if use_memmap:
            array_dir = feature_dir / "feature_arrays"
            array_dir.mkdir(parents=True, exist_ok=True)
            path = array_dir / f"{_safe_filename(pair.row_id)}.{split}.float32.mmap"
            if path.exists():
                path.unlink()
            arrays[split] = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
        else:
            arrays[split] = np.empty(shape, dtype=np.float32)
    return arrays


def _memmap_path(array: np.ndarray) -> str:
    if isinstance(array, np.memmap) and array.filename is not None:
        return str(array.filename)
    return ""


def _cleanup_split_feature_arrays(feature_result: SplitFeatureArrays | Exception) -> None:
    if isinstance(feature_result, Exception):
        return
    arrays = (feature_result.train_x, feature_result.validation_x, feature_result.test_x)
    for array in arrays:
        if isinstance(array, np.memmap):
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
    for raw_path in feature_result.feature_paths:
        path = Path(raw_path)
        if path.exists():
            path.unlink()


def _flush_memmap(array: np.ndarray) -> None:
    if isinstance(array, np.memmap):
        array.flush()


def _validate_feature_row(row: np.ndarray, *, feature_kind: str, context: str) -> None:
    if feature_kind == "static":
        if row.ndim != 1:
            raise ValueError(f"{context}: static feature row must be 1D")
        if row.shape[0] == 0:
            raise ValueError(f"{context}: static feature row must be non-empty")
    elif feature_kind == "sequence":
        if row.ndim != 2:
            raise ValueError(f"{context}: sequence feature row must be 2D")
        if row.shape[0] == 0 or row.shape[1] == 0:
            raise ValueError(f"{context}: sequence feature row must be non-empty")
    else:
        raise ValueError(f"unsupported feature_kind={feature_kind!r}")
    _raise_if_non_finite(row, name=f"{context} feature row")


def _validate_split_feature_array(
    features: np.ndarray,
    *,
    feature_kind: str,
    expected_samples: int,
    context: str,
    finite_already_checked: bool = False,
) -> None:
    if feature_kind == "static":
        _validate_static_feature_matrix(
            features,
            expected_samples=expected_samples,
            context=context,
            finite_already_checked=finite_already_checked,
        )
    elif feature_kind == "sequence":
        _validate_sequence_feature_array(
            features,
            expected_samples=expected_samples,
            context=context,
            finite_already_checked=finite_already_checked,
        )
    else:
        raise ValueError(f"unsupported feature_kind={feature_kind!r}")


def _build_signal(pair: PairSpec, entry: Mapping[str, object], config: Mapping[str, object]) -> np.ndarray:
    expected_num_layers = _optional_int(config.get("expected_num_layers"))
    expected_hidden_dim = _optional_int(config.get("expected_hidden_dim"))
    if "layer_vectors" not in entry:
        raise ValueError("entry is missing layer_vectors")
    if pair.source == "Teacher":
        return teacher_hidden_dim_signal(
            entry["layer_vectors"],
            expected_num_layers=expected_num_layers,
            expected_hidden_dim=expected_hidden_dim,
        )
    if pair.source == "Ours":
        explicit_yes = _first_present(entry, ("final_yes_logit", "yes_logit"))
        explicit_no = _first_present(entry, ("final_no_logit", "no_logit"))
        if explicit_yes is not None or explicit_no is not None:
            return ours_semantic_trace_signal(
                entry["layer_vectors"],
                final_yes_logit=explicit_yes,
                final_no_logit=explicit_no,
                expected_num_layers=expected_num_layers,
                expected_hidden_dim=expected_hidden_dim,
                epsilon=_epsilon(config),
            )
        final_logits = entry.get("first_token_logits", entry.get("final_logits"))
        ours = _resolved_ours_signal_config(config, final_logits=final_logits)
        return ours_semantic_trace_signal(
            entry["layer_vectors"],
            final_logits=final_logits,
            yes_token_id=_optional_int(ours.get("yes_token_id")),
            no_token_id=_optional_int(ours.get("no_token_id")),
            expected_num_layers=expected_num_layers,
            expected_hidden_dim=expected_hidden_dim,
            epsilon=_epsilon(config),
        )
    raise ValueError(f"unsupported pair source={pair.source!r}")


def _sequence_features_for_signal(signal: np.ndarray, pair: PairSpec) -> np.ndarray:
    if pair.feature_protocol == "window_stat28_sequence":
        values = features_for_protocol(signal, pair)
        if values.ndim != 2:
            raise ValueError("window_stat28_sequence must produce a 2D sequence array")
        _raise_if_non_finite(values, name="window_stat28_sequence")
        return values.astype(np.float32, copy=False)
    if pair.feature_protocol == "raw_sequence":
        values = features_for_protocol(signal, pair)
        if values.ndim != 2:
            raise ValueError("raw_sequence must produce a 2D sequence array")
        _raise_if_non_finite(values, name="raw_sequence")
        return values.astype(np.float32, copy=False)
    values = features_for_protocol(signal, pair).reshape(1, -1)
    return values.astype(np.float32, copy=False)


def _split_feature_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    entries: Sequence[Mapping[str, object]],
    feature_kind: str,
) -> SplitFeatureArrays:
    if features.shape[0] != labels.shape[0] or features.shape[0] != len(entries):
        raise ValueError("features, labels, and entries must have matching sample counts")
    split_names = np.asarray([str(entry.get("wavelet_split", "")) for entry in entries])
    masks = {split: split_names == split for split in SPLIT_NAMES}
    missing = [split for split, mask in masks.items() if not bool(mask.any())]
    if missing:
        raise ValueError(f"missing required splits: {missing}")
    return SplitFeatureArrays(
        feature_kind=feature_kind,
        feature_shape=tuple(int(dim) for dim in features.shape),
        train_x=features[masks["train"]],
        train_y=labels[masks["train"]],
        validation_x=features[masks["validation"]],
        validation_y=labels[masks["validation"]],
        test_x=features[masks["test"]],
        test_y=labels[masks["test"]],
    )


def _configured_sample_grid_contract_from_preflight(
    preflight: Mapping[str, object],
    population: WaveletPopulation,
) -> SampleGridContract:
    entries = [dict(entry) for entry in population.primary_entries]
    labels = np.asarray(population.labels, dtype=np.int64)
    audit = preflight.get("sample_grid_audit")
    if isinstance(audit, Mapping):
        path = str(audit.get("configured_sample_grid_path", "") or audit.get("sample_grid_path", "") or "")
        if path:
            sample_path = Path(path)
            if sample_path.is_file():
                contract = _read_sample_grid_contract(sample_path)
                _validate_entries_against_sample_grid(entries, labels, contract)
                return contract
    return _build_sample_grid_contract(entries, labels)


def _read_sample_grid_contract(path: Path) -> SampleGridContract:
    rows: list[dict[str, object]] = []
    hash_values: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "population_key": str(row["population_key"]),
                    "image_id": str(row["image_id"]),
                    "subset": str(row["subset"]),
                    "split": str(row["split"]),
                    "label": int(row["label"]),
                }
            )
            hash_values.append(str(row.get("row_order_hash", "") or "").strip())
    if not rows:
        raise ValueError(f"{path}: sample grid must not be empty")
    unique_hashes = set(hash_values)
    if any(not value for value in hash_values) or len(unique_hashes) != 1:
        raise ValueError(f"{path}: every row must contain the same non-empty row_order_hash")
    row_order_hash = next(iter(unique_hashes))
    computed = _sample_grid_row_order_hash(rows)
    if computed != row_order_hash:
        raise ValueError(
            f"{path}: row_order_hash mismatch; stored={row_order_hash} computed={computed}"
        )
    return SampleGridContract(rows=tuple(rows), row_order_hash=row_order_hash, path=str(path))


def _build_sample_grid_contract(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
) -> SampleGridContract:
    rows = _sample_grid_rows_from_entries(entries, labels)
    return SampleGridContract(
        rows=tuple(rows),
        row_order_hash=_sample_grid_row_order_hash(rows),
    )


def _write_sample_grid_contract(contract: SampleGridContract, path: Path) -> SampleGridContract:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_GRID_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in contract.rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in SAMPLE_GRID_FIELDS},
                    "row_order_hash": contract.row_order_hash,
                }
            )
    return SampleGridContract(
        rows=contract.rows,
        row_order_hash=contract.row_order_hash,
        path=str(path),
    )


def _validate_selected_sample_grid_against_configured(
    selected: SampleGridContract,
    configured: SampleGridContract,
) -> None:
    configured_by_key: dict[str, Mapping[str, object]] = {}
    for row in configured.rows:
        key = str(row["population_key"])
        if key in configured_by_key:
            raise ValueError(f"duplicate configured sample grid key: {key}")
        configured_by_key[key] = row
    seen: set[str] = set()
    for row in selected.rows:
        key = str(row["population_key"])
        if key in seen:
            raise ValueError(f"duplicate selected sample grid key: {key}")
        seen.add(key)
        configured_row = configured_by_key.get(key)
        if configured_row is None:
            raise ValueError(f"selected sample grid key is missing from configured grid: {key}")
        for field in ("image_id", "subset", "split", "label"):
            if str(row[field]) != str(configured_row[field]):
                raise ValueError(
                    f"selected sample grid drift for population_key={key}, field={field}: "
                    f"selected={row[field]} configured={configured_row[field]}"
                )


def _preflight_with_run_artifacts(
    preflight: Mapping[str, object],
    *,
    sample_grid: SampleGridContract,
    configured_sample_grid: SampleGridContract,
    metrics_ledger_path: Path,
) -> dict[str, object]:
    updated = dict(preflight)
    sample_audit = dict(updated.get("sample_grid_audit", {}) or {})
    sample_audit.update(
        {
            "sample_grid_path": sample_grid.path,
            "selected_sample_grid_path": sample_grid.path,
            "configured_sample_grid_path": configured_sample_grid.path,
            "num_rows": len(sample_grid.rows),
            "selected_num_rows": len(sample_grid.rows),
            "configured_num_rows": len(configured_sample_grid.rows),
            "row_order_hash": sample_grid.row_order_hash,
            "selected_row_order_hash": sample_grid.row_order_hash,
            "configured_row_order_hash": configured_sample_grid.row_order_hash,
        }
    )
    updated["sample_grid_audit"] = sample_audit
    updated["metrics_ledger_path"] = str(metrics_ledger_path)
    return updated


def _sample_grid_rows_from_entries(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
) -> list[dict[str, object]]:
    if labels.ndim != 1 or labels.shape[0] != len(entries):
        raise ValueError("row-count mismatch: entries and labels must match before sample grid validation")
    rows: list[dict[str, object]] = []
    for index, (entry, label) in enumerate(zip(entries, labels, strict=True)):
        rows.append(
            {
                "row_index": index,
                "population_key": _entry_population_key(entry),
                "image_id": _required_entry_text(entry, "image_id", index),
                "subset": _required_entry_text(entry, "subset", index),
                "split": _required_entry_text(entry, "wavelet_split", index),
                "label": int(label),
            }
        )
    return rows


def _validate_entries_against_sample_grid(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    sample_grid: SampleGridContract,
) -> None:
    if labels.ndim != 1 or labels.shape[0] != len(entries):
        raise ValueError("row-count mismatch: entries and labels must have matching lengths")
    if len(entries) != len(sample_grid.rows):
        raise ValueError(
            f"row-count mismatch: feature input has {len(entries)} rows, "
            f"sample grid has {len(sample_grid.rows)} rows"
        )
    expected_by_key: dict[str, Mapping[str, object]] = {}
    for row in sample_grid.rows:
        key = str(row["population_key"])
        if key in expected_by_key:
            raise ValueError(f"duplicate sample grid key in contract: {key}")
        expected_by_key[key] = row

    actual_rows = _sample_grid_rows_from_entries(entries, labels)
    seen: set[str] = set()
    for row in actual_rows:
        key = str(row["population_key"])
        if key in seen:
            raise ValueError(f"duplicate sample grid key in feature input: {key}")
        seen.add(key)
        expected = expected_by_key.get(key)
        if expected is None:
            raise ValueError(f"missing sample grid key in contract: {key}")
        if str(row["split"]) != str(expected["split"]):
            raise ValueError(
                f"split drift for population_key={key}: "
                f"feature_input={row['split']} sample_grid={expected['split']}"
            )
        if int(row["label"]) != int(expected["label"]):
            raise ValueError(
                f"label drift for population_key={key}: "
                f"feature_input={row['label']} sample_grid={expected['label']}"
            )
    computed_hash = _sample_grid_row_order_hash(actual_rows)
    if computed_hash != sample_grid.row_order_hash:
        raise ValueError(
            f"row-order hash drift: feature_input={computed_hash} "
            f"sample_grid={sample_grid.row_order_hash}"
        )


def _sample_grid_row_order_hash(rows: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "row_index": int(row["row_index"]),
            "population_key": str(row["population_key"]),
            "image_id": str(row["image_id"]),
            "subset": str(row["subset"]),
            "split": str(row["split"]),
            "label": int(row["label"]),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _entry_population_key(entry: Mapping[str, object]) -> str:
    value = entry.get("wavelet_population_key")
    if value is not None and str(value).strip():
        return str(value)
    return population_key(entry)


def _required_entry_text(entry: Mapping[str, object], field: str, index: int) -> str:
    value = entry.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"sample grid row {index} missing {field}")
    return str(value)


def _success_metric_row(
    base: dict[str, object],
    *,
    labels: SplitFeatureArrays,
    validation_scores: np.ndarray | None,
    test_scores: np.ndarray | None,
    feature_seconds: object,
    train_eval_seconds: object,
    total_seconds: object,
) -> dict[str, object]:
    if validation_scores is None or test_scores is None:
        raise ValueError("validation and test scores are required")
    evaluation = evaluate_validation_test(
        labels.validation_y,
        validation_scores,
        labels.test_y,
        test_scores,
    )
    test_metrics = evaluation["test"]
    row = dict(base)
    row.update(
        {
            "pr_auc": test_metrics["pr_auc"],
            "average_precision": test_metrics["average_precision"],
            "roc_auc": test_metrics["roc_auc"],
            "best_val_threshold": evaluation["threshold"],
            "threshold_selection_reason": evaluation.get("threshold_selection_reason", ""),
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "tpr_at_1pct_fpr": test_metrics["tpr_at_1pct_fpr"],
            "fpr_at_95pct_tpr": test_metrics["fpr_at_95pct_tpr"],
            "status": "success",
            "failure_reason": "",
            "feature_seconds": feature_seconds,
            "train_eval_seconds": train_eval_seconds,
            "total_seconds": total_seconds,
        }
    )
    return _complete_metric_row(row)


def _failure_metric_row(
    base: Mapping[str, object],
    failure_reason: str,
    *,
    feature_seconds: object,
    train_eval_seconds: object,
    total_seconds: object,
) -> dict[str, object]:
    row = dict(base)
    row.update(
        {
            "status": "failure",
            "failure_reason": failure_reason,
            "feature_seconds": feature_seconds,
            "train_eval_seconds": train_eval_seconds,
            "total_seconds": total_seconds,
        }
    )
    return _complete_metric_row(row)


def _complete_metric_row(row: dict[str, object]) -> dict[str, object]:
    defaults = {
        "train_samples": "",
        "val_samples": "",
        "test_samples": "",
        "train_pos": "",
        "val_pos": "",
        "test_pos": "",
        "feature_shape": "",
        "pr_auc": "",
        "average_precision": "",
        "roc_auc": "",
        "best_val_threshold": "",
        "threshold_selection_reason": "",
        "test_f1": "",
        "test_precision": "",
        "test_recall": "",
        "balanced_accuracy": "",
        "tpr_at_1pct_fpr": "",
        "fpr_at_95pct_tpr": "",
        "feature_seconds": "",
        "train_eval_seconds": "",
        "total_seconds": "",
        "learning_rate": "",
        "max_epochs": "",
        "patience": "",
        "best_epoch": "",
        "best_validation_pr_auc": "",
        "epochs_ran": "",
        "early_stopped": "",
        "converged": "",
        "max_epoch_reached": "",
    }
    for metric_name in METRIC_NAMES:
        defaults.setdefault(metric_name, "")
    for key, value in defaults.items():
        row.setdefault(key, value)
    return row


def _base_metric_row(
    pair: PairSpec,
    spec: ReadoutSpec,
    config: Mapping[str, object],
    *,
    run_id: str,
    quick_run: bool,
) -> dict[str, object]:
    suffix = spec.config_suffix
    metric_pair_id = _metric_pair_id(pair, spec)
    metric_row_id = _metric_row_id(pair, spec)
    return {
        "run_id": run_id,
        "model_name": config.get("model_name", ""),
        "dataset_name": config.get("dataset_name", ""),
        "subset_scope": ",".join(str(item) for item in config.get("subsets", [])),  # type: ignore[arg-type]
        "seed": config.get("seed", ""),
        "quick_run": quick_run,
        "block": pair.block,
        "pair_id": metric_pair_id,
        "row_id": metric_row_id,
        "source": pair.source,
        "signal_builder": pair.signal_builder,
        "yes_no_trace_source": _yes_no_trace_source(pair, config),
        "config_name": f"{metric_row_id}::{suffix}",
        "method_family": _method_family_for_pair(pair),
        "classifier": pair.classifier or spec.classifier_name,
        "sequence_model": pair.sequence_model or "",
        "learning_rate": spec.params.get("learning_rate", ""),
        "max_epochs": spec.params.get("max_epochs", ""),
        "patience": spec.params.get("patience", ""),
        "transform": pair.transform,
        "feature_protocol": pair.feature_protocol,
        "wavelet": pair.wavelet or "",
        "level": pair.level or "",
        "threshold": pair.threshold,
        "window_mode": pair.window_mode,
        "window_strategy": pair.window_strategy,
        "window_size": pair.window_size or "",
        "stride": pair.stride or "",
        "mode": pair.mode,
        "cwt_scales": list(pair.cwt_scales),
    }


def _metric_pair_id(pair: PairSpec, spec: ReadoutSpec) -> str:
    """Return the final paired identity used in metrics rows.

    The base grid keeps one pair for each controlled wavelet/window/readout
    slot. Some readouts still expand internally into named variants, such as
    the sequence learning-rate grid. Each such variant is a distinct experiment
    configuration, so it needs its own final pair_id with exactly Teacher/Ours
    rows in metrics_long.
    """

    if spec.kind == "sequence" and spec.config_suffix != (pair.sequence_model or spec.train_name):
        return f"{pair.pair_id}__{spec.config_suffix}"
    if spec.kind == "static" and pair.classifier is not None and spec.config_suffix != pair.classifier:
        return f"{pair.pair_id}__{spec.config_suffix}"
    return pair.pair_id


def _metric_row_id(pair: PairSpec, spec: ReadoutSpec) -> str:
    return f"{_metric_pair_id(pair, spec)}::{pair.source}"


def _method_family_for_pair(pair: PairSpec) -> str:
    if pair.source == "Teacher":
        return "teacher_bagua"
    if pair.source == "Ours":
        return "ours_wavelet"
    raise ValueError(f"unsupported paired source: {pair.source!r}")


def _split_count_fields(features: SplitFeatureArrays) -> dict[str, object]:
    return {
        "train_samples": int(features.train_y.shape[0]),
        "val_samples": int(features.validation_y.shape[0]),
        "test_samples": int(features.test_y.shape[0]),
        "train_pos": int(np.sum(features.train_y == 1)),
        "val_pos": int(np.sum(features.validation_y == 1)),
        "test_pos": int(np.sum(features.test_y == 1)),
    }


def _feature_manifest_row(
    pair: PairSpec,
    config: Mapping[str, object],
    *,
    run_id: str,
    quick_run: bool,
    feature_result: SplitFeatureArrays | Exception,
    feature_seconds: float,
    feature_kind: str,
) -> dict[str, object]:
    row = {
        "run_id": run_id,
        "model_name": config.get("model_name", ""),
        "dataset_name": config.get("dataset_name", ""),
        "subset_scope": ",".join(str(item) for item in config.get("subsets", [])),  # type: ignore[arg-type]
        "seed": config.get("seed", ""),
        "quick_run": quick_run,
        "block": pair.block,
        "pair_id": pair.pair_id,
        "row_id": pair.row_id,
        "source": pair.source,
        "signal_builder": pair.signal_builder,
        "yes_no_trace_source": _yes_no_trace_source(pair, config),
        "transform": pair.transform,
        "feature_protocol": pair.feature_protocol,
        "wavelet": pair.wavelet or "",
        "level": pair.level or "",
        "threshold": pair.threshold,
        "classifier": pair.classifier or "",
        "sequence_model": pair.sequence_model or "",
        "window_mode": pair.window_mode,
        "window_strategy": pair.window_strategy,
        "window_size": pair.window_size or "",
        "stride": pair.stride or "",
        "mode": pair.mode,
        "cwt_scales": list(pair.cwt_scales),
        "feature_kind": feature_kind,
        "feature_seconds": feature_seconds,
        "status": "success",
        "failure_reason": "",
    }
    if isinstance(feature_result, Exception):
        row["status"] = "failure"
        row["failure_reason"] = _failure_reason(feature_result)
        return row
    row.update(
        {
            "feature_kind": feature_result.feature_kind,
            "feature_shape": _shape_text(feature_result.feature_shape),
            "train_shape": _shape_text(feature_result.train_x.shape),
            "validation_shape": _shape_text(feature_result.validation_x.shape),
            "test_shape": _shape_text(feature_result.test_x.shape),
            "train_samples": int(feature_result.train_y.shape[0]),
            "validation_samples": int(feature_result.validation_y.shape[0]),
            "test_samples": int(feature_result.test_y.shape[0]),
            "feature_storage": feature_result.feature_storage,
            "feature_paths": list(feature_result.feature_paths),
        }
    )
    return row


def _write_feature_shape_manifest(rows: Sequence[Mapping[str, object]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({field for row in rows for field in row if field not in FEATURE_MANIFEST_FIELDS})
    fields = list(FEATURE_MANIFEST_FIELDS) + extra_fields
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
    return output


def _write_training_curve(
    curve: Sequence[Mapping[str, float]],
    output_dir: Path,
    *,
    config_name: str,
    metadata: Mapping[str, object],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(config_name)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    fields = sorted({field for row in curve for field in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in curve:
            writer.writerow({field: row.get(field, "") for field in fields})
    payload = {"metadata": dict(metadata), "curve": [dict(row) for row in curve]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def _resolve_pairs(config: Mapping[str, object], *, quick_run: bool) -> tuple[PairSpec, ...]:
    paired = _paired_config(config)
    raw_pairs = paired.get("pairs")
    if isinstance(raw_pairs, Sequence) and not isinstance(raw_pairs, (str, bytes)):
        pairs = tuple(pair_spec_from_mapping(row) for row in raw_pairs if isinstance(row, Mapping))
    else:
        blocks = paired.get("blocks")
        pairs = build_paired_grid(blocks=_coerce_blocks(blocks) if blocks is not None else None)
    pairs = assert_paired_grid_complete(
        pairs,
        expected_blocks=_ordered_blocks(row.block for row in pairs),
        expected_sources=tuple(str(source) for source in paired.get("expected_sources", PAIR_SOURCES)),  # type: ignore[arg-type]
    )
    if quick_run:
        pairs = _limit_pairs_for_quick(pairs, config)
    return _assert_selected_pairs_complete(
        pairs,
        expected_sources=tuple(str(source) for source in paired.get("expected_sources", PAIR_SOURCES)),  # type: ignore[arg-type]
    )


def _assert_selected_pairs_complete(
    rows: Sequence[PairSpec],
    *,
    expected_sources: Sequence[str],
) -> tuple[PairSpec, ...]:
    pairs = tuple(rows)
    if not pairs:
        raise ValueError("paired grid selection must not be empty")
    expected_source_set = set(str(source) for source in expected_sources)
    grouped: dict[str, list[PairSpec]] = {}
    seen: set[tuple[str, str]] = set()
    for row in pairs:
        key = (row.pair_id, row.source)
        if key in seen:
            raise ValueError(f"duplicate paired grid row for pair_id={row.pair_id!r}, source={row.source!r}")
        seen.add(key)
        grouped.setdefault(row.pair_id, []).append(row)
    for pair_id, pair_rows in grouped.items():
        sources = {row.source for row in pair_rows}
        if sources != expected_source_set:
            raise ValueError(
                f"pair_id={pair_id!r} must have exact sources {sorted(expected_source_set)}, "
                f"got {sorted(sources)}"
            )
        keys = {row.paired_key() for row in pair_rows}
        if len(keys) != 1:
            raise ValueError(f"pair_id={pair_id!r} Teacher/Ours rows are not config-matched")
    return pairs


def _limit_pairs_for_quick(pairs: Sequence[PairSpec], config: Mapping[str, object]) -> tuple[PairSpec, ...]:
    quick = _quick_config(config)
    blocks = quick.get("blocks")
    filtered = tuple(pairs)
    paired = _paired_config(config)
    if blocks is not None and str(paired.get("block_source", "")) != "cli":
        allowed_blocks = set(_coerce_blocks(blocks))
        filtered = tuple(pair for pair in filtered if pair.block in allowed_blocks)
    max_pair_ids = _first_int(quick, ("max_pair_ids", "max_pairs"))
    if max_pair_ids is not None:
        selected_ids = tuple(dict.fromkeys(pair.pair_id for pair in filtered))[:max_pair_ids]
        allowed_ids = set(selected_ids)
        filtered = tuple(pair for pair in filtered if pair.pair_id in allowed_ids)
    if not filtered:
        raise ValueError("quick mode removed all paired rows")
    return filtered


def _population_entries_and_labels(
    population: WaveletPopulation,
    config: Mapping[str, object],
    *,
    quick_run: bool,
) -> tuple[list[dict[str, object]], np.ndarray]:
    entries = [dict(entry) for entry in population.primary_entries]
    labels = np.asarray(population.labels, dtype=np.int64)
    if not entries:
        raise ValueError("WaveletPopulation.primary_entries must not be empty")
    if labels.ndim != 1 or labels.shape[0] != len(entries):
        raise ValueError("WaveletPopulation labels must match primary_entries")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("WaveletPopulation labels must contain only 0/1 values")
    if quick_run:
        return _limit_entries_for_quick(entries, labels, config)
    return entries, labels


def _limit_entries_for_quick(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], np.ndarray]:
    quick = _quick_config(config)
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


def _readout_specs(config: Mapping[str, object], *, quick_run: bool) -> tuple[ReadoutSpec, ...]:
    specs = [*_static_readout_specs(config), *_sequence_readout_specs(config)]
    if quick_run:
        allowed = _quick_allowed_readouts(config)
        if allowed:
            specs = [
                spec
                for spec in specs
                if _readout_allowed_in_quick(spec, allowed)
            ]
    return tuple(specs)


def _static_specs_for_pair(pair: PairSpec, specs: Sequence[ReadoutSpec]) -> list[ReadoutSpec]:
    static_specs = [spec for spec in specs if spec.kind == "static"]
    if pair.sequence_model is not None:
        return []
    if pair.classifier is None:
        return static_specs
    return [
        spec
        for spec in static_specs
        if _readout_name_matches(spec, pair.classifier)
    ]


def _sequence_specs_for_pair(pair: PairSpec, specs: Sequence[ReadoutSpec]) -> list[ReadoutSpec]:
    sequence_specs = [spec for spec in specs if spec.kind == "sequence"]
    if pair.classifier is not None:
        return []
    if pair.sequence_model is None:
        return sequence_specs
    return [
        spec
        for spec in sequence_specs
        if _readout_name_matches(spec, pair.sequence_model)
    ]


def _metric_specs_for_pair(pair: PairSpec, readouts: Sequence[ReadoutSpec]) -> tuple[ReadoutSpec, ...]:
    return (
        *_static_specs_for_pair(pair, readouts),
        *_sequence_specs_for_pair(pair, readouts),
    )


def _expected_metric_pair_ids(
    pairs: Sequence[PairSpec],
    readouts: Sequence[ReadoutSpec],
    config: Mapping[str, object],
) -> tuple[str, ...]:
    del config
    pair_ids: list[str] = []
    for pair in pairs:
        for spec in _metric_specs_for_pair(pair, readouts):
            pair_ids.append(_metric_pair_id(pair, spec))
    if not pair_ids:
        raise ValueError("paired metric grid must not be empty")
    return tuple(dict.fromkeys(pair_ids))


def _unavailable_static_specs(specs: Sequence[ReadoutSpec], config: Mapping[str, object]) -> list[ReadoutSpec]:
    unavailable: list[ReadoutSpec] = []
    for spec in specs:
        if spec.train_name != "xgboost" or not _xgboost_is_missing():
            continue
        if not bool(config.get("allow_no_xgboost", True)):
            raise XGBoostNotInstalledError("xgboost is not installed")
        unavailable.append(spec)
    return unavailable


def _xgboost_is_missing() -> bool:
    sentinel = object()
    if sys.modules.get("xgboost", sentinel) is None:
        return True
    try:
        module = importlib.import_module("xgboost")
    except ImportError:
        return True
    return not hasattr(module, "XGBClassifier")


def _static_readout_specs(config: Mapping[str, object]) -> tuple[ReadoutSpec, ...]:
    if "classifiers" in config:
        classifier_cfg = dict(config.get("classifiers", {}) or {})
    else:
        classifier_cfg = {name: {"enabled": True} for name in DEFAULT_STATIC_CLASSIFIERS}
    specs: list[ReadoutSpec] = []
    for raw_name, raw_section in classifier_cfg.items():
        name = _normalize_static_name(str(raw_name))
        if name is None:
            continue
        section = _section_mapping(raw_section)
        if not bool(section.get("enabled", True)):
            continue
        if name == "xgboost":
            specs.extend(_xgboost_readout_specs(section))
        else:
            specs.append(
                ReadoutSpec(
                    kind="static",
                    train_name=name,
                    classifier_name=name,
                    config_suffix=name,
                    params=_static_params(name, section),
                )
            )
    return tuple(specs)


def _sequence_readout_specs(config: Mapping[str, object]) -> tuple[ReadoutSpec, ...]:
    sections: dict[str, Any] = {}
    for key in ("sequence_models", "sequence_readouts"):
        value = config.get(key)
        if isinstance(value, Mapping):
            sections.update(dict(value))
    paired = config.get("paired_wavelet_v2")
    if isinstance(paired, Mapping):
        for key in ("sequence_models", "sequence_readouts"):
            value = paired.get(key)
            if isinstance(value, Mapping):
                sections.update(dict(value))
    classifiers = config.get("classifiers")
    if isinstance(classifiers, Mapping):
        for name in SEQUENCE_MODELS:
            if name in classifiers:
                sections[name] = classifiers[name]
    specs: list[ReadoutSpec] = []
    for raw_name, raw_section in sections.items():
        name = str(raw_name).strip().lower()
        if name not in SEQUENCE_MODELS:
            continue
        section = _section_mapping(raw_section)
        if not bool(section.get("enabled", True)):
            continue
        learning_rates = [float(value) for value in _as_grid_values(section.get("learning_rate", 1e-3))]
        if not learning_rates:
            raise ValueError(f"sequence model {name} must define at least one learning_rate")
        for learning_rate in learning_rates:
            params = {
                "batch_size": int(section.get("batch_size", 32)),
                "max_epochs": int(section.get("max_epochs", section.get("epochs", DEFAULT_SEQUENCE_MAX_EPOCHS))),
                "patience": int(section.get("patience", DEFAULT_SEQUENCE_PATIENCE)),
                "learning_rate": learning_rate,
                "weight_decay": float(section.get("weight_decay", 1e-4)),
                "hidden_dim": int(section.get("hidden_dim", 128)),
                "dropout": float(section.get("dropout", 0.1)),
            }
            if bool(config.get("quick_run", False)):
                quick_epochs = _first_int(_quick_config(config), ("max_epochs", "epochs"))
                params["max_epochs"] = min(params["max_epochs"], quick_epochs or DEFAULT_SEQUENCE_EPOCHS_QUICK)
            suffix = name
            if len(learning_rates) > 1:
                suffix = f"{name}_lr{_compact_value(learning_rate)}"
            specs.append(
                ReadoutSpec(
                    kind="sequence",
                    train_name=name,
                    classifier_name=suffix,
                    config_suffix=suffix,
                    params=params,
                )
            )
    return tuple(specs)


def _xgboost_readout_specs(section: Mapping[str, object]) -> list[ReadoutSpec]:
    return [
        ReadoutSpec(
            kind="static",
            train_name="xgboost",
            classifier_name="xgboost",
            config_suffix="xgboost",
            params=_static_params("xgboost", section),
        )
    ]


def _static_params(name: str, section: Mapping[str, object]) -> dict[str, object]:
    ignored = {"enabled", "n_estimators", "n_jobs"}
    model_params = {
        key: value
        for key, value in section.items()
        if key not in ignored
    }
    default_estimators = DEFAULT_XGBOOST_ESTIMATORS if name == "xgboost" else DEFAULT_TREE_ESTIMATORS
    params: dict[str, object] = {
        "max_iter": int(section.get("max_iter", DEFAULT_STATIC_MAX_ITER)),
        "n_estimators": (
            int(section.get("n_estimators", default_estimators))
            if not isinstance(section.get("n_estimators"), list)
            else default_estimators
        ),
        "n_jobs": int(section.get("n_jobs", 1)),
        "model_params": model_params,
    }
    if name == "xgboost":
        params["model_params"] = _xgboost_base_model_params(section)
    return params


def _xgboost_base_model_params(section: Mapping[str, object]) -> dict[str, object]:
    ignored = {"enabled", "n_estimators", "n_jobs"}
    params = {
        key: value
        for key, value in section.items()
        if key not in ignored
    }
    params.setdefault("early_stopping_rounds", DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS)
    return params


def _normalize_static_name(name: str) -> str | None:
    lowered = name.strip().lower()
    if lowered in XGBOOST_ALIASES:
        return "xgboost"
    if lowered in STATIC_CLASSIFIERS:
        return lowered
    return None


def _quick_allowed_readouts(config: Mapping[str, object]) -> set[str]:
    quick = _quick_config(config)
    raw = quick.get("classifiers", quick.get("readouts"))
    if raw is None:
        return set()
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = [str(item) for item in raw]
    else:
        raise ValueError("quick.classifiers must be a sequence or string")
    return {value.strip().lower() for value in values if value.strip()}


def _readout_allowed_in_quick(spec: ReadoutSpec, allowed: set[str]) -> bool:
    names = {
        spec.train_name.lower(),
        spec.classifier_name.lower(),
        spec.config_suffix.lower(),
    }
    if spec.train_name == "xgboost":
        names.update({"xgb", "xgboost"})
    return bool(names & allowed)


def _readout_name_matches(spec: ReadoutSpec, name: str) -> bool:
    target = str(name).strip().lower()
    names = {
        spec.train_name.lower(),
        spec.classifier_name.lower(),
        spec.config_suffix.lower(),
    }
    if spec.train_name == "xgboost":
        names.update({"xgb", "xgboost"})
    return target in names


def _validate_static_feature_matrix(
    features: np.ndarray,
    *,
    expected_samples: int,
    context: str,
    finite_already_checked: bool = False,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"{context}: static features must be a 2D array")
    if features.shape[0] != expected_samples:
        raise ValueError(f"{context}: feature row count mismatch")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{context}: static features must have non-empty dimensions")
    if not finite_already_checked:
        _raise_if_non_finite(features, name=f"{context} static features")


def _validate_sequence_feature_array(
    features: np.ndarray,
    *,
    expected_samples: int,
    context: str,
    finite_already_checked: bool = False,
) -> None:
    if features.ndim != 3:
        raise ValueError(f"{context}: sequence features must be a 3D array")
    if features.shape[0] != expected_samples:
        raise ValueError(f"{context}: sequence feature row count mismatch")
    if features.shape[0] == 0 or features.shape[1] == 0 or features.shape[2] == 0:
        raise ValueError(f"{context}: sequence features must have non-empty dimensions")
    if not finite_already_checked:
        _raise_if_non_finite(features, name=f"{context} sequence features")


def _require_two_classes(labels: np.ndarray, *, name: str) -> None:
    if np.unique(labels).shape[0] < 2:
        raise ValueError(f"{name} must contain at least two classes")


def _require_population(preflight: Mapping[str, object]) -> WaveletPopulation:
    population = preflight.get("population")
    if not isinstance(population, WaveletPopulation):
        raise ValueError("preflight['population'] must be a loaded WaveletPopulation")
    return population


def _paired_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("paired_wavelet_v2", {})
    if not isinstance(value, Mapping):
        raise ValueError("config['paired_wavelet_v2'] must be a mapping")
    return dict(value)


def _quick_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("quick", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("config['quick'] must be a mapping")
    return dict(value)


def _ours_signal_config(config: Mapping[str, object]) -> dict[str, object]:
    for key in ("ours_signal", "ours_wavelet"):
        value = config.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _resolved_ours_signal_config(
    config: Mapping[str, object],
    *,
    final_logits: object,
) -> dict[str, object]:
    ours = _ours_signal_config(config)
    ours["yes_no_trace_source"] = str(ours.get("yes_no_trace_source") or YES_NO_TRACE_SOURCE)
    if ours.get("yes_token_id") is not None and ours.get("no_token_id") is not None:
        ours.setdefault("chosen_yes_token", str(ours.get("yes_token", "")))
        ours.setdefault("chosen_no_token", str(ours.get("no_token", "")))
        ours.setdefault("tokenizer_candidate_table", [])
        _store_ours_signal_config(config, ours)
        return ours

    tokenizer = ours.get("tokenizer", config.get("tokenizer"))
    vocab = _optional_mapping(ours.get("vocab", config.get("vocab")))
    if tokenizer is None and vocab is None:
        return ours

    candidate_table = _build_tokenizer_candidate_table(
        tokenizer=tokenizer,
        vocab=vocab,
        logits_size=_logits_size(final_logits),
    )
    yes_token_id, no_token_id = resolve_yes_no_token_ids(
        tokenizer=tokenizer,
        vocab=vocab,
        logits_size=_logits_size(final_logits),
    )
    _mark_selected_token_candidates(candidate_table, yes_token_id=yes_token_id, no_token_id=no_token_id)
    ours["yes_token_id"] = int(yes_token_id)
    ours["no_token_id"] = int(no_token_id)
    ours["chosen_yes_token"] = _chosen_candidate_token(candidate_table, label="yes", token_id=yes_token_id)
    ours["chosen_no_token"] = _chosen_candidate_token(candidate_table, label="no", token_id=no_token_id)
    ours.setdefault("token_id_source", _token_id_source(tokenizer=tokenizer, vocab=vocab))
    ours["tokenizer_candidate_table"] = candidate_table
    _store_ours_signal_config(config, ours)
    return ours


def _build_tokenizer_candidate_table(
    *,
    tokenizer: object | None = None,
    vocab: Mapping[str, int] | None = None,
    logits_size: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, candidates in (("yes", YES_CANDIDATES), ("no", NO_CANDIDATES)):
        for candidate in candidates:
            for source, token_id in _candidate_token_ids_with_source(
                candidate,
                tokenizer=tokenizer,
                vocab=vocab,
            ):
                valid = token_id >= 0 and (logits_size is None or token_id < int(logits_size))
                if not valid:
                    continue
                rows.append(
                    {
                        "label": label,
                        "candidate": candidate,
                        "token_id": int(token_id),
                        "selected": False,
                        "source": source,
                    }
                )
    return rows


def _candidate_token_ids_with_source(
    text: str,
    *,
    tokenizer: object | None,
    vocab: Mapping[str, int] | None,
) -> list[tuple[str, int]]:
    ids: list[tuple[str, int]] = []
    if vocab is not None and text in vocab:
        ids.append(("vocab", int(vocab[text])))
    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(text)  # type: ignore[attr-defined]
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(("convert_tokens_to_ids", int(token_id)))
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        if isinstance(encoded, Sequence) and len(encoded) == 1:
            ids.append(("encode", int(encoded[0])))
    if tokenizer is not None and callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if isinstance(input_ids, Sequence) and len(input_ids) == 1:
            ids.append(("callable", int(input_ids[0])))
    return ids


def _mark_selected_token_candidates(
    rows: list[dict[str, object]],
    *,
    yes_token_id: int,
    no_token_id: int,
) -> None:
    selected = {"yes": int(yes_token_id), "no": int(no_token_id)}
    for row in rows:
        row["selected"] = (
            int(row.get("token_id", -1)) == selected.get(str(row.get("label", "")))
        )


def _chosen_candidate_token(rows: Sequence[Mapping[str, object]], *, label: str, token_id: int) -> str:
    for row in rows:
        if (
            str(row.get("label", "")) == label
            and int(row.get("token_id", -1)) == int(token_id)
        ):
            return str(row.get("candidate", ""))
    return ""


def _store_ours_signal_config(config: Mapping[str, object], ours: Mapping[str, object]) -> None:
    if not isinstance(config, dict):
        return
    key = "ours_signal" if "ours_signal" in config or "ours_wavelet" not in config else "ours_wavelet"
    existing = config.get(key)
    merged = dict(existing) if isinstance(existing, Mapping) else {}
    merged.update(dict(ours))
    config[key] = merged


def _optional_mapping(value: object) -> Mapping[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("ours_signal.vocab must be a mapping when provided")
    return {str(key): int(item) for key, item in value.items()}


def _token_id_source(*, tokenizer: object | None, vocab: Mapping[str, int] | None) -> str:
    if vocab is not None:
        return "vocab_resolved"
    source = getattr(tokenizer, "source", None)
    return str(source or "tokenizer_resolved")


def _logits_size(values: object) -> int | None:
    if values is None:
        return None
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 2:
        if array.shape[0] == 0:
            raise ValueError("final_logits 2D array must have at least one row")
        array = array[-1]
    if array.ndim != 1:
        raise ValueError("final_logits must be a 1D vector or 2D layer-by-vocab array")
    if array.shape[0] == 0:
        raise ValueError("final_logits must not be empty")
    return int(array.shape[0])


def _yes_no_trace_source(pair: PairSpec, config: Mapping[str, object]) -> str:
    if pair.source != "Ours":
        return ""
    ours = _ours_signal_config(config)
    return str(ours.get("yes_no_trace_source") or YES_NO_TRACE_SOURCE)


def _epsilon(config: Mapping[str, object]) -> float:
    paired = _paired_config(config)
    value = paired.get("epsilon", config.get("epsilon", 1e-12))
    epsilon = float(value)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    return epsilon


def _coerce_blocks(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        candidates = [part.strip().upper() for part in values.split(",") if part.strip()]
    elif isinstance(values, Sequence):
        candidates = [str(item).strip().upper() for item in values if str(item).strip()]
    else:
        raise ValueError("blocks must be a sequence or comma-separated string")
    unknown = sorted(set(candidates) - set(PAIR_BLOCKS))
    if unknown:
        raise ValueError(f"unknown paired v2 blocks: {unknown}")
    return tuple(block for block in PAIR_BLOCKS if block in set(candidates))


def _ordered_blocks(values: Sequence[str] | Any) -> tuple[str, ...]:
    present = {str(value) for value in values}
    return tuple(block for block in PAIR_BLOCKS if block in present)


def _section_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {"enabled": True}
    if isinstance(value, bool):
        return {"enabled": value}
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("classifier sections must be mappings or booleans")


def _as_grid_values(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _first_int(mapping: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if value not in {None, ""}:
            return int(value)
    return None


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_text(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _first_present(mapping: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def _compact_value(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _failure_reason(error: BaseException) -> str:
    if isinstance(error, ImportError):
        return str(error)
    reason = str(error)
    if XGBOOST_NOT_INSTALLED in reason:
        return XGBOOST_NOT_INSTALLED
    return reason or type(error).__name__


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(np.asarray(array)).all():
        raise ValueError(f"{name} contains non-finite values")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


__all__ = [
    "run",
    "run_experiment",
    "run_paired_experiment",
    "run_paired_wavelet_experiment",
]
