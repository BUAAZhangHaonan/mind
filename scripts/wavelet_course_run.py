#!/usr/bin/env python3
"""Run wavelet-course diagnostics on Qwen RePOPE Stage 0 cache entries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

import numpy as np
import torch
import yaml

from mind.wavelet_course.baselines import (
    SplitScores,
    final_hidden_logreg,
    mean_layer_hidden_logreg,
    norm_traj_logreg,
    sphere_traj_meanpool_logreg,
    train_logistic_regression,
    train_teacher_lstm,
    train_xgboost_grid,
)
from mind.wavelet_course.cache_loading import load_repope_qwen_cache_entries
from mind.wavelet_course.metrics import evaluate_validation_test
from mind.wavelet_course.ours_wavelet_features import (
    WaveletConfigError,
    extract_ours_wavelet_features,
    save_ours_features,
)
from mind.wavelet_course.population import (
    WaveletPopulation,
    build_wavelet_population,
    population_key,
)
from mind.wavelet_course.reporting import (
    best_config_rows,
    write_best_configs_csv,
    write_json,
    write_metrics_csv,
    write_summary_md,
)
from mind.wavelet_course.teacher_bagua_features import write_teacher_memmap
from mind.wavelet_course.utils import DEFAULT_SPLIT_RATIOS, SPLIT_NAMES


DEFAULT_CONFIG = Path("configs/wavelet_course/repope_qwen3_vl_8b.yaml")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/qwen3_vl_8b.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/wavelet_course")

EXPECTED_TEACHER_NAMES = [
    "teacher_bagua_haar_l1_lstm",
    "teacher_bagua_db2_l1_lstm",
    "teacher_bagua_db4_l1_lstm",
]
EXPECTED_OURS_NAMES = [
    "ours_db2_swt_l2_logreg",
    "ours_db2_swt_l3_logreg",
    "ours_sym4_swt_l2_logreg",
    "ours_db2_swt_l2_xgb",
    "ours_db2_swt_l3_xgb",
    "ours_sym4_swt_l2_xgb",
]
EXPECTED_BASELINE_NAMES = [
    "final_hidden_logreg",
    "mean_layer_hidden_logreg",
    "norm_traj_logreg",
    "sphere_traj_meanpool_logreg",
]
EXPECTED_CONFIG_NAMES = EXPECTED_TEACHER_NAMES + EXPECTED_OURS_NAMES + EXPECTED_BASELINE_NAMES

BASELINE_BUILDERS = {
    "final_hidden_logreg": final_hidden_logreg,
    "mean_layer_hidden_logreg": mean_layer_hidden_logreg,
    "norm_traj_logreg": norm_traj_logreg,
    "sphere_traj_meanpool_logreg": sphere_traj_meanpool_logreg,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--teacher-bagua-max-train-samples", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(load_yaml_config(Path(args.config)), args)
    validate_experiment_config_names(config)

    output_root = Path(str(config.get("output_root", DEFAULT_OUTPUT_ROOT)))
    reports_dir = output_root / "reports"
    audit_dir = output_root / "audit"
    features_dir = output_root / "features"
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = reports_dir / "experiment_config_resolved.json"
    write_json(config, resolved_config_path)
    ensure_device_available(str(config["device"]), allow_cpu=bool(config["allow_cpu"]))

    try:
        preflight = run_preflight(config, audit_dir=audit_dir)
    except Exception as error:
        print(f"preflight_failed={error}", file=sys.stderr)
        if args.preflight_only:
            return 2
        raise
    print_preflight_stats(preflight)
    if args.preflight_only:
        return 0

    metric_rows = run_configured_experiments(config, preflight=preflight, output_root=output_root)
    write_json(config, resolved_config_path)
    metrics_path = write_metrics_csv(metric_rows, reports_dir / "metrics.csv")
    best_rows = best_config_rows(metric_rows)
    best_configs_path = write_best_configs_csv(metric_rows, reports_dir / "best_configs.csv")
    failures = [row for row in metric_rows if str(row.get("status", "")) == "failure"]
    summary_path = write_summary_md(
        output=reports_dir / "summary.md",
        config=config,
        cache_audit=preflight["cache_audit"],
        population_summary=preflight["population_summary"],
        metrics_rows=metric_rows,
        best_rows=best_rows,
        metrics_path=metrics_path,
        best_configs_path=best_configs_path,
        quick=bool(config.get("quick_run", False)),
        failures=failures,
    )
    print_final_summary(
        preflight=preflight,
        best_rows=best_rows,
        failures=failures,
        summary_path=summary_path,
        metrics_path=metrics_path,
    )
    fatal_failures = [row for row in failures if not _is_allowed_failure(row)]
    return 0 if not fatal_failures else 1


def load_yaml_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return dict(payload)


def resolve_config(config: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("output_root", str(DEFAULT_OUTPUT_ROOT))
    resolved["device"] = args.device or str(resolved.get("device", "cuda:0"))
    resolved["allow_cpu"] = bool(args.allow_cpu or resolved.get("allow_cpu", False))
    resolved["allow_no_xgboost"] = bool(resolved.get("allow_no_xgboost", True))
    resolved["quick_run"] = bool(args.quick)

    teacher = dict(resolved.get("teacher_bagua", {}) or {})
    classifiers = dict(resolved.get("classifiers", {}) or {})
    teacher_lstm = dict(classifiers.get("teacher_lstm", {}) or {})
    if args.teacher_bagua_max_train_samples is not None:
        teacher["max_train_samples"] = int(args.teacher_bagua_max_train_samples)
    if args.quick:
        quick = dict(resolved.get("quick", {}) or {})
        teacher["max_train_samples"] = int(
            teacher.get("max_train_samples") or quick.get("teacher_max_train_samples") or 128
        )
        teacher["epochs"] = int(quick.get("epochs", 3))
        teacher_lstm["epochs"] = int(quick.get("epochs", 3))
    teacher_lstm.setdefault("epochs", int(teacher.get("epochs", 10) or 10))
    teacher_lstm.setdefault("batch_size", int(teacher.get("batch_size", 16) or 16))
    teacher_lstm.setdefault("learning_rate", float(teacher.get("learning_rate", 0.001) or 0.001))
    teacher_lstm.setdefault("patience", int(teacher.get("patience", 3) or 3))
    classifiers["teacher_lstm"] = teacher_lstm
    resolved["teacher_bagua"] = teacher
    resolved["classifiers"] = classifiers

    ours = dict(resolved.get("ours_wavelet", {}) or {})
    if ours.get("yes_token_id") is not None and ours.get("no_token_id") is not None:
        ours["token_id_source"] = "final_margin_broadcast_explicit_token_ids"
    resolved["ours_wavelet"] = ours
    resolved["configs"] = flatten_experiment_configs(resolved)
    return resolved


def flatten_experiment_configs(config: Mapping[str, object]) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    configs.extend(_teacher_experiment_configs(config.get("teacher_bagua", {})))
    configs.extend(_ours_experiment_configs(config.get("ours_wavelet", {})))
    configs.extend(_baseline_experiment_configs(config.get("baselines", {})))
    return configs


def validate_experiment_config_names(config_or_configs: object) -> None:
    if isinstance(config_or_configs, Mapping):
        configs = flatten_experiment_configs(config_or_configs)
        names = [str(item.get("config_name", "")) for item in configs]
    elif isinstance(config_or_configs, list):
        names = [str(item.get("config_name", "")) for item in config_or_configs if isinstance(item, Mapping)]
    else:
        raise ValueError("configs must be a mapping or list")
    if names != EXPECTED_CONFIG_NAMES:
        raise ValueError("wavelet-course config list must exactly match: " + ", ".join(EXPECTED_CONFIG_NAMES))


def _teacher_experiment_configs(section: object) -> list[dict[str, object]]:
    if not isinstance(section, Mapping):
        return []
    configs = section.get("configs", [])
    if not isinstance(configs, list):
        raise ValueError("teacher_bagua.configs must be a list")
    result: list[dict[str, object]] = []
    for item in configs:
        if not isinstance(item, Mapping):
            continue
        normalized = dict(item)
        normalized["config_name"] = _pop_config_name(normalized)
        normalized.setdefault("method_family", "teacher_bagua")
        normalized.setdefault("classifier", "lstm")
        result.append(normalized)
    return result


def _ours_experiment_configs(section: object) -> list[dict[str, object]]:
    if not isinstance(section, Mapping):
        return []
    configs = section.get("configs", [])
    if not isinstance(configs, list):
        raise ValueError("ours_wavelet.configs must be a list")
    raw_items = [dict(item) for item in configs if isinstance(item, Mapping)]
    if any("classifier" in item for item in raw_items):
        result = []
        for item in raw_items:
            normalized = dict(item)
            normalized["config_name"] = _pop_config_name(normalized)
            normalized.setdefault("method_family", "ours_wavelet")
            result.append(normalized)
        return result

    expanded: list[dict[str, object]] = []
    for classifier in ("logreg", "xgb"):
        for item in raw_items:
            base_name = str(item.get("config_name") or item.get("name") or "")
            normalized = {
                key: value
                for key, value in item.items()
                if key not in {"config_name", "name"}
            }
            normalized["config_name"] = f"{base_name}_{classifier}"
            normalized["method_family"] = "ours_wavelet"
            normalized["classifier"] = classifier
            expanded.append(normalized)
    return expanded


def _baseline_experiment_configs(section: object) -> list[dict[str, object]]:
    if isinstance(section, Mapping):
        raw_configs = section.get("configs", [])
        if not isinstance(raw_configs, list):
            raise ValueError("baselines.configs must be a list")
        names_or_configs: list[object] = list(raw_configs)
    elif isinstance(section, list):
        names_or_configs = list(section)
    else:
        return []

    result: list[dict[str, object]] = []
    for item in names_or_configs:
        if isinstance(item, str):
            config_name = item
            normalized: dict[str, object] = {"feature_builder": item}
        elif isinstance(item, Mapping):
            normalized = dict(item)
            config_name = _pop_config_name(normalized)
        else:
            continue
        normalized["config_name"] = config_name
        normalized.setdefault("method_family", "mind_baseline")
        normalized.setdefault("classifier", "logreg")
        normalized.setdefault("feature_builder", config_name)
        result.append(normalized)
    return result


def _pop_config_name(config: dict[str, object]) -> str:
    if "config_name" in config:
        return str(config.pop("config_name"))
    return str(config.pop("name", ""))


def ensure_device_available(device: str, *, allow_cpu: bool) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            f"requested device {device}, but torch.cuda.is_available() is false and allow_cpu=false"
        )


def run_preflight(config: Mapping[str, object], *, audit_dir: Path) -> dict[str, object]:
    cache_audit_path = audit_dir / "cache_acceptance.json"
    population_audit_path = audit_dir / "population_audit.csv"
    try:
        entries = load_repope_qwen_cache_entries(
            stage0_root=Path(str(config.get("stage0_root", "outputs/stage0"))),
            manifest_path=Path(str(config.get("stage0_root", "outputs/stage0"))) / "manifests" / "cache_manifest.json",
            model_name=str(config["model_name"]),
            dataset_name=str(config["dataset_name"]),
            subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
            expected_num_layers=int(config["expected_num_layers"]),
            expected_hidden_dim=int(config["expected_hidden_dim"]),
        )
    except Exception as error:
        cache_audit = {"accepted": False, "num_entries": 0, "failure_reason": str(error)}
        write_json(cache_audit, cache_audit_path)
        write_population_audit_csv([], population_audit_path)
        raise RuntimeError(str(error)) from error

    cache_audit = build_cache_acceptance(entries, config)
    try:
        population = build_wavelet_population(
            entries,
            manifest_dir=Path(str(config.get("stage0_root", "outputs/stage0"))) / "manifests",
            dataset_name=str(config["dataset_name"]),
            subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
            seed=int(config["seed"]),
            ratios=split_ratio_values(config),
        )
        split_validation = validate_population_splits(
            population,
            require_positive_in_each_split=bool(config.get("require_positive_in_each_split", False)),
        )
    except Exception as error:
        if "population" in locals():
            write_population_audit_csv(population.audit_rows, population_audit_path)
        else:
            write_population_audit_csv([], population_audit_path)
        cache_audit["split_validation"] = {"valid": False, "failure_reason": str(error)}
        write_json(cache_audit, cache_audit_path)
        raise RuntimeError(str(error)) from error

    cache_audit["split_validation"] = split_validation
    write_json(cache_audit, cache_audit_path)
    write_population_audit_csv(population.audit_rows, population_audit_path)
    return {
        "entries": entries,
        "cache_audit": cache_audit,
        "population": population,
        "population_summary": population_summary(population),
        "split_validation": split_validation,
    }


def split_ratio_values(config: Mapping[str, object]) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    if isinstance(ratios, Mapping):
        return tuple(float(ratios[name]) for name in SPLIT_NAMES)  # type: ignore[return-value]
    return tuple(float(value) for value in ratios)  # type: ignore[arg-type]


def build_cache_acceptance(entries: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> dict[str, object]:
    subset_counts = Counter(str(row.get("subset", "")) for row in entries)
    return {
        "accepted": True,
        "num_entries": len(entries),
        "model_name": str(config["model_name"]),
        "dataset_name": str(config["dataset_name"]),
        "expected_num_layers": int(config["expected_num_layers"]),
        "expected_hidden_dim": int(config["expected_hidden_dim"]),
        "subsets": {subset: int(count) for subset, count in sorted(subset_counts.items())},
    }


def validate_population_splits(
    population: WaveletPopulation,
    *,
    require_positive_in_each_split: bool,
) -> dict[str, object]:
    counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_NAMES}
    for entry, label in zip(population.primary_entries, population.labels, strict=True):
        split = str(entry.get("wavelet_split", ""))
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid wavelet split {split!r}")
        counts[split][int(label)] += 1
    for split in SPLIT_NAMES:
        if counts[split][0] + counts[split][1] == 0:
            raise RuntimeError(f"{split} split has no primary rows")
        if require_positive_in_each_split and counts[split][1] == 0:
            raise RuntimeError(f"{split} split has no positives")
    if counts["train"][0] == 0 or counts["train"][1] == 0:
        raise RuntimeError("train split lacks two classes")
    return {
        "valid": True,
        "split_source": population.split_source,
        "counts": {
            split: {"neg": int(counts[split][0]), "pos": int(counts[split][1])}
            for split in SPLIT_NAMES
        },
    }


def population_summary(population: WaveletPopulation) -> dict[str, object]:
    labels = np.asarray(population.labels, dtype=np.int64)
    return {
        "num_primary_population": int(labels.shape[0]),
        "num_hard_hallucination": int(np.sum(labels == 1)),
        "num_correct": int(np.sum(labels == 0)),
        "split_source": population.split_source,
    }


def write_population_audit_csv(rows: Sequence[Mapping[str, object]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "subset",
        "total",
        "gt_yes",
        "gt_no",
        "parsed_yes",
        "parsed_no",
        "correct",
        "hard_hallucination",
        "false_negative",
        "parsed_none",
        "invalid_label",
        "primary_pos",
        "primary_neg",
        "train_pos",
        "train_neg",
        "validation_pos",
        "validation_neg",
        "test_pos",
        "test_neg",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in base_fields})
    fields = base_fields + extra_fields
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return output


def run_configured_experiments(
    config: Mapping[str, object],
    *,
    preflight: Mapping[str, object],
    output_root: Path,
) -> list[dict[str, object]]:
    population = preflight["population"]
    if not isinstance(population, WaveletPopulation):
        raise TypeError("preflight population has unexpected type")
    entries = population.primary_entries
    labels = np.asarray(population.labels, dtype=np.int64)
    features_dir = output_root / "features"
    shape_metadata: dict[str, dict[str, object]] = {"teacher_bagua": {}, "ours_wavelet": {}}
    metric_rows: list[dict[str, object]] = []

    for experiment in flatten_experiment_configs(config):
        config_start = time.perf_counter()
        try:
            row = run_one_config(
                experiment,
                config,
                entries=entries,
                labels=labels,
                features_dir=features_dir,
                shape_metadata=shape_metadata,
            )
        except Exception as error:
            row = failure_metric_row(
                experiment,
                config,
                _failure_reason(error),
                timing={"total_seconds": time.perf_counter() - config_start},
            )
        metric_rows.append(row)

    write_feature_shape_json(config, output_root, shape_metadata, family="teacher_bagua")
    write_feature_shape_json(config, output_root, shape_metadata, family="ours_wavelet")
    return metric_rows


def run_one_config(
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    *,
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    features_dir: Path,
    shape_metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    total_start = time.perf_counter()
    timing: dict[str, object] = {
        "feature_seconds": "",
        "train_eval_seconds": "",
        "total_seconds": "",
    }
    stage = "pre_feature"
    feature_start = total_start
    train_start = total_start
    active_entries = entries
    active_labels = labels
    try:
        if not entries:
            raise ValueError("primary population is empty")
        family = str(experiment.get("method_family", ""))
        if family == "teacher_bagua":
            active_entries, active_labels = limit_teacher_entries_for_extraction(entries, labels, config)
            train_labels = labels_for_split(active_entries, active_labels, "train")
            if np.unique(train_labels).shape[0] < 2:
                timing["total_seconds"] = time.perf_counter() - total_start
                return failure_metric_row(
                    experiment,
                    config,
                    "teacher_bagua_train_requires_at_least_two_classes_after_sample_limit",
                    timing=timing,
                )
            stage = "feature"
            feature_start = time.perf_counter()
            features = teacher_feature_matrix(experiment, config, active_entries, active_labels, features_dir, shape_metadata)
            feature_end = time.perf_counter()
            timing["feature_seconds"] = feature_end - feature_start
            stage = "train_eval"
            train_start = feature_end
            scores = train_classifier(
                experiment,
                config,
                features=features,
                labels=active_labels,
                entries=active_entries,
                teacher=True,
            )
        elif family == "ours_wavelet":
            stage = "feature"
            feature_start = time.perf_counter()
            features = ours_feature_matrix(experiment, config, entries, features_dir, shape_metadata)
            feature_end = time.perf_counter()
            timing["feature_seconds"] = feature_end - feature_start
            stage = "train_eval"
            train_start = feature_end
            scores = train_classifier(experiment, config, features=features, labels=labels, entries=entries, teacher=False)
        elif family in {"mind_baseline", "halp_like_baseline"}:
            stage = "feature"
            feature_start = time.perf_counter()
            features = baseline_feature_matrix(experiment, entries)
            feature_end = time.perf_counter()
            timing["feature_seconds"] = feature_end - feature_start
            stage = "train_eval"
            train_start = feature_end
            scores = train_classifier(experiment, config, features=features, labels=labels, entries=entries, teacher=False)
        else:
            raise ValueError(f"unknown method_family={family!r}")
        train_end = time.perf_counter()
        timing["train_eval_seconds"] = train_end - train_start
        timing["total_seconds"] = time.perf_counter() - total_start
        if isinstance(scores, str):
            return failure_metric_row(experiment, config, scores, timing=timing)
        return success_metric_row(
            experiment,
            config,
            labels=active_labels,
            scores=scores,
            entries=active_entries,
            feature_shape=features.shape,
            timing=timing,
        )
    except Exception as error:
        stage_end = time.perf_counter()
        if stage == "feature" and timing["feature_seconds"] == "":
            timing["feature_seconds"] = stage_end - feature_start
        elif stage == "train_eval" and timing["train_eval_seconds"] == "":
            timing["train_eval_seconds"] = stage_end - train_start
        timing["total_seconds"] = time.perf_counter() - total_start
        return failure_metric_row(experiment, config, _failure_reason(error), timing=timing)


def limit_teacher_entries_for_extraction(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    config: Mapping[str, object],
) -> tuple[Sequence[Mapping[str, object]], np.ndarray]:
    teacher = dict(config.get("teacher_bagua", {}) or {})
    limit = teacher.get("max_train_samples")
    if limit in (None, ""):
        return entries, labels
    max_train = int(limit)
    if max_train < 0:
        raise ValueError("teacher_bagua.max_train_samples must be non-negative")
    if labels.shape[0] != len(entries):
        raise ValueError("entries and labels must have matching sample counts")

    quick_run = bool(config.get("quick_run", False))
    limited_splits = {"train"}
    if quick_run:
        limited_splits.update({"validation", "test"})

    split_indices: dict[str, list[int]] = {split: [] for split in SPLIT_NAMES}
    for index, entry in enumerate(entries):
        split = str(entry.get("wavelet_split", ""))
        if split in split_indices:
            split_indices[split].append(index)

    keep_indices: set[int] = set(range(len(entries)))
    for split in limited_splits:
        original_indices = split_indices[split]
        limited_indices = stratified_limit_split_indices(original_indices, labels, max_train)
        keep_indices.difference_update(original_indices)
        keep_indices.update(limited_indices)

    limited_entries: list[Mapping[str, object]] = []
    limited_labels: list[int] = []
    for index, (entry, label) in enumerate(zip(entries, labels, strict=True)):
        if index not in keep_indices:
            continue
        limited_entries.append(entry)
        limited_labels.append(int(label))
    return limited_entries, np.asarray(limited_labels, dtype=np.int64)


def stratified_limit_split_indices(
    split_indices: Sequence[int],
    labels: np.ndarray,
    limit: int,
) -> list[int]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    indices = [int(index) for index in split_indices]
    if len(indices) <= limit:
        return indices
    if limit == 0:
        return []
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if any(index < 0 or index >= labels.shape[0] for index in indices):
        raise ValueError("split indices must be valid label positions")

    first_index_by_label: dict[int, int] = {}
    for index in indices:
        label = int(labels[index])
        if label not in first_index_by_label:
            first_index_by_label[label] = index

    selected = set(first_index_by_label.values()) if len(first_index_by_label) <= limit else set()
    for index in indices:
        if len(selected) >= limit:
            break
        selected.add(index)
    return [index for index in indices if index in selected]


def labels_for_split(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    split_name: str,
) -> np.ndarray:
    if labels.shape[0] != len(entries):
        raise ValueError("entries and labels must have matching sample counts")
    split_names = np.asarray([str(entry.get("wavelet_split", "")) for entry in entries])
    return labels[split_names == split_name]


def teacher_feature_matrix(
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    features_dir: Path,
    shape_metadata: dict[str, dict[str, object]],
) -> np.ndarray:
    config_name = str(experiment["config_name"])
    path = features_dir / f"{config_name}.teacher_bagua.memmap"
    metadata_path = features_dir / f"{config_name}.teacher_bagua_feature_shape.json"
    metadata = write_teacher_memmap(
        entries,
        labels.tolist(),
        path,
        {
            "wavelet": experiment.get("wavelet", "db2"),
            "level": int(experiment.get("level", 1)),
            "threshold": experiment.get("threshold", "universal_soft"),
        },
        metadata_path=metadata_path,
    )
    shape_metadata["teacher_bagua"][config_name] = metadata
    return np.memmap(path, dtype=np.float32, mode="r", shape=tuple(metadata["shape"]))


def ours_feature_matrix(
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    features_dir: Path,
    shape_metadata: dict[str, dict[str, object]],
) -> np.ndarray:
    config_name = str(experiment["config_name"])
    tokenizer = load_local_tokenizer(config)
    feature_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    ours_config = dict(config.get("ours_wavelet", {}) or {})
    tokenizer_source = getattr(tokenizer, "source", None) if tokenizer is not None else None
    if tokenizer_source:
        ours_config["token_id_source"] = str(tokenizer_source)
        if isinstance(config, dict):
            existing_ours = dict(config.get("ours_wavelet", {}) or {})
            existing_ours["token_id_source"] = str(tokenizer_source)
            config["ours_wavelet"] = existing_ours
    extract_config = {
        "wavelet": experiment.get("wavelet"),
        "level": int(experiment.get("level", 2)),
        "yes_token_id": ours_config.get("yes_token_id"),
        "no_token_id": ours_config.get("no_token_id"),
    }
    for entry in entries:
        result = extract_ours_wavelet_features(
            entry["layer_vectors"],
            entry["first_token_logits"],
            extract_config,
            tokenizer=tokenizer,
        )
        if feature_names is None:
            feature_names = result.feature_names
        elif feature_names != result.feature_names:
            raise ValueError("Ours-Wavelet feature names changed across entries")
        feature_rows.append(result.features)
    features = np.stack(feature_rows, axis=0).astype(np.float32, copy=False)
    names = feature_names or []
    payload = save_ours_features(
        features,
        names,
        features_dir / f"{config_name}.ours_wavelet.npy",
        names_path=features_dir / f"{config_name}.ours_wavelet_feature_names.json",
    )
    payload["shape"] = list(features.shape)
    payload["config"] = {
        "wavelet": experiment.get("wavelet"),
        "level": int(experiment.get("level", 2)),
        "token_id_source": ours_config.get("token_id_source", ""),
    }
    shape_metadata["ours_wavelet"][config_name] = payload
    return features


def baseline_feature_matrix(
    experiment: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> np.ndarray:
    builder_name = str(experiment.get("feature_builder") or experiment.get("config_name"))
    builder = BASELINE_BUILDERS.get(builder_name)
    if builder is None:
        raise ValueError(f"unknown baseline feature builder={builder_name!r}")
    return builder(entries)


def load_local_tokenizer(config: Mapping[str, object]) -> Any | None:
    ours = dict(config.get("ours_wavelet", {}) or {})
    if ours.get("yes_token_id") is not None and ours.get("no_token_id") is not None:
        return None
    model_id = load_model_id(config)
    auto_error: BaseException | None = None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        auto_error = exc
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=True,
            )
            _set_tokenizer_source(tokenizer, "local_auto_tokenizer")
            return tokenizer
        except Exception as exc:
            auto_error = exc

    try:
        tokenizer_json = _resolve_local_tokenizer_json(model_id)
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        return _TokenizerJsonWrapper(tokenizer, tokenizer_json)
    except Exception as json_error:
        raise RuntimeError(
            "failed to load local tokenizer for "
            f"model_id={model_id!r}; "
            f"AutoTokenizer error: {_format_error(auto_error)}; "
            f"tokenizer.json error: {_format_error(json_error)}"
        ) from json_error


class _TokenizerJsonWrapper:
    source = "local_tokenizer_json"

    def __init__(self, tokenizer: Any, tokenizer_json_path: Path) -> None:
        self._tokenizer = tokenizer
        self.tokenizer_json_path = str(tokenizer_json_path)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = getattr(encoded, "ids", encoded)
        return [int(token_id) for token_id in ids]


def _set_tokenizer_source(tokenizer: Any, source: str) -> None:
    try:
        setattr(tokenizer, "source", source)
    except Exception:
        pass


def _format_error(error: BaseException | None) -> str:
    if error is None:
        return "not attempted"
    return f"{type(error).__name__}: {error}"


def _resolve_local_tokenizer_json(model_id: str) -> Path:
    model_path = Path(model_id).expanduser()
    if model_path.is_dir():
        tokenizer_json = model_path / "tokenizer.json"
        if tokenizer_json.is_file():
            return tokenizer_json

    repo_dir = _hf_cache_repo_dir(model_id)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"missing HF cache repo dir: {repo_dir}")

    refs_main = repo_dir / "refs" / "main"
    refs_error: FileNotFoundError | None = None
    if refs_main.is_file():
        snapshot_name = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_name:
            tokenizer_json = repo_dir / "snapshots" / snapshot_name / "tokenizer.json"
            if tokenizer_json.is_file():
                return tokenizer_json
            refs_error = FileNotFoundError(f"refs/main points to missing tokenizer.json: {tokenizer_json}")

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(f"missing HF cache snapshots dir: {snapshots_dir}")
    candidates = [path / "tokenizer.json" for path in snapshots_dir.iterdir() if (path / "tokenizer.json").is_file()]
    if not candidates:
        if refs_error is not None:
            raise FileNotFoundError(
                f"{refs_error}; no tokenizer.json found under snapshots dir: {snapshots_dir}"
            ) from refs_error
        raise FileNotFoundError(f"no tokenizer.json found under snapshots dir: {snapshots_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    return candidates[0]


def _hf_cache_repo_dir(model_id: str) -> Path:
    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_root:
        hub_dir = Path(cache_root).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        hub_dir = hf_home / "hub"
    return hub_dir / f"models--{model_id.replace('/', '--')}"


def load_model_id(config: Mapping[str, object]) -> str:
    if config.get("model_id"):
        return str(config["model_id"])
    model_config_path = Path(str(config.get("model_config_path", DEFAULT_MODEL_CONFIG_PATH)))
    payload = load_yaml_config(model_config_path)
    model_id = payload.get("model_id")
    if not model_id:
        raise ValueError(f"{model_config_path}: missing model_id")
    return str(model_id)


def train_classifier(
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    *,
    features: np.ndarray,
    labels: np.ndarray,
    entries: Sequence[Mapping[str, object]],
    teacher: bool,
) -> SplitScores | str:
    classifier = str(experiment.get("classifier", ""))
    seed = int(config.get("seed", 0))
    if teacher:
        split_arrays = split_feature_matrix(features, labels, entries=entries)
        train_x, train_y = maybe_limit_teacher_train(split_arrays["train_x"], split_arrays["train_y"], config)
        lstm_cfg = dict((config.get("classifiers", {}) or {}).get("teacher_lstm", {}) or {})  # type: ignore[union-attr]
        result = train_teacher_lstm(
            train_x,
            train_y,
            split_arrays["validation_x"],
            split_arrays["validation_y"],
            test_x=split_arrays["test_x"],
            device=str(config.get("device", "")),
            batch_size=int(lstm_cfg.get("batch_size", 32)),
            max_epochs=int(lstm_cfg.get("epochs", 12)),
            patience=int(lstm_cfg.get("patience", 3)),
            learning_rate=float(lstm_cfg.get("learning_rate", 0.001)),
            seed=seed,
        )
        return result.scores
    split_arrays = split_feature_matrix(features, labels, entries=entries)
    if classifier == "logreg":
        logreg_cfg = dict((config.get("classifiers", {}) or {}).get("logreg", {}) or {})  # type: ignore[union-attr]
        return train_logistic_regression(
            split_arrays["train_x"],
            split_arrays["train_y"],
            validation_x=split_arrays["validation_x"],
            test_x=split_arrays["test_x"],
            max_iter=int(logreg_cfg.get("max_iter", 1000)),
            random_state=seed,
        ).scores
    if classifier == "xgb":
        result = train_xgboost_grid(
            split_arrays["train_x"],
            split_arrays["train_y"],
            validation_x=split_arrays["validation_x"],
            validation_y=split_arrays["validation_y"],
            test_x=split_arrays["test_x"],
            allow_no_xgboost=bool(config.get("allow_no_xgboost", True)),
            random_state=seed,
        )
        if result.status != "success" or result.scores is None:
            reason = result.rows[0].get("reason", "xgboost_training_failed") if result.rows else "xgboost_training_failed"
            return str(reason)
        return result.scores
    raise ValueError(f"unsupported classifier={classifier!r}")


def split_feature_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    entries: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    split_names = np.asarray([str(entry.get("wavelet_split", "")) for entry in entries])
    values = np.asarray(features, dtype=np.float32)
    if values.shape[0] != labels.shape[0] or values.shape[0] != split_names.shape[0]:
        raise ValueError("features, labels, and split names must have matching sample counts")
    masks = {split: split_names == split for split in SPLIT_NAMES}
    return {
        "train_x": values[masks["train"]],
        "train_y": labels[masks["train"]],
        "validation_x": values[masks["validation"]],
        "validation_y": labels[masks["validation"]],
        "test_x": values[masks["test"]],
        "test_y": labels[masks["test"]],
    }


def maybe_limit_teacher_train(train_x: np.ndarray, train_y: np.ndarray, config: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    teacher = dict(config.get("teacher_bagua", {}) or {})
    limit = teacher.get("max_train_samples")
    if limit in (None, ""):
        return train_x, train_y
    indices = stratified_limit_split_indices(range(train_y.shape[0]), train_y, int(limit))
    return train_x[indices], train_y[indices]


def success_metric_row(
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    *,
    labels: np.ndarray,
    scores: SplitScores,
    entries: Sequence[Mapping[str, object]],
    feature_shape: tuple[int, ...],
    timing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    split_names = np.asarray([str(entry.get("wavelet_split", "")) for entry in entries])
    masks = {split: split_names == split for split in SPLIT_NAMES}
    if scores.validation is None or scores.test is None:
        raise ValueError("validation and test scores are required")
    evaluation = evaluate_validation_test(
        labels[masks["validation"]],
        scores.validation,
        labels[masks["test"]],
        scores.test,
    )
    test_metrics = evaluation["test"]
    return {
        **base_metric_row(experiment, config),
        "train_samples": int(masks["train"].sum()),
        "val_samples": int(masks["validation"].sum()),
        "test_samples": int(masks["test"].sum()),
        "train_pos": int(labels[masks["train"]].sum()),
        "val_pos": int(labels[masks["validation"]].sum()),
        "test_pos": int(labels[masks["test"]].sum()),
        "feature_shape": "x".join(str(dim) for dim in feature_shape),
        "pr_auc": test_metrics["pr_auc"],
        "average_precision": test_metrics["average_precision"],
        "roc_auc": test_metrics["roc_auc"],
        "best_val_threshold": evaluation["threshold"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "tpr_at_1pct_fpr": test_metrics["tpr_at_1pct_fpr"],
        "fpr_at_95pct_tpr": test_metrics["fpr_at_95pct_tpr"],
        "status": "success",
        "failure_reason": "",
        **timing_metric_fields(timing),
    }


def failure_metric_row(
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    failure_reason: str,
    *,
    timing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        **base_metric_row(experiment, config),
        "status": "failure",
        "failure_reason": failure_reason,
        **timing_metric_fields(timing),
    }


def base_metric_row(experiment: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    return {
        "config_name": experiment.get("config_name", ""),
        "method_family": experiment.get("method_family", ""),
        "model_name": config.get("model_name", ""),
        "dataset_name": config.get("dataset_name", ""),
        "subset_scope": ",".join(str(item) for item in config.get("subsets", [])),  # type: ignore[arg-type]
        "classifier": experiment.get("classifier", ""),
        "wavelet": experiment.get("wavelet", ""),
        "wavelet_level": experiment.get("level", ""),
        "transform": experiment.get("transform", experiment.get("feature_builder", "")),
    }


def timing_metric_fields(timing: Mapping[str, object] | None = None) -> dict[str, object]:
    values = dict(timing or {})
    return {
        "feature_seconds": values.get("feature_seconds", ""),
        "train_eval_seconds": values.get("train_eval_seconds", ""),
        "total_seconds": values.get("total_seconds", ""),
    }


def write_feature_shape_json(
    config: Mapping[str, object],
    output_root: Path,
    shape_metadata: Mapping[str, Mapping[str, object]],
    *,
    family: str,
) -> Path:
    section = dict(config.get(family, {}) or {})
    relative = section.get("feature_shape_json", f"features/{family}_feature_shape.json")
    payload = {
        "family": family,
        "quick_run": bool(config.get("quick_run", False)),
        "configs": dict(shape_metadata.get(family, {})),
    }
    return write_json(payload, output_root / str(relative))


def _failure_reason(error: Exception) -> str:
    if isinstance(error, WaveletConfigError):
        return f"wavelet_config_error: {error}"
    if isinstance(error, ImportError):
        return str(error)
    return str(error)


def _is_allowed_failure(row: Mapping[str, object]) -> bool:
    reason = str(row.get("failure_reason", ""))
    classifier = str(row.get("classifier", ""))
    return (
        (classifier == "xgb" and "xgboost_not_installed" in reason)
        or reason.startswith("wavelet_config_error:")
    )


def print_preflight_stats(preflight: Mapping[str, object]) -> None:
    cache = preflight["cache_audit"]
    summary = preflight["population_summary"]
    assert isinstance(cache, Mapping)
    assert isinstance(summary, Mapping)
    primary = int(summary.get("num_primary_population", 0) or 0)
    positive = int(summary.get("num_hard_hallucination", 0) or 0)
    ratio = 0.0 if primary == 0 else positive / primary
    print(
        "preflight cache_accepted={accepted} cache_entries={entries} "
        "primary_population={primary} positive={positive} positive_ratio={ratio:.6f}".format(
            accepted=str(bool(cache.get("accepted", False))).lower(),
            entries=cache.get("num_entries", 0),
            primary=primary,
            positive=positive,
            ratio=ratio,
        )
    )


def print_final_summary(
    *,
    preflight: Mapping[str, object],
    best_rows: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    summary_path: Path,
    metrics_path: Path,
) -> None:
    summary = preflight["population_summary"]
    assert isinstance(summary, Mapping)
    primary = int(summary.get("num_primary_population", 0) or 0)
    positive = int(summary.get("num_hard_hallucination", 0) or 0)
    ratio = 0.0 if primary == 0 else positive / primary
    print(f"primary_population={primary} positive={positive} positive_ratio={ratio:.6f}")
    for family in ("teacher_bagua", "ours_wavelet", "mind_baseline"):
        winner = _family_best(best_rows, family)
        print(f"{family}_best={_format_winner(winner)}")
    overall = next((row for row in best_rows if row.get("rank_name") == "overall_best"), None)
    print(f"overall_best={_format_winner(overall)}")
    print("failed_configs=" + _failure_list(failures))
    print(f"summary_md={summary_path}")
    print(f"metrics_csv={metrics_path}")


def _family_best(rows: Sequence[Mapping[str, object]], family: str) -> Mapping[str, object] | None:
    aliases = {"mind_baseline": {"mind_baseline", "halp_like_baseline"}}
    allowed = aliases.get(family, {family})
    return next((row for row in rows if str(row.get("method_family", "")) in allowed), None)


def _format_winner(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "none"
    return f"{row.get('config_name', '')}:pr_auc={row.get('pr_auc', '')}"


def _failure_list(failures: Sequence[Mapping[str, object]]) -> str:
    if not failures:
        return "none"
    return ",".join(str(row.get("config_name", "")) for row in failures)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"wavelet_course_run failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
