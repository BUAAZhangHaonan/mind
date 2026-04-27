#!/usr/bin/env python3
"""Orchestrate and report GPU Experiment 3 query-local bank results.

This runner keeps the experiment wiring in one place.  The heavy work belongs
to GPU scripts:

- ``scripts/experiments/build_pooled_bank.py``
- ``scripts/experiments/build_query_local_features.py``
- ``scripts/experiments/build_gpu_linear_probe_features.py``
- ``scripts/experiments/train_gpu_detector.py``

The runner consumes Phase 2 metrics for ``object_cond``, ``shared``,
``shuffled``, and ``no_manifold``.  It computes only the new
``query_local_k30`` and ``linear_probe`` metrics when ``--run-missing`` is set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_MODELS = (
    "qwen3-vl-8b",
    "internvl3.5-8b",
    "llava-onevision-7b",
    "molmo-7b-d-0924",
)
DEFAULT_BENCHMARKS = ("popular", "dash-b")
METHODS = (
    "object_cond",
    "shared",
    "shuffled",
    "query_local_k30",
    "no_manifold",
    "linear_probe",
)
STATIC_PHASE2_METHODS = ("object_cond", "shared", "shuffled", "no_manifold")
DYNAMIC_METHODS = ("query_local_k30", "linear_probe")
BENCHMARK_LABELS = {
    "popular": "POPE popular",
    "dash-b": "DASH-B",
}
BENCHMARK_KEYS_BY_LABEL = {value: key for key, value in BENCHMARK_LABELS.items()}
SPLIT_BY_BENCHMARK = {
    "popular": "popular",
    "dash-b": "main",
}
PHASE2_METHOD_MAP = {
    "object_cond": ("object_conditioned", "full_curve"),
    "shared": ("shared", "full_curve"),
    "shuffled": ("shuffled_object", "full_curve"),
}
METRIC_COLUMNS = (
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "pr_auc",
    "pr_auc_ci_lower",
    "pr_auc_ci_upper",
)
EXPECTED_ROW_COUNT = len(DEFAULT_MODELS) * len(DEFAULT_BENCHMARKS) * len(METHODS)


@dataclass(frozen=True)
class ExperimentJob:
    model: str
    benchmark: str
    method: str
    feature_path: Path
    metrics_path: Path
    predictions_path: Path


def parse_list(value: str, *, allowed: Sequence[str]) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(allowed)
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}")
    return items


def gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(printable)
        return
    subprocess.run(list(command), check=True, env=gpu_env())


def cache_candidates(round_root: Path, model: str, benchmark: str) -> list[Path]:
    if benchmark == "popular":
        return [round_root / "outputs" / "round2_2026_04" / "cache" / model / "pope" / "popular"]
    if benchmark == "dash-b":
        return [
            round_root / "outputs" / "round2_2026_04" / "cache" / model / "dash-b" / "main",
            round_root / "outputs" / "decisive_round_2026_04" / "cache" / model / "dash-b" / "main",
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def resolve_cache_path(round_root: Path, model: str, benchmark: str) -> Path:
    for path in cache_candidates(round_root, model, benchmark):
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing eval cache for {model}/{benchmark}.")


def label_override_candidates(round_root: Path, model: str, benchmark: str) -> list[Path]:
    if benchmark == "popular":
        return [
            round_root / "outputs" / "round2_2026_04" / "features" / f"round2-{model}-popular" / "popular.parquet",
            round_root / "outputs" / "decisive_round_2026_04" / "features" / f"bankid-{model}-popular-object-object-heldout" / "popular.parquet",
        ]
    if benchmark == "dash-b":
        return [
            round_root / "outputs" / "round2_2026_04" / "features" / f"round2-{model}-dash-b" / "main.parquet",
            round_root / "outputs" / "decisive_round_2026_04" / "features" / f"bankid-{model}-dash-b-object" / "main.parquet",
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def resolve_label_override_path(round_root: Path, model: str, benchmark: str) -> Path | None:
    for path in label_override_candidates(round_root, model, benchmark):
        if path.exists():
            return path
    return None


def object_reference_root(round_root: Path, benchmark: str) -> Path:
    if benchmark == "popular":
        return round_root / "outputs" / "round2_2026_04" / "reference_banks"
    if benchmark == "dash-b":
        return round_root / "outputs" / "round2_2026_04" / "reference_banks_dash_b"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def default_feature_path(root: Path, model: str, benchmark: str, method: str) -> Path:
    return root / model / benchmark / f"{method}.parquet"


def default_metrics_path(root: Path, model: str, benchmark: str, method: str) -> Path:
    return root / model / benchmark / f"{method}.json"


def default_predictions_path(root: Path, model: str, benchmark: str, method: str) -> Path:
    return root / model / benchmark / f"{method}.parquet"


def default_pooled_bank_root(output_root: Path, benchmark: str) -> Path:
    return output_root / "pooled_banks" / benchmark


def iter_jobs(
    *,
    models: Sequence[str],
    benchmarks: Sequence[str],
    methods: Sequence[str],
    feature_root: Path,
    metrics_root: Path,
    predictions_root: Path,
) -> Iterable[ExperimentJob]:
    for model in models:
        for benchmark in benchmarks:
            for method in methods:
                if method not in DYNAMIC_METHODS:
                    continue
                yield ExperimentJob(
                    model=model,
                    benchmark=benchmark,
                    method=method,
                    feature_path=default_feature_path(feature_root, model, benchmark, method),
                    metrics_path=default_metrics_path(metrics_root, model, benchmark, method),
                    predictions_path=default_predictions_path(predictions_root, model, benchmark, method),
                )


def build_pooled_bank_command(
    *,
    python_executable: str,
    pooled_builder: Path,
    reference_root: Path,
    pooled_bank_root: Path,
    model: str,
    device: str,
) -> list[str]:
    return [
        python_executable,
        str(pooled_builder),
        "--reference-root",
        str(reference_root),
        "--output-root",
        str(pooled_bank_root),
        "--model-name",
        model,
        "--device",
        device,
    ]


def build_query_local_feature_command(
    job: ExperimentJob,
    *,
    python_executable: str,
    feature_builder: Path,
    cache_path: Path,
    pooled_bank_root: Path,
    device: str,
    batch_size: int,
    reference_chunk_size: int,
    k_neighbors: int,
    label_overrides: Path | None,
) -> list[str]:
    if job.method != "query_local_k30":
        raise ValueError("query-local feature command requires method=query_local_k30.")
    command = [
        python_executable,
        str(feature_builder),
        "--cache-path",
        str(cache_path),
        "--pooled-bank-root",
        str(pooled_bank_root),
        "--model-name",
        job.model,
        "--output-path",
        str(job.feature_path),
        "--k-neighbors",
        str(int(k_neighbors)),
        "--batch-size",
        str(int(batch_size)),
        "--reference-chunk-size",
        str(int(reference_chunk_size)),
        "--device",
        device,
    ]
    if label_overrides is not None:
        command.extend(["--label-overrides", str(label_overrides)])
    return command


def build_linear_feature_command(
    job: ExperimentJob,
    *,
    python_executable: str,
    feature_builder: Path,
    cache_path: Path,
    device: str,
    batch_size: int,
    label_overrides: Path | None,
) -> list[str]:
    if job.method != "linear_probe":
        raise ValueError("linear feature command requires method=linear_probe.")
    command = [
        python_executable,
        str(feature_builder),
        "--cache-path",
        str(cache_path),
        "--model-name",
        job.model,
        "--output-path",
        str(job.feature_path),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        device,
    ]
    if label_overrides is not None:
        command.extend(["--label-overrides", str(label_overrides)])
    return command


def build_detector_command(
    job: ExperimentJob,
    *,
    python_executable: str,
    detector: Path,
    device: str,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
    columns: str,
) -> list[str]:
    return [
        python_executable,
        str(detector),
        "--features",
        str(job.feature_path),
        "--columns",
        columns,
        "--variant-name",
        "full_curve",
        "--split-strategy",
        "image_grouped",
        "--output-json",
        str(job.metrics_path),
        "--predictions-parquet",
        str(job.predictions_path),
        "--device",
        device,
        "--bootstrap-resamples",
        str(int(bootstrap_resamples)),
        "--num-folds",
        str(int(num_folds)),
        "--random-state",
        str(int(random_state)),
        "--max-iter",
        str(int(max_iter)),
    ]


def pooled_bank_exists(pooled_bank_root: Path, model: str) -> bool:
    model_root = pooled_bank_root / model
    if not model_root.exists():
        return False
    return (
        any(model_root.rglob("*.pt"))
        or any(model_root.rglob("*.safetensors"))
        or any(model_root.rglob("*.parquet"))
    )


def ensure_pooled_bank(
    *,
    model: str,
    benchmark: str,
    round_root: Path,
    output_root: Path,
    python_executable: str,
    pooled_builder: Path,
    device: str,
    force: bool,
    dry_run: bool,
) -> Path:
    pooled_root = default_pooled_bank_root(output_root, benchmark)
    if not force and pooled_bank_exists(pooled_root, model):
        return pooled_root
    reference_root = object_reference_root(round_root, benchmark)
    pooled_root.mkdir(parents=True, exist_ok=True)
    run_command(
        build_pooled_bank_command(
            python_executable=python_executable,
            pooled_builder=pooled_builder,
            reference_root=reference_root,
            pooled_bank_root=pooled_root,
            model=model,
            device=device,
        ),
        dry_run=dry_run,
    )
    if not dry_run and not pooled_bank_exists(pooled_root, model):
        raise FileNotFoundError(f"Pooled bank builder did not create a bank under {pooled_root / model}")
    return pooled_root


def detector_columns_for_method(method: str, *, linear_columns: str) -> str:
    if method == "query_local_k30":
        return "all_features"
    if method == "linear_probe":
        return linear_columns
    raise ValueError(f"Unsupported dynamic method: {method}")


def ensure_job_outputs(
    job: ExperimentJob,
    *,
    round_root: Path,
    output_root: Path,
    python_executable: str,
    pooled_builder: Path,
    query_local_feature_builder: Path,
    linear_feature_builder: Path,
    detector: Path,
    device: str,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
    feature_batch_size: int,
    reference_chunk_size: int,
    k_neighbors: int,
    linear_columns: str,
    force_pooled: bool,
    force_features: bool,
    force_metrics: bool,
    dry_run: bool,
) -> None:
    if force_features or not job.feature_path.exists():
        cache_path = resolve_cache_path(round_root, job.model, job.benchmark)
        label_overrides = resolve_label_override_path(round_root, job.model, job.benchmark)
        job.feature_path.parent.mkdir(parents=True, exist_ok=True)
        if job.method == "query_local_k30":
            pooled_root = ensure_pooled_bank(
                model=job.model,
                benchmark=job.benchmark,
                round_root=round_root,
                output_root=output_root,
                python_executable=python_executable,
                pooled_builder=pooled_builder,
                device=device,
                force=force_pooled,
                dry_run=dry_run,
            )
            command = build_query_local_feature_command(
                job,
                python_executable=python_executable,
                feature_builder=query_local_feature_builder,
                cache_path=cache_path,
                pooled_bank_root=pooled_root,
                device=device,
                batch_size=feature_batch_size,
                reference_chunk_size=reference_chunk_size,
                k_neighbors=k_neighbors,
                label_overrides=label_overrides,
            )
        elif job.method == "linear_probe":
            command = build_linear_feature_command(
                job,
                python_executable=python_executable,
                feature_builder=linear_feature_builder,
                cache_path=cache_path,
                device=device,
                batch_size=feature_batch_size,
                label_overrides=label_overrides,
            )
        else:
            raise ValueError(f"Unsupported dynamic method: {job.method}")
        run_command(command, dry_run=dry_run)

    if force_metrics or not job.metrics_path.exists():
        if not dry_run and not job.feature_path.exists():
            raise FileNotFoundError(f"Feature parquet is missing for {job}: {job.feature_path}")
        job.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        job.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            build_detector_command(
                job,
                python_executable=python_executable,
                detector=detector,
                device=device,
                bootstrap_resamples=bootstrap_resamples,
                num_folds=num_folds,
                random_state=random_state,
                max_iter=max_iter,
                columns=detector_columns_for_method(job.method, linear_columns=linear_columns),
            ),
            dry_run=dry_run,
        )


def _read_metric_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [column for column in METRIC_COLUMNS if column not in payload]
    if missing:
        raise ValueError(f"Metric JSON {path} is missing columns: {missing}")
    return payload


def metric_row_from_job(job: ExperimentJob) -> dict[str, object]:
    if not job.metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics JSON: {job.metrics_path}")
    payload = _read_metric_payload(job.metrics_path)
    row = {
        "model": job.model,
        "benchmark": BENCHMARK_LABELS[job.benchmark],
        "benchmark_key": job.benchmark,
        "method": job.method,
        "status": "ok",
        "split_strategy": payload.get("split_strategy", "image_grouped"),
        "feature_source": payload.get("features", str(job.feature_path)),
        "metrics_path": str(job.metrics_path),
        "predictions_path": str(job.predictions_path),
        "n_rows": payload.get("n_rows", ""),
    }
    for column in METRIC_COLUMNS:
        row[column] = float(payload[column])
    return row


def _benchmark_key(row: dict[str, object]) -> str:
    value = str(row.get("benchmark_key") or "")
    if value in BENCHMARK_LABELS:
        return value
    label = str(row.get("benchmark"))
    if label in BENCHMARK_KEYS_BY_LABEL:
        return BENCHMARK_KEYS_BY_LABEL[label]
    raise ValueError(f"Cannot resolve benchmark key from row: {row}")


def _phase2_method_for(row: dict[str, object], *, no_manifold_bank_type: str) -> str | None:
    bank_type = str(row.get("bank_type"))
    variant = str(row.get("variant"))
    for method, expected in PHASE2_METHOD_MAP.items():
        if (bank_type, variant) == expected:
            return method
    if bank_type == no_manifold_bank_type and variant == "no_manifold":
        return "no_manifold"
    return None


def phase2_rows_to_query_local_methods(
    phase2_rows: Sequence[dict[str, object]],
    *,
    no_manifold_bank_type: str = "object_conditioned",
) -> list[dict[str, object]]:
    mapped: list[dict[str, object]] = []
    for row in phase2_rows:
        method = _phase2_method_for(dict(row), no_manifold_bank_type=no_manifold_bank_type)
        if method is None:
            continue
        benchmark_key = _benchmark_key(dict(row))
        mapped_row = {
            "model": str(row["model"]),
            "benchmark": BENCHMARK_LABELS[benchmark_key],
            "benchmark_key": benchmark_key,
            "method": method,
            "status": str(row.get("status", "ok")),
            "split_strategy": str(row.get("split_strategy", "image_grouped")),
            "feature_source": row.get("feature_source", ""),
            "metrics_path": row.get("metrics_path", ""),
            "predictions_path": row.get("predictions_path", ""),
            "n_rows": row.get("n_rows", ""),
        }
        for column in METRIC_COLUMNS:
            mapped_row[column] = float(row[column])
        mapped.append(mapped_row)
    return _sort_rows(mapped, strict=False)


def load_phase2_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 2 CSV: {path}")
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _expected_keys() -> set[tuple[str, str, str]]:
    return {
        (model, BENCHMARK_LABELS[benchmark], method)
        for model in DEFAULT_MODELS
        for benchmark in DEFAULT_BENCHMARKS
        for method in METHODS
    }


def _row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["model"]), str(row["benchmark"]), str(row["method"]))


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_final_rows(rows: Sequence[dict[str, object]]) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_ROW_COUNT} rows, found {len(rows)}.")

    non_ok = [row for row in rows if str(row.get("status", "ok")) != "ok"]
    if non_ok:
        preview = ", ".join(
            f"{row.get('model')}/{row.get('benchmark')}/{row.get('method')}"
            for row in non_ok[:5]
        )
        raise ValueError(f"Found non-ok rows in final table: {preview}")

    keys = [_row_key(dict(row)) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate model/benchmark/method rows.")
    missing = sorted(_expected_keys() - set(keys))
    unexpected = sorted(set(keys) - _expected_keys())
    if missing:
        raise ValueError(f"Missing expected rows: {missing[:5]}")
    if unexpected:
        raise ValueError(f"Unexpected rows: {unexpected[:5]}")

    bad_metrics: list[str] = []
    for row in rows:
        for column in METRIC_COLUMNS:
            if not _is_finite(row.get(column)):
                bad_metrics.append(f"{_row_key(dict(row))}:{column}")
    if bad_metrics:
        raise ValueError(f"Rows have missing metric cells: {bad_metrics[:5]}")


def _sort_rows(rows: Sequence[dict[str, object]], *, strict: bool = True) -> list[dict[str, object]]:
    model_order = {name: index for index, name in enumerate(DEFAULT_MODELS)}
    benchmark_order = {BENCHMARK_LABELS[key]: index for index, key in enumerate(DEFAULT_BENCHMARKS)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    sorted_rows = sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            model_order.get(str(row["model"]), 99),
            benchmark_order.get(str(row["benchmark"]), 99),
            method_order.get(str(row["method"]), 99),
        ),
    )
    if strict:
        validate_final_rows(sorted_rows)
    return sorted_rows


def _format_metric_cell(row: dict[str, object] | None) -> str:
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


def write_query_local_tables(
    rows: Sequence[dict[str, object]],
    *,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    sorted_rows = _sort_rows(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted_rows).to_csv(csv_path, index=False)

    row_map = {_row_key(row): row for row in sorted_rows}
    columns = ["model", "benchmark", *METHODS]
    lines = [
        "# Experiment 3: Query-Local Neighbor Bank",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for model in DEFAULT_MODELS:
        for benchmark_key in DEFAULT_BENCHMARKS:
            benchmark = BENCHMARK_LABELS[benchmark_key]
            values = [
                model,
                benchmark,
                *[_format_metric_cell(row_map.get((model, benchmark, method))) for method in METHODS],
            ]
            lines.append("| " + " | ".join(values) + " |")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("status", "ok")) == "ok"]


def _group_rows(rows: Sequence[dict[str, object]]) -> dict[tuple[str, str], dict[str, dict[str, object]]]:
    groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in _completed_rows(rows):
        groups.setdefault((str(row["model"]), str(row["benchmark"])), {})[str(row["method"])] = row
    return groups


def _count_pr_wins(
    rows: Sequence[dict[str, object]],
    *,
    left: str,
    right: str,
    benchmark: str | None = None,
) -> tuple[int, int]:
    wins = 0
    total = 0
    for (_model, group_benchmark), by_method in _group_rows(rows).items():
        if benchmark is not None and group_benchmark != benchmark:
            continue
        if left not in by_method or right not in by_method:
            continue
        total += 1
        if float(by_method[left]["pr_auc"]) > float(by_method[right]["pr_auc"]):
            wins += 1
    return wins, total


def _best_method_lines(rows: Sequence[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for (model, benchmark), by_method in sorted(_group_rows(rows).items()):
        best = max(by_method.values(), key=lambda row: (float(row["pr_auc"]), float(row["roc_auc"])))
        lines.append(
            f"- {model} / {benchmark}: {best['method']} "
            f"(PR-AUC {float(best['pr_auc']):.4f}, ROC-AUC {float(best['roc_auc']):.4f})"
        )
    return lines


def _linear_probe_gap_summary(rows: Sequence[dict[str, object]]) -> tuple[int, int]:
    closer = 0
    total = 0
    for by_method in _group_rows(rows).values():
        if not {"linear_probe", "query_local_k30", "object_cond"}.issubset(by_method):
            continue
        total += 1
        linear_pr = float(by_method["linear_probe"]["pr_auc"])
        query_gap = abs(linear_pr - float(by_method["query_local_k30"]["pr_auc"]))
        object_gap = abs(linear_pr - float(by_method["object_cond"]["pr_auc"]))
        if query_gap < object_gap:
            closer += 1
    return closer, total


def _rank_variation_lines(rows: Sequence[dict[str, object]]) -> list[str]:
    ranks_by_model: dict[str, list[str]] = {}
    for (model, _benchmark), by_method in _group_rows(rows).items():
        ranking = sorted(by_method.values(), key=lambda row: float(row["pr_auc"]), reverse=True)
        ranks_by_model.setdefault(model, []).append(" > ".join(str(row["method"]) for row in ranking))
    lines: list[str] = []
    for model, rankings in sorted(ranks_by_model.items()):
        unique_count = len(set(rankings))
        lines.append(f"- {model}: {unique_count} distinct PR-AUC rank orders across the two benchmarks.")
    return lines


def write_query_local_analysis(rows: Sequence[dict[str, object]], *, output_path: Path) -> None:
    validate_final_rows(rows)
    query_vs_object = _count_pr_wins(rows, left="query_local_k30", right="object_cond")
    query_vs_shared = _count_pr_wins(rows, left="query_local_k30", right="shared")
    query_vs_shuffled = _count_pr_wins(rows, left="query_local_k30", right="shuffled")
    query_vs_no_manifold_dash = _count_pr_wins(
        rows,
        left="query_local_k30",
        right="no_manifold",
        benchmark="DASH-B",
    )
    closer_to_linear = _linear_probe_gap_summary(rows)

    if query_vs_object[0] > query_vs_object[1] / 2 and query_vs_no_manifold_dash[0] > query_vs_no_manifold_dash[1] / 2:
        verdict = "Query-local selection improves the geometry story, especially if the DASH-B wins are stable."
    elif query_vs_no_manifold_dash[0] == 0:
        verdict = "Query-local selection does not pass the DASH-B no-manifold test."
    else:
        verdict = "The query-local result is mixed. The table should drive the next decision, not the prior expectation."

    lines = [
        "# Experiment 3 Query-Local Bank Analysis",
        "",
        "## Answers",
        "",
        (
            "1. query_local_k30 beats object_cond on "
            f"{query_vs_object[0]}/{query_vs_object[1]} PR-AUC comparisons."
        ),
        (
            "2. query_local_k30 beats shared on "
            f"{query_vs_shared[0]}/{query_vs_shared[1]} comparisons and shuffled on "
            f"{query_vs_shuffled[0]}/{query_vs_shuffled[1]} comparisons."
        ),
        (
            "3. On DASH-B, query_local_k30 beats no_manifold on "
            f"{query_vs_no_manifold_dash[0]}/{query_vs_no_manifold_dash[1]} comparisons."
        ),
        (
            "4. query_local_k30 is closer to linear_probe than object_cond on "
            f"{closer_to_linear[0]}/{closer_to_linear[1]} PR-AUC comparisons."
        ),
        "5. Best method by model and benchmark:",
        *_best_method_lines(rows),
        "6. Rank-order variation by model:",
        *_rank_variation_lines(rows),
        "",
        f"Overall verdict: {verdict}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_rows(
    *,
    phase2_csv: Path,
    dynamic_jobs: Sequence[ExperimentJob],
    no_manifold_bank_type: str,
) -> list[dict[str, object]]:
    phase2_rows = phase2_rows_to_query_local_methods(
        load_phase2_rows(phase2_csv),
        no_manifold_bank_type=no_manifold_bank_type,
    )
    dynamic_rows = [metric_row_from_job(job) for job in dynamic_jobs]
    rows = [*phase2_rows, *dynamic_rows]
    return _sort_rows(rows)


def run_or_report(
    *,
    jobs: Sequence[ExperimentJob],
    round_root: Path,
    output_root: Path,
    python_executable: str,
    pooled_builder: Path,
    query_local_feature_builder: Path,
    linear_feature_builder: Path,
    detector: Path,
    device: str,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
    feature_batch_size: int,
    reference_chunk_size: int,
    k_neighbors: int,
    linear_columns: str,
    force_pooled: bool,
    force_features: bool,
    force_metrics: bool,
    run_missing: bool,
    dry_run: bool,
    phase2_csv: Path,
    no_manifold_bank_type: str,
    csv_path: Path,
    markdown_path: Path,
    analysis_path: Path,
) -> list[dict[str, object]]:
    if run_missing or dry_run:
        for job in jobs:
            ensure_job_outputs(
                job,
                round_root=round_root,
                output_root=output_root,
                python_executable=python_executable,
                pooled_builder=pooled_builder,
                query_local_feature_builder=query_local_feature_builder,
                linear_feature_builder=linear_feature_builder,
                detector=detector,
                device=device,
                bootstrap_resamples=bootstrap_resamples,
                num_folds=num_folds,
                random_state=random_state,
                max_iter=max_iter,
                feature_batch_size=feature_batch_size,
                reference_chunk_size=reference_chunk_size,
                k_neighbors=k_neighbors,
                linear_columns=linear_columns,
                force_pooled=force_pooled,
                force_features=force_features,
                force_metrics=force_metrics,
                dry_run=dry_run,
            )
    if dry_run:
        return []
    rows = collect_rows(
        phase2_csv=phase2_csv,
        dynamic_jobs=jobs,
        no_manifold_bank_type=no_manifold_bank_type,
    )
    write_query_local_tables(rows, csv_path=csv_path, markdown_path=markdown_path)
    write_query_local_analysis(rows, output_path=analysis_path)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--round-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gpu_round_2026_04/query_local_bank"))
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--metrics-root", type=Path, default=None)
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--phase2-csv", type=Path, default=Path("docs/tables/experiment_bank_identity_v2.csv"))
    parser.add_argument("--no-manifold-bank-type", default="object_conditioned")
    parser.add_argument("--csv-path", type=Path, default=Path("docs/tables/experiment_query_local_bank.csv"))
    parser.add_argument("--markdown-path", type=Path, default=Path("docs/tables/experiment_query_local_bank.md"))
    parser.add_argument("--analysis-path", type=Path, default=Path("docs/review/experiment3_query_local_analysis.md"))
    parser.add_argument("--pooled-builder", type=Path, default=Path("scripts/experiments/build_pooled_bank.py"))
    parser.add_argument(
        "--query-local-feature-builder",
        type=Path,
        default=Path("scripts/experiments/build_query_local_features.py"),
    )
    parser.add_argument(
        "--linear-feature-builder",
        type=Path,
        default=Path("scripts/experiments/build_gpu_linear_probe_features.py"),
    )
    parser.add_argument("--detector", type=Path, default=Path("scripts/experiments/train_gpu_detector.py"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--reference-chunk-size", type=int, default=16_384)
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument(
        "--linear-columns",
        default="all_features",
        help="Feature column selector passed to train_gpu_detector.py for linear_probe.",
    )
    parser.add_argument("--force-pooled", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = parse_list(args.models, allowed=DEFAULT_MODELS)
    benchmarks = parse_list(args.benchmarks, allowed=DEFAULT_BENCHMARKS)
    methods = parse_list(args.methods, allowed=METHODS)
    selected_dynamic = tuple(method for method in methods if method in DYNAMIC_METHODS)
    expected_count = len(models) * len(benchmarks) * len(methods)
    if expected_count != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Final query-local reporting requires the full {EXPECTED_ROW_COUNT}-row grid; "
            f"the selected grid has {expected_count} rows."
        )

    feature_root = args.feature_root or args.output_root / "features"
    metrics_root = args.metrics_root or args.output_root / "metrics"
    predictions_root = args.predictions_root or args.output_root / "predictions"
    jobs = list(
        iter_jobs(
            models=models,
            benchmarks=benchmarks,
            methods=selected_dynamic,
            feature_root=feature_root,
            metrics_root=metrics_root,
            predictions_root=predictions_root,
        )
    )
    run_or_report(
        jobs=jobs,
        round_root=args.round_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        pooled_builder=args.pooled_builder,
        query_local_feature_builder=args.query_local_feature_builder,
        linear_feature_builder=args.linear_feature_builder,
        detector=args.detector,
        device=args.device,
        bootstrap_resamples=args.bootstrap_resamples,
        num_folds=args.num_folds,
        random_state=args.random_state,
        max_iter=args.max_iter,
        feature_batch_size=args.feature_batch_size,
        reference_chunk_size=args.reference_chunk_size,
        k_neighbors=args.k_neighbors,
        linear_columns=args.linear_columns,
        force_pooled=args.force_pooled,
        force_features=args.force_features,
        force_metrics=args.force_metrics,
        run_missing=args.run_missing,
        dry_run=args.dry_run,
        phase2_csv=args.phase2_csv,
        no_manifold_bank_type=args.no_manifold_bank_type,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
        analysis_path=args.analysis_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
