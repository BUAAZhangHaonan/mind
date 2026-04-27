#!/usr/bin/env python3
"""Orchestrate and report GPU Experiment 2 bank-identity controls.

This script owns only orchestration and reporting.  The heavy math lives in:

- ``scripts/experiments/build_gpu_drift_features.py``
- ``scripts/experiments/train_gpu_detector.py``

Feature-builder CLI contract used here:

```
python scripts/experiments/build_gpu_drift_features.py \
  --cache-path CACHE --reference-root BANK_ROOT --model-name MODEL \
  --bank-scope SCOPE --curve-type CURVE --output-path FEATURE_PATH --device cuda
```

Detector CLI contract used here is the implemented ``train_gpu_detector.py`` CLI.
For ``no_manifold`` features the detector is trained with the full-curve column
recipe, because the variant name describes the distance source, not the column
set.
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
BANK_TYPES = ("object_conditioned", "shared", "shuffled_object")
VARIANTS = ("drift_only", "no_manifold", "full_curve")
BENCHMARK_LABELS = {
    "popular": "POPE popular",
    "dash-b": "DASH-B",
}
BANK_SCOPE_BY_TYPE = {
    "object_conditioned": "object",
    "shared": "shared",
    "shuffled_object": "shuffled_object",
}
SPLIT_BY_BENCHMARK = {
    "popular": "popular",
    "dash-b": "main",
}
METRIC_COLUMNS = (
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "pr_auc",
    "pr_auc_ci_lower",
    "pr_auc_ci_upper",
)
EXPECTED_ROW_COUNT = len(DEFAULT_MODELS) * len(DEFAULT_BENCHMARKS) * len(BANK_TYPES) * len(VARIANTS)


@dataclass(frozen=True)
class ExperimentJob:
    model: str
    benchmark: str
    bank_type: str
    variant: str
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


def detector_variant_name(variant: str) -> str:
    if variant == "drift_only":
        return "drift_only"
    if variant in {"no_manifold", "full_curve"}:
        return "full_curve"
    raise ValueError(f"Unsupported variant: {variant}")


def default_feature_path(root: Path, model: str, benchmark: str, bank_type: str, variant: str) -> Path:
    feature_source = "no_manifold" if variant == "no_manifold" else "local_pca"
    return root / model / benchmark / bank_type / f"{feature_source}.parquet"


def default_metrics_path(root: Path, model: str, benchmark: str, bank_type: str, variant: str) -> Path:
    return root / model / benchmark / bank_type / f"{variant}.json"


def default_predictions_path(root: Path, model: str, benchmark: str, bank_type: str, variant: str) -> Path:
    return root / model / benchmark / bank_type / f"{variant}.parquet"


def curve_type_for_variant(variant: str) -> str:
    if variant == "no_manifold":
        return "no_manifold"
    if variant in {"drift_only", "full_curve"}:
        return "local_pca"
    raise ValueError(f"Unsupported variant: {variant}")


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


def reference_root_for_job(round_root: Path, output_root: Path, job: ExperimentJob) -> Path:
    if job.bank_type == "object_conditioned":
        return object_reference_root(round_root, job.benchmark)
    return output_root / "reference_banks" / job.benchmark / job.bank_type


def ensure_reference_root(
    job: ExperimentJob,
    *,
    round_root: Path,
    output_root: Path,
    python_executable: str,
    build_manifolds: Path,
    k_neighbors: int,
    shuffle_seed: int,
    dry_run: bool,
) -> Path:
    reference_root = reference_root_for_job(round_root, output_root, job)
    counts_path = reference_root / job.model / "reference_counts.csv"
    if counts_path.exists():
        return reference_root
    if job.bank_type == "object_conditioned":
        raise FileNotFoundError(f"Missing object-conditioned bank counts: {counts_path}")

    source_root = object_reference_root(round_root, job.benchmark)
    bank_scope = BANK_SCOPE_BY_TYPE[job.bank_type]
    command = [
        python_executable,
        str(build_manifolds),
        "--reference-bank-root",
        str(source_root),
        "--output-root",
        str(reference_root),
        "--model-name",
        job.model,
        "--k-neighbors",
        str(int(k_neighbors)),
        "--bank-scope",
        bank_scope,
        "--shuffle-seed",
        str(int(shuffle_seed)),
    ]
    run_command(command, dry_run=dry_run)
    if not dry_run and not counts_path.exists():
        raise FileNotFoundError(f"Reference bank build did not create {counts_path}")
    return reference_root


def iter_jobs(
    *,
    models: Sequence[str],
    benchmarks: Sequence[str],
    bank_types: Sequence[str],
    variants: Sequence[str],
    feature_root: Path,
    metrics_root: Path,
    predictions_root: Path,
) -> Iterable[ExperimentJob]:
    for model in models:
        for benchmark in benchmarks:
            for bank_type in bank_types:
                for variant in variants:
                    yield ExperimentJob(
                        model=model,
                        benchmark=benchmark,
                        bank_type=bank_type,
                        variant=variant,
                        feature_path=default_feature_path(feature_root, model, benchmark, bank_type, variant),
                        metrics_path=default_metrics_path(metrics_root, model, benchmark, bank_type, variant),
                        predictions_path=default_predictions_path(predictions_root, model, benchmark, bank_type, variant),
                    )


def build_feature_command(
    job: ExperimentJob,
    *,
    python_executable: str,
    feature_builder: Path,
    cache_path: Path,
    reference_root: Path,
    device: str,
    batch_size: int,
    reference_chunk_size: int,
    label_overrides: Path | None,
) -> list[str]:
    command = [
        python_executable,
        str(feature_builder),
        "--cache-path",
        str(cache_path),
        "--reference-root",
        str(reference_root),
        "--model-name",
        job.model,
        "--output-path",
        str(job.feature_path),
        "--bank-scope",
        BANK_SCOPE_BY_TYPE[job.bank_type],
        "--curve-type",
        curve_type_for_variant(job.variant),
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
) -> list[str]:
    return [
        python_executable,
        str(detector),
        "--features",
        str(job.feature_path),
        "--variant-name",
        detector_variant_name(job.variant),
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


def _gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return
    subprocess.run(command, check=True, env=_gpu_env())


def ensure_job_outputs(
    job: ExperimentJob,
    *,
    round_root: Path,
    output_root: Path,
    python_executable: str,
    feature_builder: Path,
    build_manifolds: Path,
    detector: Path,
    device: str,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
    feature_batch_size: int,
    reference_chunk_size: int,
    k_neighbors: int,
    shuffle_seed: int,
    force_features: bool,
    force_metrics: bool,
    dry_run: bool,
) -> None:
    if force_features or not job.feature_path.exists():
        cache_path = resolve_cache_path(round_root, job.model, job.benchmark)
        label_overrides = resolve_label_override_path(round_root, job.model, job.benchmark)
        reference_root = ensure_reference_root(
            job,
            round_root=round_root,
            output_root=output_root,
            python_executable=python_executable,
            build_manifolds=build_manifolds,
            k_neighbors=k_neighbors,
            shuffle_seed=shuffle_seed,
            dry_run=dry_run,
        )
        job.feature_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            build_feature_command(
                job,
                python_executable=python_executable,
                feature_builder=feature_builder,
                cache_path=cache_path,
                reference_root=reference_root,
                device=device,
                batch_size=feature_batch_size,
                reference_chunk_size=reference_chunk_size,
                label_overrides=label_overrides,
            ),
            dry_run=dry_run,
        )
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
        "bank_type": job.bank_type,
        "variant": job.variant,
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


def _expected_keys() -> set[tuple[str, str, str, str]]:
    return {
        (model, BENCHMARK_LABELS[benchmark], bank_type, variant)
        for model in DEFAULT_MODELS
        for benchmark in DEFAULT_BENCHMARKS
        for bank_type in BANK_TYPES
        for variant in VARIANTS
    }


def _row_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["model"]),
        str(row["benchmark"]),
        str(row["bank_type"]),
        str(row["variant"]),
    )


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
            f"{row.get('model')}/{row.get('benchmark')}/{row.get('bank_type')}/{row.get('variant')}"
            for row in non_ok[:5]
        )
        raise ValueError(f"Found non-ok rows in final table: {preview}")

    keys = [_row_key(dict(row)) for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate model/benchmark/bank/variant rows.")
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


def _sort_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    model_order = {name: index for index, name in enumerate(DEFAULT_MODELS)}
    benchmark_order = {BENCHMARK_LABELS[key]: index for index, key in enumerate(DEFAULT_BENCHMARKS)}
    bank_order = {name: index for index, name in enumerate(BANK_TYPES)}
    variant_order = {name: index for index, name in enumerate(VARIANTS)}
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            model_order.get(str(row["model"]), 99),
            benchmark_order.get(str(row["benchmark"]), 99),
            bank_order.get(str(row["bank_type"]), 99),
            variant_order.get(str(row["variant"]), 99),
        ),
    )


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


def write_bank_identity_tables(
    rows: Sequence[dict[str, object]],
    *,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    validate_final_rows(rows)
    sorted_rows = _sort_rows(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted_rows).to_csv(csv_path, index=False)

    row_map = {_row_key(row): row for row in sorted_rows}
    columns = ["model", "benchmark", "bank_type", *VARIANTS]
    lines = [
        "# Experiment 2: GPU Bank Identity Control",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for model in DEFAULT_MODELS:
        for benchmark_key in DEFAULT_BENCHMARKS:
            benchmark = BENCHMARK_LABELS[benchmark_key]
            for bank_type in BANK_TYPES:
                values = [
                    model,
                    benchmark,
                    bank_type,
                    *[
                        _format_metric_cell(row_map.get((model, benchmark, bank_type, variant)))
                        for variant in VARIANTS
                    ],
                ]
                lines.append("| " + " | ".join(values) + " |")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_rows(rows: Sequence[dict[str, object]], *, variant: str | None = None) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("status", "ok")) == "ok" and (variant is None or str(row.get("variant")) == variant)
    ]


def _compare_bank_type(rows: Sequence[dict[str, object]], *, left: str, right: str) -> tuple[int, int]:
    full_rows = _completed_rows(rows, variant="full_curve")
    groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in full_rows:
        groups.setdefault((str(row["model"]), str(row["benchmark"])), {})[str(row["bank_type"])] = row
    wins = 0
    total = 0
    for by_bank in groups.values():
        if left not in by_bank or right not in by_bank:
            continue
        total += 1
        if float(by_bank[left]["pr_auc"]) > float(by_bank[right]["pr_auc"]):
            wins += 1
    return wins, total


def _dash_full_beats_no_manifold(rows: Sequence[dict[str, object]]) -> tuple[int, int]:
    completed_rows = _completed_rows(rows)
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in completed_rows:
        groups.setdefault(
            (str(row["model"]), str(row["benchmark"]), str(row["bank_type"])),
            {},
        )[str(row["variant"])] = row
    wins = 0
    total = 0
    for (_model, benchmark, _bank_type), by_variant in groups.items():
        if benchmark != "DASH-B" or "full_curve" not in by_variant or "no_manifold" not in by_variant:
            continue
        total += 1
        if float(by_variant["full_curve"]["pr_auc"]) > float(by_variant["no_manifold"]["pr_auc"]):
            wins += 1
    return wins, total


def _row_lookup(rows: Sequence[dict[str, object]]) -> dict[tuple[str, str, str, str], dict[str, object]]:
    return {
        _row_key(dict(row)): dict(row)
        for row in rows
        if str(row.get("status", "ok")) == "ok" and _is_finite(row.get("roc_auc"))
    }


def _ci_overlap(left: dict[str, object], right: dict[str, object], *, metric: str) -> bool:
    left_low = float(left[f"{metric}_ci_lower"])
    left_high = float(left[f"{metric}_ci_upper"])
    right_low = float(right[f"{metric}_ci_lower"])
    right_high = float(right[f"{metric}_ci_upper"])
    return max(left_low, right_low) <= min(left_high, right_high)


def find_roc_discrepancies(
    gpu_rows: Sequence[dict[str, object]],
    cpu_rows: Sequence[dict[str, object]],
    *,
    threshold: float = 0.02,
) -> list[dict[str, object]]:
    gpu_lookup = _row_lookup(gpu_rows)
    cpu_lookup = _row_lookup(cpu_rows)
    discrepancies: list[dict[str, object]] = []
    for key in sorted(gpu_lookup):
        if key not in cpu_lookup:
            continue
        gpu = gpu_lookup[key]
        cpu = cpu_lookup[key]
        difference = abs(float(gpu["roc_auc"]) - float(cpu["roc_auc"]))
        if difference <= threshold or _ci_overlap(gpu, cpu, metric="roc_auc"):
            continue
        discrepancies.append(
            {
                "model": key[0],
                "benchmark": key[1],
                "bank_type": key[2],
                "variant": key[3],
                "gpu_roc_auc": float(gpu["roc_auc"]),
                "cpu_roc_auc": float(cpu["roc_auc"]),
                "difference": round(difference, 10),
                "gpu_ci": (float(gpu["roc_auc_ci_lower"]), float(gpu["roc_auc_ci_upper"])),
                "cpu_ci": (float(cpu["roc_auc_ci_lower"]), float(cpu["roc_auc_ci_upper"])),
            }
        )
    return discrepancies


def load_cpu_baseline_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _best_full_curve_by_group(rows: Sequence[dict[str, object]]) -> list[str]:
    full_rows = _completed_rows(rows, variant="full_curve")
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in full_rows:
        groups.setdefault((str(row["model"]), str(row["benchmark"])), []).append(row)
    lines = []
    for (model, benchmark), candidates in sorted(groups.items()):
        best = max(candidates, key=lambda row: (float(row["pr_auc"]), float(row["roc_auc"])))
        lines.append(
            f"- {model} / {benchmark}: {best['bank_type']} "
            f"(PR-AUC {float(best['pr_auc']):.4f}, ROC-AUC {float(best['roc_auc']):.4f})"
        )
    return lines


def write_bank_identity_analysis(
    rows: Sequence[dict[str, object]],
    *,
    cpu_rows: Sequence[dict[str, object]],
    baseline_csv: Path,
    output_path: Path,
) -> None:
    validate_final_rows(rows)
    object_vs_shared = _compare_bank_type(rows, left="object_conditioned", right="shared")
    object_vs_shuffled = _compare_bank_type(rows, left="object_conditioned", right="shuffled_object")
    shared_vs_shuffled = _compare_bank_type(rows, left="shared", right="shuffled_object")
    dash_full = _dash_full_beats_no_manifold(rows)
    discrepancies = find_roc_discrepancies(rows, cpu_rows)

    if object_vs_shared[0] == object_vs_shared[1] and object_vs_shuffled[0] == object_vs_shuffled[1]:
        conclusion = "The complete v2 table supports object-conditioned geometry more strongly than the CPU table."
    elif dash_full[0] > dash_full[1] / 2:
        conclusion = "The complete v2 table keeps geometry alive, but the bank identity story is still mixed."
    else:
        conclusion = (
            "The complete v2 table does not rescue object-conditioned banks. "
            "The useful signal is still not tied reliably to the object label."
        )

    if cpu_rows:
        discrepancy_intro = f"Compared against `{baseline_csv.as_posix()}`."
        discrepancy_lines = [
            (
                f"- {item['model']} / {item['benchmark']} / {item['bank_type']} / {item['variant']}: "
                f"GPU ROC-AUC {item['gpu_roc_auc']:.4f} vs CPU {item['cpu_roc_auc']:.4f} "
                f"(diff {item['difference']:.4f})"
            )
            for item in discrepancies
        ] or ["- None under the >0.02 ROC-AUC and non-overlapping-CI rule."]
    else:
        discrepancy_intro = f"No CPU baseline CSV was found at `{baseline_csv.as_posix()}`."
        discrepancy_lines = ["- CPU-vs-GPU discrepancy checks were skipped."]

    lines = [
        "# Experiment 2 Bank Identity Analysis v2",
        "",
        "## Answers",
        "",
        (
            "1. object_conditioned beats shared on "
            f"{object_vs_shared[0]}/{object_vs_shared[1]} full_curve comparisons. "
            "This is the direct test of object-specific geometry."
        ),
        (
            "2. object_conditioned beats shuffled_object on "
            f"{object_vs_shuffled[0]}/{object_vs_shuffled[1]} full_curve comparisons. "
            "This checks whether the object label itself matters."
        ),
        (
            "3. shared beats shuffled_object on "
            f"{shared_vs_shuffled[0]}/{shared_vs_shuffled[1]} full_curve comparisons. "
            "This checks whether pooled grounded states help more than wrong labels."
        ),
        (
            "4. On DASH-B, full_curve beats no_manifold on "
            f"{dash_full[0]}/{dash_full[1]} bank comparisons. "
            "This is the acid test for the manifold story."
        ),
        f"5. {conclusion}",
        "",
        "## Best Full-Curve Bank by Model and Benchmark",
        "",
        *_best_full_curve_by_group(rows),
        "",
        "## CPU-vs-GPU ROC-AUC Discrepancies",
        "",
        discrepancy_intro,
        "",
        *discrepancy_lines,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_rows(jobs: Sequence[ExperimentJob]) -> list[dict[str, object]]:
    rows = [metric_row_from_job(job) for job in jobs]
    validate_final_rows(rows)
    return _sort_rows(rows)


def run_or_report(
    *,
    jobs: Sequence[ExperimentJob],
    round_root: Path,
    output_root: Path,
    python_executable: str,
    feature_builder: Path,
    build_manifolds: Path,
    detector: Path,
    device: str,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
    feature_batch_size: int,
    reference_chunk_size: int,
    k_neighbors: int,
    shuffle_seed: int,
    force_features: bool,
    force_metrics: bool,
    run_missing: bool,
    dry_run: bool,
    csv_path: Path,
    markdown_path: Path,
    analysis_path: Path,
    baseline_csv: Path,
) -> list[dict[str, object]]:
    if run_missing or dry_run:
        for job in jobs:
            ensure_job_outputs(
                job,
                round_root=round_root,
                output_root=output_root,
                python_executable=python_executable,
                feature_builder=feature_builder,
                build_manifolds=build_manifolds,
                detector=detector,
                device=device,
                bootstrap_resamples=bootstrap_resamples,
                num_folds=num_folds,
                random_state=random_state,
                max_iter=max_iter,
                feature_batch_size=feature_batch_size,
                reference_chunk_size=reference_chunk_size,
                k_neighbors=k_neighbors,
                shuffle_seed=shuffle_seed,
                force_features=force_features,
                force_metrics=force_metrics,
                dry_run=dry_run,
            )
    if dry_run:
        return []
    rows = collect_rows(jobs)
    cpu_rows = load_cpu_baseline_rows(baseline_csv)
    write_bank_identity_tables(rows, csv_path=csv_path, markdown_path=markdown_path)
    write_bank_identity_analysis(
        rows,
        cpu_rows=cpu_rows,
        baseline_csv=baseline_csv,
        output_path=analysis_path,
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--bank-types", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--round-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gpu_round_2026_04/bank_identity"))
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--metrics-root", type=Path, default=None)
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=Path("docs/tables/experiment_bank_identity_v2.csv"))
    parser.add_argument("--markdown-path", type=Path, default=Path("docs/tables/experiment_bank_identity_v2.md"))
    parser.add_argument("--analysis-path", type=Path, default=Path("docs/review/experiment2_bank_analysis_v2.md"))
    parser.add_argument("--baseline-csv", type=Path, default=Path("docs/tables/experiment_bank_identity.csv"))
    parser.add_argument("--feature-builder", type=Path, default=Path("scripts/experiments/build_gpu_drift_features.py"))
    parser.add_argument("--build-manifolds", type=Path, default=Path("scripts/build_manifolds.py"))
    parser.add_argument("--detector", type=Path, default=Path("scripts/experiments/train_gpu_detector.py"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--reference-chunk-size", type=int, default=16_384)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=13)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = parse_list(args.models, allowed=DEFAULT_MODELS)
    benchmarks = parse_list(args.benchmarks, allowed=DEFAULT_BENCHMARKS)
    bank_types = parse_list(args.bank_types, allowed=BANK_TYPES)
    variants = parse_list(args.variants, allowed=VARIANTS)
    feature_root = args.feature_root or args.output_root / "features"
    metrics_root = args.metrics_root or args.output_root / "metrics"
    predictions_root = args.predictions_root or args.output_root / "predictions"
    jobs = list(
        iter_jobs(
            models=models,
            benchmarks=benchmarks,
            bank_types=bank_types,
            variants=variants,
            feature_root=feature_root,
            metrics_root=metrics_root,
            predictions_root=predictions_root,
        )
    )
    expected_count = len(models) * len(benchmarks) * len(bank_types) * len(variants)
    if expected_count != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Final v2 reporting requires the full {EXPECTED_ROW_COUNT}-row grid; "
            f"the selected grid has {expected_count} rows."
        )
    run_or_report(
        jobs=jobs,
        round_root=args.round_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        feature_builder=args.feature_builder,
        build_manifolds=args.build_manifolds,
        detector=args.detector,
        device=args.device,
        bootstrap_resamples=args.bootstrap_resamples,
        num_folds=args.num_folds,
        random_state=args.random_state,
        max_iter=args.max_iter,
        feature_batch_size=args.feature_batch_size,
        reference_chunk_size=args.reference_chunk_size,
        k_neighbors=args.k_neighbors,
        shuffle_seed=args.shuffle_seed,
        force_features=args.force_features,
        force_metrics=args.force_metrics,
        run_missing=args.run_missing,
        dry_run=args.dry_run,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
        analysis_path=args.analysis_path,
        baseline_csv=args.baseline_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
