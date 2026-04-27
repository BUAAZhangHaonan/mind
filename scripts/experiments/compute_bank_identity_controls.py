#!/usr/bin/env python3
"""Run bank-identity controls from cached states and saved reference banks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import types
from typing import Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(REPO_SRC)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _ensure_lightweight_models_module() -> None:
    if "mind.models" in sys.modules:
        return

    yes_no_pattern = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

    def parse_yes_no_answer(text: str) -> int | None:
        match = yes_no_pattern.search(text)
        if match is None:
            return None
        return 1 if match.group(1).lower() == "yes" else 0

    shim = types.ModuleType("mind.models")
    shim.parse_yes_no_answer = parse_yes_no_answer
    sys.modules["mind.models"] = shim


_ensure_lightweight_models_module()

from mind.evaluation.baselines import (  # noqa: E402
    compute_bootstrap_confidence_intervals,
    drift_only_columns,
    evaluate_feature_frame,
    feature_columns,
    load_cache_entries,
    load_reference_bank,
    load_reference_stats,
)
from mind.drift import build_drift_features, calibrate_drift_curve  # noqa: E402
from mind.evaluation.metrics import compute_object_hallucination_label  # noqa: E402
from mind.manifolds import resolve_reference_scope_key  # noqa: E402
from scripts.build_manifolds import save_reference_bank_from_saved_tensors  # noqa: E402
from scripts.compute_drift import build_feature_frame as build_drift_feature_frame  # noqa: E402


DEFAULT_MODELS = (
    "qwen3-vl-8b",
    "internvl3.5-8b",
    "llava-onevision-7b",
    "molmo-7b-d-0924",
)
DEFAULT_BENCHMARKS = ("popular", "dash-b")
BANK_TYPES = {
    "object_conditioned": "object",
    "shared": "shared",
    "shuffled_object": "shuffled_object",
}
VARIANTS = ("drift_only", "no_manifold", "full_curve")
BENCHMARK_LABELS = {
    "popular": "POPE popular",
    "dash-b": "DASH-B",
}
SPLIT_BY_BENCHMARK = {
    "popular": "popular",
    "dash-b": "main",
}


def _sorted_drift_columns(frame: pd.DataFrame, *, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix)],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )


def build_prompt_full_curve_from_existing_features(features: pd.DataFrame) -> pd.DataFrame:
    """Prompt-defined recipe: raw curve plus calibrated summary stats only."""
    metadata_columns = [
        column
        for column in [
            "sample_id",
            "image_id",
            "ground_truth_label",
            "answer_label",
            "label",
            "subset",
            "object_name",
        ]
        if column in features.columns
    ]
    raw_columns = _sorted_drift_columns(features, prefix="raw_drift_")
    calibrated_columns = _sorted_drift_columns(features, prefix="cal_drift_")
    if not raw_columns:
        raise ValueError("Feature frame has no raw_drift_* columns.")
    if calibrated_columns:
        calibrated = features[calibrated_columns].to_numpy(dtype=np.float32)
        summary = pd.DataFrame(
            {
                "cal_mean_drift": (
                    features["cal_mean_drift"].to_numpy(dtype=np.float32)
                    if "cal_mean_drift" in features.columns
                    else calibrated.mean(axis=1)
                ),
                "cal_max_drift": (
                    features["cal_max_drift"].to_numpy(dtype=np.float32)
                    if "cal_max_drift" in features.columns
                    else calibrated.max(axis=1)
                ),
                "cal_final_drift": calibrated[:, -1],
                "cal_drift_slope": np.polyfit(
                    np.arange(calibrated.shape[1], dtype=np.float32),
                    calibrated.T,
                    deg=1,
                )[0],
                "cal_drift_variance": calibrated.var(axis=1),
            },
            index=features.index,
        )
    else:
        required = [
            "cal_mean_drift",
            "cal_max_drift",
            "cal_final_drift",
            "cal_drift_slope",
            "cal_drift_variance",
        ]
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise ValueError(f"Feature frame is missing calibrated summaries: {missing}")
        summary = features[required].copy()
    return pd.concat(
        [
            features[metadata_columns].reset_index(drop=True),
            features[raw_columns].reset_index(drop=True),
            summary.reset_index(drop=True),
        ],
        axis=1,
    )


def parse_list(value: str, *, allowed: Sequence[str]) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(allowed)
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}")
    return items


def object_reference_root(round_root: Path, benchmark: str) -> Path:
    if benchmark == "popular":
        return round_root / "outputs" / "round2_2026_04" / "reference_banks"
    if benchmark == "dash-b":
        return round_root / "outputs" / "round2_2026_04" / "reference_banks_dash_b"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def derived_reference_root(output_root: Path, benchmark: str, bank_type: str) -> Path:
    return output_root / "reference_banks" / benchmark / bank_type


def existing_derived_reference_roots(round_root: Path, benchmark: str, bank_type: str) -> list[Path]:
    if bank_type == "shared":
        name = "reference_banks_shared" if benchmark == "popular" else "reference_banks_dash_b_shared"
    elif bank_type == "shuffled_object":
        name = "reference_banks_shuffled" if benchmark == "popular" else "reference_banks_dash_b_shuffled"
    else:
        return []
    return [round_root / "outputs" / "decisive_round_2026_04" / name]


def resolve_reference_root(
    *,
    round_root: Path,
    output_root: Path,
    model: str,
    benchmark: str,
    bank_type: str,
    shuffle_seed: int,
) -> tuple[Path | None, str | None]:
    source_root = object_reference_root(round_root, benchmark)
    bank_scope = BANK_TYPES[bank_type]
    if bank_scope == "object":
        if not (source_root / model).exists():
            return None, f"missing object reference bank under {source_root / model}"
        return source_root, None
    for existing_root in existing_derived_reference_roots(round_root, benchmark, bank_type):
        if (existing_root / model / "reference_counts.csv").exists():
            return existing_root, None
    target_root = derived_reference_root(output_root, benchmark, bank_type)
    counts_path = target_root / model / "reference_counts.csv"
    if counts_path.exists():
        return target_root, None
    if not (source_root / model).exists():
        return None, f"missing source object reference bank under {source_root / model}"
    save_reference_bank_from_saved_tensors(
        reference_root=source_root,
        output_root=target_root,
        model_name=model,
        bank_scope=bank_scope,
        shuffle_seed=shuffle_seed,
    )
    return target_root, None


def cache_candidates(round_root: Path, model: str, benchmark: str) -> list[Path]:
    if benchmark == "popular":
        return [round_root / "outputs" / "round2_2026_04" / "cache" / model / "pope" / "popular"]
    if benchmark == "dash-b":
        return [
            round_root / "outputs" / "round2_2026_04" / "cache" / model / "dash-b" / "main",
            round_root / "outputs" / "decisive_round_2026_04" / "cache" / model / "dash-b" / "main",
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def resolve_cache_path(round_root: Path, model: str, benchmark: str) -> Path | None:
    for path in cache_candidates(round_root, model, benchmark):
        if path.exists():
            return path
    return None


def bank_experiment_name(model: str, benchmark: str, bank_scope: str) -> str:
    if benchmark == "popular":
        return f"bankid-{model}-popular-{bank_scope}-object-heldout"
    if benchmark == "dash-b":
        return f"bankid-{model}-dash-b-{bank_scope}"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def existing_feature_candidates(
    *,
    round_root: Path,
    output_root: Path,
    model: str,
    benchmark: str,
    bank_scope: str,
) -> list[Path]:
    split = SPLIT_BY_BENCHMARK[benchmark]
    experiment_name = bank_experiment_name(model, benchmark, bank_scope)
    candidates = [
        output_root / "features" / experiment_name / f"{split}.parquet",
        round_root / "outputs" / "decisive_round_2026_04" / "features" / experiment_name / f"{split}.parquet",
        round_root / "outputs" / "round2_2026_04" / "features" / experiment_name / f"{split}.parquet",
    ]
    if bank_scope == "object":
        round2_name = f"round2-{model}-{'popular' if benchmark == 'popular' else 'dash-b'}"
        candidates.extend(
            [
                round_root / "outputs" / "decisive_round_2026_04" / "features" / round2_name / f"{split}.parquet",
                round_root / "outputs" / "round2_2026_04" / "features" / round2_name / f"{split}.parquet",
            ]
        )
    return candidates


def resolve_existing_feature_path(
    *,
    round_root: Path,
    output_root: Path,
    model: str,
    benchmark: str,
    bank_scope: str,
) -> Path | None:
    for path in existing_feature_candidates(
        round_root=round_root,
        output_root=output_root,
        model=model,
        benchmark=benchmark,
        bank_scope=bank_scope,
    ):
        if path.exists():
            return path
    return None


def computed_feature_path(output_root: Path, model: str, benchmark: str, bank_scope: str) -> Path:
    return (
        output_root
        / "features"
        / bank_experiment_name(model, benchmark, bank_scope)
        / f"{SPLIT_BY_BENCHMARK[benchmark]}.parquet"
    )


def load_or_compute_feature_frame(
    *,
    round_root: Path,
    output_root: Path,
    model: str,
    benchmark: str,
    bank_type: str,
    reference_root: Path,
    cache_entries: list[dict[str, object]] | None,
    batch_size: int,
) -> tuple[pd.DataFrame | None, str | None]:
    bank_scope = BANK_TYPES[bank_type]
    existing = resolve_existing_feature_path(
        round_root=round_root,
        output_root=output_root,
        model=model,
        benchmark=benchmark,
        bank_scope=bank_scope,
    )
    if existing is not None:
        return pd.read_parquet(existing), str(existing)
    if cache_entries is None:
        return None, "missing eval cache"
    reference_bank = load_reference_bank(reference_root, model, bank_scope=bank_scope)
    reference_stats = load_reference_stats(reference_root, model, bank_scope=bank_scope)
    frame = build_drift_feature_frame(
        cache_entries=cache_entries,
        reference_bank=reference_bank,
        reference_stats=reference_stats,
        bank_scope=bank_scope,
        batch_size=batch_size,
    )
    output_path = computed_feature_path(output_root, model, benchmark, bank_scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame, str(output_path)


def evaluate_variant_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    bootstrap_resamples: int,
    bootstrap_random_state: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metrics, results = evaluate_feature_frame(
        frame,
        columns=columns,
        split_strategy="image_grouped",
        random_state=13,
        num_folds=5,
    )
    intervals = compute_bootstrap_confidence_intervals(
        results,
        group_column="image_id",
        n_resamples=bootstrap_resamples,
        random_state=bootstrap_random_state,
    )
    return metrics, intervals


def metric_row(
    *,
    model: str,
    benchmark: str,
    bank_type: str,
    variant: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
    bootstrap_resamples: int,
    bootstrap_random_state: int,
    feature_source: str,
    reference_root: Path,
) -> dict[str, object]:
    metrics, intervals = evaluate_variant_frame(
        frame,
        columns=columns,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_random_state=bootstrap_random_state,
    )
    return {
        "model": model,
        "benchmark": BENCHMARK_LABELS[benchmark],
        "bank_type": bank_type,
        "variant": variant,
        "roc_auc": float(metrics["roc_auc"]),
        "roc_auc_ci_lower": float(intervals["roc_auc"]["lower"]),
        "roc_auc_ci_upper": float(intervals["roc_auc"]["upper"]),
        "pr_auc": float(metrics["pr_auc"]),
        "pr_auc_ci_lower": float(intervals["pr_auc"]["lower"]),
        "pr_auc_ci_upper": float(intervals["pr_auc"]["upper"]),
        "status": "ok",
        "n_rows": int(len(frame)),
        "split_strategy": "image_grouped",
        "feature_source": feature_source,
        "reference_root": str(reference_root),
    }


def _format_missing_reference_coverage(missing_entries: list[dict[str, object]]) -> str:
    preview = ", ".join(
        f"{entry['sample_id']}[{entry['reason']}]"
        for entry in missing_entries[:5]
    )
    if len(missing_entries) > 5:
        preview += f", ... (+{len(missing_entries) - 5} more)"
    return f"Missing reference coverage for {len(missing_entries)} cache entries: {preview}"


def build_batched_no_manifold_feature_frame(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_stats: dict[str, dict[int, dict[str, float]]],
    bank_scope: str,
    batch_size: int,
) -> pd.DataFrame:
    """Exact nearest-neighbor residual baseline, computed in batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    missing_entries: list[dict[str, object]] = []
    prepared_entries: list[dict[str, object]] = []
    grouped_indices: dict[tuple[tuple[int, ...], str], list[int]] = {}
    for entry in cache_entries:
        selected_layers = [int(layer) for layer in entry["selected_layers"]]
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
        grouped_indices.setdefault((tuple(selected_layers), bank_key), []).append(len(prepared_entries) - 1)

    if missing_entries:
        raise ValueError(_format_missing_reference_coverage(missing_entries))

    raw_curves: list[np.ndarray | None] = [None for _ in prepared_entries]
    reference_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for (selected_layers_key, bank_key), group_indices in grouped_indices.items():
        selected_layers = list(selected_layers_key)
        for start in range(0, len(group_indices), batch_size):
            batch_indices = group_indices[start : start + batch_size]
            layer_vectors_batch = torch.stack(
                [prepared_entries[index]["entry"]["layer_vectors"] for index in batch_indices],
                dim=0,
            ).to(dtype=torch.float32)
            batch_curve = np.empty((len(batch_indices), len(selected_layers)), dtype=np.float32)
            for offset, layer_index in enumerate(selected_layers):
                cache_key = (bank_key, int(layer_index))
                if cache_key not in reference_cache:
                    reference_vectors = reference_bank[bank_key][layer_index].to(dtype=torch.float32)
                    reference_norms = reference_vectors.square().sum(dim=1).unsqueeze(0)
                    reference_cache[cache_key] = (reference_vectors, reference_norms)
                reference_vectors, reference_norms = reference_cache[cache_key]
                query_vectors = layer_vectors_batch[:, offset, :]
                query_norms = query_vectors.square().sum(dim=1, keepdim=True)
                distance_squares = (
                    query_norms
                    + reference_norms
                    - 2.0 * (query_vectors @ reference_vectors.T)
                ).clamp_min(0.0)
                neighbor_count = min(32, int(reference_vectors.shape[0]))
                topk = torch.sqrt(torch.topk(distance_squares, k=neighbor_count, largest=False, dim=1).values)
                residuals = topk[:, 0] / topk.mean(dim=1).clamp_min(1e-8)
                batch_curve[:, offset] = residuals.detach().cpu().numpy().astype(np.float32)
            for batch_offset, prepared_index in enumerate(batch_indices):
                raw_curves[prepared_index] = batch_curve[batch_offset]

    rows: list[dict[str, object]] = []
    for prepared_index, prepared in enumerate(prepared_entries):
        entry = prepared["entry"]
        selected_layers = prepared["selected_layers"]
        raw_curve = raw_curves[prepared_index]
        if raw_curve is None:
            raise RuntimeError(f"Missing no_manifold curve for {entry['sample_id']}")
        calibrated_curve = calibrate_drift_curve(
            raw_curve,
            selected_layers=selected_layers,
            layer_stats=reference_stats[str(prepared["bank_key"])],
            mean_key="neighbor_residual_mean",
            std_key="neighbor_residual_std",
        )
        features = build_drift_features(raw_curve=raw_curve, calibrated_curve=calibrated_curve)
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "image_id": int(entry.get("image_id", -1)),
                "ground_truth_label": int(entry["label"]),
                "answer_label": -1 if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
                "label": compute_object_hallucination_label(
                    ground_truth_label=int(entry["label"]),
                    answer_label=None if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
                ),
                "subset": entry["subset"],
                "object_name": str(entry["object_name"]),
                **features,
            }
        )
    return pd.DataFrame(rows)


def missing_rows(
    *,
    model: str,
    benchmark: str,
    bank_type: str,
    variants: Sequence[str],
    status: str,
    reason: str,
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "benchmark": BENCHMARK_LABELS[benchmark],
            "bank_type": bank_type,
            "variant": variant,
            "status": status,
            "reason": reason,
        }
        for variant in variants
    ]


def run_model_benchmark_bank(
    *,
    round_root: Path,
    output_root: Path,
    model: str,
    benchmark: str,
    bank_type: str,
    allow_missing: bool,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
    batch_size: int,
    shuffle_seed: int,
) -> list[dict[str, object]]:
    cache_path = resolve_cache_path(round_root, model, benchmark)
    bank_scope = BANK_TYPES[bank_type]
    existing_feature_path = resolve_existing_feature_path(
        round_root=round_root,
        output_root=output_root,
        model=model,
        benchmark=benchmark,
        bank_scope=bank_scope,
    )
    if cache_path is None and existing_feature_path is None:
        return missing_rows(
            model=model,
            benchmark=benchmark,
            bank_type=bank_type,
            variants=VARIANTS,
            status="missing_cache",
            reason="missing eval cache",
        )

    reference_root, reference_error = resolve_reference_root(
        round_root=round_root,
        output_root=output_root,
        model=model,
        benchmark=benchmark,
        bank_type=bank_type,
        shuffle_seed=shuffle_seed,
    )
    if reference_root is None:
        if allow_missing:
            return missing_rows(
                model=model,
                benchmark=benchmark,
                bank_type=bank_type,
                variants=VARIANTS,
                status="missing_reference_bank",
                reason=str(reference_error),
            )
        raise FileNotFoundError(str(reference_error))

    cache_entries = load_cache_entries(cache_path) if cache_path is not None else None
    feature_frame, feature_source = load_or_compute_feature_frame(
        round_root=round_root,
        output_root=output_root,
        model=model,
        benchmark=benchmark,
        bank_type=bank_type,
        reference_root=reference_root,
        cache_entries=cache_entries,
        batch_size=batch_size,
    )

    rows: list[dict[str, object]] = []
    if feature_frame is None:
        rows.extend(
            missing_rows(
                model=model,
                benchmark=benchmark,
                bank_type=bank_type,
                variants=("drift_only", "full_curve"),
                status="missing_cache",
                reason=str(feature_source),
            )
        )
    else:
        try:
            rows.append(
                metric_row(
                    model=model,
                    benchmark=benchmark,
                    bank_type=bank_type,
                    variant="drift_only",
                    frame=feature_frame,
                    columns=drift_only_columns(feature_frame),
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_random_state=bootstrap_random_state,
                    feature_source=str(feature_source),
                    reference_root=reference_root,
                )
            )
            prompt_frame = build_prompt_full_curve_from_existing_features(feature_frame)
            rows.append(
                metric_row(
                    model=model,
                    benchmark=benchmark,
                    bank_type=bank_type,
                    variant="full_curve",
                    frame=prompt_frame,
                    columns=feature_columns(prompt_frame),
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_random_state=bootstrap_random_state,
                    feature_source=str(feature_source),
                    reference_root=reference_root,
                )
            )
        except Exception as error:
            if not allow_missing:
                raise
            rows.extend(
                missing_rows(
                    model=model,
                    benchmark=benchmark,
                    bank_type=bank_type,
                    variants=("drift_only", "full_curve"),
                    status="evaluation_error",
                    reason=str(error),
                )
            )

    if cache_entries is None:
        rows.extend(
            missing_rows(
                model=model,
                benchmark=benchmark,
                bank_type=bank_type,
                variants=("no_manifold",),
                status="missing_cache",
                reason="missing eval cache",
            )
        )
    else:
        try:
            reference_bank = load_reference_bank(reference_root, model, bank_scope=BANK_TYPES[bank_type])
            reference_stats = load_reference_stats(reference_root, model, bank_scope=BANK_TYPES[bank_type])
            no_manifold_frame = build_batched_no_manifold_feature_frame(
                cache_entries=cache_entries,
                reference_bank=reference_bank,
                reference_stats=reference_stats,
                bank_scope=BANK_TYPES[bank_type],
                batch_size=batch_size,
            )
            no_manifold_prompt = build_prompt_full_curve_from_existing_features(no_manifold_frame)
            rows.append(
                metric_row(
                    model=model,
                    benchmark=benchmark,
                    bank_type=bank_type,
                    variant="no_manifold",
                    frame=no_manifold_prompt,
                    columns=feature_columns(no_manifold_prompt),
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_random_state=bootstrap_random_state,
                    feature_source="computed:no_manifold_neighbor_residual",
                    reference_root=reference_root,
                )
            )
        except Exception as error:
            if not allow_missing:
                raise
            rows.extend(
                missing_rows(
                    model=model,
                    benchmark=benchmark,
                    bank_type=bank_type,
                    variants=("no_manifold",),
                    status="evaluation_error",
                    reason=str(error),
                )
            )

    order = {variant: index for index, variant in enumerate(VARIANTS)}
    return sorted(rows, key=lambda row: order.get(str(row["variant"]), 99))


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
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(csv_path, index=False)

    row_map = {
        (str(row["model"]), str(row["benchmark"]), str(row["bank_type"]), str(row["variant"])): dict(row)
        for row in rows
    }
    groups = sorted({key[:3] for key in row_map})
    columns = ["model", "benchmark", "bank_type", *VARIANTS]
    lines = [
        "# Experiment 2: Bank Identity Control",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for model, benchmark, bank_type in groups:
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


def _completed(rows: Sequence[dict[str, object]], *, variant: str | None = None) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("status", "ok")) == "ok" and (variant is None or str(row.get("variant")) == variant)
    ]


def _compare_bank_type(
    rows: Sequence[dict[str, object]],
    *,
    left: str,
    right: str,
) -> tuple[int, int]:
    full_rows = _completed(rows, variant="full_curve")
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


def write_bank_identity_analysis(rows: Sequence[dict[str, object]], *, output_path: Path) -> None:
    object_vs_shared = _compare_bank_type(rows, left="object_conditioned", right="shared")
    object_vs_shuffled = _compare_bank_type(rows, left="object_conditioned", right="shuffled_object")
    shared_vs_shuffled = _compare_bank_type(rows, left="shared", right="shuffled_object")

    completed_rows = _completed(rows)
    by_group: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in completed_rows:
        by_group.setdefault(
            (str(row["model"]), str(row["benchmark"]), str(row["bank_type"])),
            {},
        )[str(row["variant"])] = row
    dash_full_beats_no = 0
    dash_total = 0
    for (_model, benchmark, _bank_type), by_variant in by_group.items():
        if benchmark != "DASH-B" or "full_curve" not in by_variant or "no_manifold" not in by_variant:
            continue
        dash_total += 1
        if float(by_variant["full_curve"]["pr_auc"]) > float(by_variant["no_manifold"]["pr_auc"]):
            dash_full_beats_no += 1

    if (
        object_vs_shared[1]
        and object_vs_shared[0] == object_vs_shared[1]
        and object_vs_shuffled[0] == object_vs_shuffled[1]
    ):
        verdict = "The completed controls support object-conditioned geometry."
    elif shared_vs_shuffled[1] and shared_vs_shuffled[0] > shared_vs_shuffled[1] / 2:
        verdict = "The completed controls point more to grounded reference states than object identity."
    else:
        verdict = (
            "The completed controls do not support object-conditioned manifold geometry as the useful part. "
            "The object label is not reliable, shared pooling is not reliably better than shuffled labels, "
            "and DASH-B usually favors no_manifold over full_curve."
        )

    missing = [row for row in rows if str(row.get("status", "ok")) != "ok"]
    missing_lines = [
        f"- {row.get('model')} / {row.get('benchmark')} / {row.get('bank_type')} / {row.get('variant')}: {row.get('status')} {row.get('reason', '')}".rstrip()
        for row in missing
    ]
    lines = [
        "# Experiment 2 Bank Identity Analysis",
        "",
        "## Answers",
        "",
        (
            "1. No. object_conditioned beats shared on "
            f"{object_vs_shared[0]}/{object_vs_shared[1]} completed full_curve comparisons. "
            "That is helpful in some settings, but it is not consistent."
        ),
        (
            "2. No. object_conditioned beats shuffled_object on "
            f"{object_vs_shuffled[0]}/{object_vs_shuffled[1]} completed full_curve comparisons. "
            "Wrong-label banks often match or beat the object bank, so the object label is not carrying a stable signal."
        ),
        (
            "3. No. shared beats shuffled_object on "
            f"{shared_vs_shuffled[0]}/{shared_vs_shuffled[1]} completed full_curve comparisons. "
            "Pooling grounded states is not enough to explain the wins either."
        ),
        (
            "4. No. On DASH-B, full_curve beats no_manifold on "
            f"{dash_full_beats_no}/{dash_total} completed bank comparisons. "
            "This fails the acid test for the manifold story in the completed DASH-B coverage."
        ),
        f"5. {verdict}",
    ]
    if missing_lines:
        lines.extend(["", "## Missing Coverage", "", *missing_lines])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bank_identity_experiment(
    *,
    round_root: Path,
    models: Sequence[str],
    benchmarks: Sequence[str],
    bank_types: Sequence[str],
    output_root: Path,
    csv_path: Path,
    markdown_path: Path,
    analysis_path: Path,
    allow_missing: bool,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
    batch_size: int,
    shuffle_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for benchmark in benchmarks:
            for bank_type in bank_types:
                rows.extend(
                    run_model_benchmark_bank(
                        round_root=round_root,
                        output_root=output_root,
                        model=model,
                        benchmark=benchmark,
                        bank_type=bank_type,
                        allow_missing=allow_missing,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_random_state=bootstrap_random_state,
                        batch_size=batch_size,
                        shuffle_seed=shuffle_seed,
                    )
                )
                write_bank_identity_tables(rows, csv_path=csv_path, markdown_path=markdown_path)
                write_bank_identity_analysis(rows, output_path=analysis_path)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, default=Path("."))
    parser.add_argument("--models", default="all")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--bank-types", default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geometry_value_2026_04/bank_identity"))
    parser.add_argument("--csv-path", type=Path, default=Path("docs/tables/experiment_bank_identity.csv"))
    parser.add_argument("--markdown-path", type=Path, default=Path("docs/tables/experiment_bank_identity.md"))
    parser.add_argument("--analysis-path", type=Path, default=Path("docs/review/experiment2_bank_analysis.md"))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-random-state", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = parse_list(args.models, allowed=DEFAULT_MODELS)
    benchmarks = parse_list(args.benchmarks, allowed=DEFAULT_BENCHMARKS)
    bank_types = parse_list(args.bank_types, allowed=tuple(BANK_TYPES))
    run_bank_identity_experiment(
        round_root=args.round_root,
        models=models,
        benchmarks=benchmarks,
        bank_types=bank_types,
        output_root=args.output_root,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
        analysis_path=args.analysis_path,
        allow_missing=args.allow_missing,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_random_state=args.bootstrap_random_state,
        batch_size=args.batch_size,
        shuffle_seed=args.shuffle_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
