#!/usr/bin/env python3
"""Run curvature checks from cached hidden states and reference banks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Any, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)


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
    apply_label_overrides_to_entries,
    apply_label_overrides_to_frame,
    compute_bootstrap_confidence_intervals,
    evaluate_feature_frame,
    feature_columns,
    load_cache_entries,
    load_reference_bank,
)
from mind.evaluation.metrics import compute_object_hallucination_label  # noqa: E402
from mind.manifolds import resolve_reference_scope_key  # noqa: E402


DEFAULT_MODELS = (
    "qwen3-vl-8b",
    "internvl3.5-8b",
    "llava-onevision-7b",
    "molmo-7b-d-0924",
)
DISTANCE_TYPES = (
    "euclidean_pca_residual",
    "centroid_angular",
    "knn_angular_k10",
    "centroid_euclidean",
)
COMPARISON_TYPES = ("full_MIND_saved", "no_manifold")
TABLE_DISTANCE_COLUMNS = (*COMPARISON_TYPES, *DISTANCE_TYPES)
BENCHMARK_LABELS = {
    "popular": "POPE popular",
    "dash-b": "DASH-B",
    "adversarial": "POPE adversarial",
    "repope": "RePOPE",
}
SPLIT_BY_BENCHMARK = {
    "popular": "popular",
    "dash-b": "main",
    "adversarial": "adversarial",
    "repope": "popular",
}
SPLIT_STRATEGY_BY_BENCHMARK = {
    "popular": "image_grouped",
    "dash-b": "image_grouped",
    "adversarial": "row",
    "repope": "row",
}


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(vector, dtype=torch.float32)
    return vector / torch.linalg.norm(vector).clamp_min(1e-12)


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    vectors = torch.as_tensor(vectors, dtype=torch.float32)
    return vectors / torch.linalg.norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)


def centroid_angular_distance(query_vector: torch.Tensor, reference_vectors: torch.Tensor) -> float:
    """Return angular distance from query to the spherical reference centroid."""

    normalized_query = _normalize_vector(query_vector)
    normalized_references = _normalize_rows(reference_vectors)
    centroid = _normalize_vector(normalized_references.mean(dim=0))
    cosine = torch.dot(normalized_query, centroid).clamp(-1.0, 1.0)
    return float(torch.arccos(cosine))


def knn_angular_distance(
    query_vector: torch.Tensor,
    reference_vectors: torch.Tensor,
    *,
    k: int = 10,
) -> float:
    """Return the mean angular distance to the k nearest spherical references."""

    normalized_query = _normalize_vector(query_vector)
    normalized_references = _normalize_rows(reference_vectors)
    cosine = (normalized_references @ normalized_query).clamp(-1.0, 1.0)
    angles = torch.arccos(cosine)
    topk = torch.topk(angles, k=min(int(k), int(angles.numel())), largest=False).values
    return float(topk.mean())


def centroid_euclidean_distance(query_vector: torch.Tensor, reference_vectors: torch.Tensor) -> float:
    """Return Euclidean distance from query to the arithmetic reference centroid."""

    query_vector = torch.as_tensor(query_vector, dtype=torch.float32)
    reference_vectors = torch.as_tensor(reference_vectors, dtype=torch.float32)
    centroid = reference_vectors.mean(dim=0)
    return float(torch.linalg.norm(query_vector - centroid))


def _layer_distances(
    queries: torch.Tensor,
    references: torch.Tensor,
    *,
    distance_name: str,
) -> np.ndarray:
    queries = torch.as_tensor(queries, dtype=torch.float32)
    references = torch.as_tensor(references, dtype=torch.float32)
    if distance_name == "centroid_euclidean":
        centroid = references.mean(dim=0)
        values = torch.linalg.norm(queries - centroid.unsqueeze(0), dim=1)
    elif distance_name == "centroid_angular":
        normalized_queries = _normalize_rows(queries)
        normalized_references = _normalize_rows(references)
        centroid = _normalize_vector(normalized_references.mean(dim=0))
        values = torch.arccos((normalized_queries @ centroid).clamp(-1.0, 1.0))
    elif distance_name == "knn_angular_k10":
        normalized_queries = _normalize_rows(queries)
        normalized_references = _normalize_rows(references)
        cosine = (normalized_queries @ normalized_references.T).clamp(-1.0, 1.0)
        angles = torch.arccos(cosine)
        values = torch.topk(angles, k=min(10, int(references.shape[0])), largest=False, dim=1).values.mean(dim=1)
    else:
        raise ValueError(f"Unsupported distance type: {distance_name}")
    return values.detach().cpu().numpy().astype(np.float32)


def _leave_one_out_distance_stats(reference_vectors: torch.Tensor, *, distance_name: str) -> dict[str, float]:
    vectors = torch.as_tensor(reference_vectors, dtype=torch.float32)
    count = int(vectors.shape[0])
    if count <= 1:
        values = np.asarray([0.0], dtype=np.float32)
    elif distance_name == "centroid_euclidean":
        centroids = (vectors.sum(dim=0, keepdim=True) - vectors) / float(count - 1)
        values = torch.linalg.norm(vectors - centroids, dim=1).detach().cpu().numpy().astype(np.float32)
    elif distance_name == "centroid_angular":
        normalized = _normalize_rows(vectors)
        centroids = normalized.sum(dim=0, keepdim=True) - normalized
        centroids = centroids / float(count - 1)
        centroids = centroids / torch.linalg.norm(centroids, dim=1, keepdim=True).clamp_min(1e-12)
        values = torch.arccos((normalized * centroids).sum(dim=1).clamp(-1.0, 1.0)).detach().cpu().numpy().astype(
            np.float32
        )
    elif distance_name == "knn_angular_k10":
        normalized = _normalize_rows(vectors)
        cosine = (normalized @ normalized.T).clamp(-1.0, 1.0)
        angles = torch.arccos(cosine)
        angles[torch.arange(count), torch.arange(count)] = torch.inf
        values = torch.topk(angles, k=min(10, count - 1), largest=False, dim=1).values.mean(dim=1)
        values = values.detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unsupported distance type: {distance_name}")
    return {
        f"{distance_name}_mean": float(values.mean()),
        f"{distance_name}_std": float(max(float(values.std()), 1e-8)),
    }


def compute_distance_stats(
    reference_bank: dict[str, dict[int, torch.Tensor]],
    *,
    distance_name: str,
) -> dict[str, dict[int, dict[str, float]]]:
    stats: dict[str, dict[int, dict[str, float]]] = {}
    for bank_key, layer_map in reference_bank.items():
        stats[bank_key] = {}
        for layer_index, vectors in layer_map.items():
            stats[bank_key][int(layer_index)] = _leave_one_out_distance_stats(
                vectors,
                distance_name=distance_name,
            )
    return stats


def _json_ready_stats(stats: dict[str, dict[int, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        bank_key: {str(layer_index): values for layer_index, values in layer_map.items()}
        for bank_key, layer_map in stats.items()
    }


def _restore_stats(payload: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[int, dict[str, float]]]:
    return {
        bank_key: {int(layer_index): {str(key): float(value) for key, value in values.items()} for layer_index, values in layer_map.items()}
        for bank_key, layer_map in payload.items()
    }


def load_or_compute_distance_stats(
    reference_bank: dict[str, dict[int, torch.Tensor]],
    *,
    distance_name: str,
    cache_path: Path | None,
) -> dict[str, dict[int, dict[str, float]]]:
    if cache_path is not None and cache_path.exists():
        return _restore_stats(json.loads(cache_path.read_text(encoding="utf-8")))
    stats = compute_distance_stats(reference_bank, distance_name=distance_name)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(_json_ready_stats(stats), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def build_prompt_full_curve_features(
    *,
    raw_curve: np.ndarray,
    selected_layers: Sequence[int],
    layer_stats: dict[int, dict[str, float]],
    distance_name: str,
) -> dict[str, float]:
    raw_curve = np.asarray(raw_curve, dtype=np.float32)
    calibrated = []
    for value, layer_index in zip(raw_curve.tolist(), selected_layers):
        stats = layer_stats[int(layer_index)]
        mean = float(stats[f"{distance_name}_mean"])
        std = max(float(stats[f"{distance_name}_std"]), 1e-8)
        calibrated.append((float(value) - mean) / std)
    calibrated_curve = np.asarray(calibrated, dtype=np.float32)
    slope = np.polyfit(np.arange(calibrated_curve.shape[0], dtype=np.float32), calibrated_curve, deg=1)[0]
    features = {f"raw_drift_{index}": float(value) for index, value in enumerate(raw_curve.tolist())}
    features.update(
        {
            "cal_mean_drift": float(calibrated_curve.mean()),
            "cal_max_drift": float(calibrated_curve.max()),
            "cal_final_drift": float(calibrated_curve[-1]),
            "cal_drift_slope": float(slope),
            "cal_drift_variance": float(calibrated_curve.var()),
        }
    )
    return features


def _metadata_row_from_entry(entry: dict[str, Any]) -> dict[str, object]:
    answer_label = None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
    return {
        "sample_id": str(entry["sample_id"]),
        "image_id": int(entry.get("image_id", -1)),
        "ground_truth_label": int(entry["label"]),
        "answer_label": -1 if answer_label is None else answer_label,
        "label": compute_object_hallucination_label(
            ground_truth_label=int(entry["label"]),
            answer_label=answer_label,
        ),
        "subset": str(entry["subset"]),
        "object_name": str(entry["object_name"]),
    }


def compute_distance_feature_frame(
    *,
    cache_entries: Sequence[dict[str, Any]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    distance_stats: dict[str, dict[int, dict[str, float]]],
    distance_name: str,
    bank_scope: str = "object",
    batch_size: int = 128,
) -> pd.DataFrame:
    prepared: list[dict[str, Any]] = []
    grouped: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    missing: list[str] = []
    for entry in cache_entries:
        object_name = str(entry["object_name"])
        bank_key = resolve_reference_scope_key(object_name, bank_scope)
        selected_layers = tuple(int(layer_index) for layer_index in entry["selected_layers"])
        if bank_key not in reference_bank or bank_key not in distance_stats:
            missing.append(f"{entry['sample_id']}[{bank_key}]")
            continue
        missing_layers = [
            layer_index
            for layer_index in selected_layers
            if layer_index not in reference_bank[bank_key] or layer_index not in distance_stats[bank_key]
        ]
        if missing_layers:
            missing.append(f"{entry['sample_id']}[{bank_key}:{missing_layers}]")
            continue
        prepared_index = len(prepared)
        prepared.append(
            {
                "entry": entry,
                "bank_key": bank_key,
                "selected_layers": selected_layers,
            }
        )
        grouped.setdefault((bank_key, selected_layers), []).append(prepared_index)
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += f", ... (+{len(missing) - 5} more)"
        raise ValueError(f"Missing reference coverage for {len(missing)} entries: {preview}")

    curves: list[np.ndarray | None] = [None for _ in prepared]
    for (bank_key, selected_layers), prepared_indices in grouped.items():
        group_curves = np.empty((len(prepared_indices), len(selected_layers)), dtype=np.float32)
        for offset, layer_index in enumerate(selected_layers):
            references = reference_bank[bank_key][int(layer_index)]
            for start in range(0, len(prepared_indices), batch_size):
                batch_indices = prepared_indices[start : start + batch_size]
                queries = torch.stack(
                    [prepared[index]["entry"]["layer_vectors"][offset] for index in batch_indices],
                    dim=0,
                )
                group_curves[start : start + len(batch_indices), offset] = _layer_distances(
                    queries,
                    references,
                    distance_name=distance_name,
                )
        for local_index, prepared_index in enumerate(prepared_indices):
            curves[prepared_index] = group_curves[local_index]

    rows: list[dict[str, object]] = []
    for index, prepared_entry in enumerate(prepared):
        raw_curve = curves[index]
        if raw_curve is None:
            raise RuntimeError("Missing computed distance curve.")
        entry = prepared_entry["entry"]
        bank_key = str(prepared_entry["bank_key"])
        selected_layers = [int(layer_index) for layer_index in prepared_entry["selected_layers"]]
        rows.append(
            {
                **_metadata_row_from_entry(entry),
                **build_prompt_full_curve_features(
                    raw_curve=raw_curve,
                    selected_layers=selected_layers,
                    layer_stats=distance_stats[bank_key],
                    distance_name=distance_name,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_prompt_full_curve_from_existing_features(features: pd.DataFrame) -> pd.DataFrame:
    metadata = [
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
    raw_columns = sorted(
        [column for column in features.columns if column.startswith("raw_drift_")],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )
    calibrated_columns = sorted(
        [column for column in features.columns if column.startswith("cal_drift_")],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )
    if not raw_columns or not calibrated_columns:
        raise ValueError("Existing feature frame must include raw_drift_* and cal_drift_* columns.")
    calibrated = features[calibrated_columns].to_numpy(dtype=np.float32)
    summary = pd.DataFrame(
        {
            "cal_mean_drift": calibrated.mean(axis=1),
            "cal_max_drift": calibrated.max(axis=1),
            "cal_final_drift": calibrated[:, -1],
            "cal_drift_slope": np.polyfit(np.arange(calibrated.shape[1], dtype=np.float32), calibrated.T, deg=1)[0],
            "cal_drift_variance": calibrated.var(axis=1),
        }
    )
    return pd.concat([features[metadata].reset_index(drop=True), features[raw_columns].reset_index(drop=True), summary], axis=1)


def evaluate_prompt_frame(
    frame: pd.DataFrame,
    *,
    split_strategy: str,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
    num_folds: int = 5,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    columns = feature_columns(frame)
    metrics, results = evaluate_feature_frame(
        frame,
        columns=columns,
        split_strategy=split_strategy,
        random_state=13,
        num_folds=num_folds,
    )
    group_column = "sample_id" if split_strategy == "row" else "image_id"
    intervals = compute_bootstrap_confidence_intervals(
        results,
        group_column=group_column,
        n_resamples=bootstrap_resamples,
        random_state=bootstrap_random_state,
    )
    return metrics, intervals


def _candidate_existing_reports(round_root: Path, model: str, benchmark: str) -> list[Path]:
    if benchmark == "popular":
        names = [f"round2-{model}-popular-final", f"round2-{model}-popular"]
    elif benchmark == "dash-b":
        names = [f"round2-{model}-dash-b"]
    elif benchmark == "adversarial":
        names = [f"round2-{model}-adversarial"]
    elif benchmark == "repope":
        names = [f"round2-{model}-repope"]
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    roots = [
        round_root / "outputs" / "decisive_round_2026_04" / "reports",
        round_root / "outputs" / "round2_2026_04" / "reports",
    ]
    return [root / name for root in roots for name in names]


def _metric_row_from_payload(
    *,
    model: str,
    benchmark: str,
    distance_type: str,
    payload: dict[str, Any],
) -> dict[str, object]:
    intervals = payload.get("confidence_intervals", {})
    roc_ci = intervals.get("roc_auc", {}) if isinstance(intervals, dict) else {}
    pr_ci = intervals.get("pr_auc", {}) if isinstance(intervals, dict) else {}
    return {
        "model": model,
        "benchmark": BENCHMARK_LABELS[benchmark],
        "distance_type": distance_type,
        "roc_auc": float(payload["roc_auc"]),
        "roc_auc_ci_lower": float(roc_ci.get("lower", payload["roc_auc"])),
        "roc_auc_ci_upper": float(roc_ci.get("upper", payload["roc_auc"])),
        "pr_auc": float(payload["pr_auc"]),
        "pr_auc_ci_lower": float(pr_ci.get("lower", payload["pr_auc"])),
        "pr_auc_ci_upper": float(pr_ci.get("upper", payload["pr_auc"])),
        "status": "ok",
    }


def _metric_row_from_prediction_csv(
    *,
    path: Path,
    model: str,
    benchmark: str,
    distance_type: str,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
) -> dict[str, object]:
    frame = pd.read_csv(path)
    intervals = compute_bootstrap_confidence_intervals(
        frame,
        group_column="sample_id" if SPLIT_STRATEGY_BY_BENCHMARK[benchmark] == "row" else "image_id",
        n_resamples=bootstrap_resamples,
        random_state=bootstrap_random_state,
    )
    roc = intervals["roc_auc"]
    pr = intervals["pr_auc"]
    return {
        "model": model,
        "benchmark": BENCHMARK_LABELS[benchmark],
        "distance_type": distance_type,
        "roc_auc": float(roc["point"]),
        "roc_auc_ci_lower": float(roc["lower"]),
        "roc_auc_ci_upper": float(roc["upper"]),
        "pr_auc": float(pr["point"]),
        "pr_auc_ci_lower": float(pr["lower"]),
        "pr_auc_ci_upper": float(pr["upper"]),
        "status": "ok",
        "source_report": str(path.parent.parent),
        "split_strategy": SPLIT_STRATEGY_BY_BENCHMARK[benchmark],
        "n_rows": float(len(frame)),
    }


def load_saved_comparison_rows(
    *,
    round_root: Path,
    model: str,
    benchmark: str,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
) -> list[dict[str, object]]:
    for report_dir in _candidate_existing_reports(round_root, model, benchmark):
        if not report_dir.exists():
            continue
        rows: list[dict[str, object]] = []
        variant_paths = {
            "full_MIND_saved": report_dir / "variant_results" / "full.csv",
            "no_manifold": report_dir / "variant_results" / "no_manifold.csv",
        }
        for distance_type, variant_path in variant_paths.items():
            if not variant_path.exists():
                continue
            rows.append(
                _metric_row_from_prediction_csv(
                    path=variant_path,
                    model=model,
                    benchmark=benchmark,
                    distance_type=distance_type,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_random_state=bootstrap_random_state,
                )
            )
        baselines_path = report_dir / "baselines.json"
        present_distance_types = {str(row["distance_type"]) for row in rows}
        if baselines_path.exists():
            payload = json.loads(baselines_path.read_text(encoding="utf-8"))
            fallback_payloads = {
                "full_MIND_saved": payload.get("full"),
                "no_manifold": payload.get("no_manifold"),
            }
            for distance_type, fallback_payload in fallback_payloads.items():
                if distance_type in present_distance_types or not isinstance(fallback_payload, dict):
                    continue
                rows.append(
                    _metric_row_from_payload(
                        model=model,
                        benchmark=benchmark,
                        distance_type=distance_type,
                        payload=fallback_payload,
                    )
                )
        if rows:
            present_distance_types = {str(row["distance_type"]) for row in rows}
            for distance_type in COMPARISON_TYPES:
                if distance_type not in present_distance_types:
                    rows.append(
                        {
                            "model": model,
                            "benchmark": BENCHMARK_LABELS[benchmark],
                            "distance_type": distance_type,
                            "status": "missing_report_variant",
                            "reason": f"{distance_type} not found under {report_dir}",
                        }
                    )
            return rows
    return [
        {
            "model": model,
            "benchmark": BENCHMARK_LABELS[benchmark],
            "distance_type": distance_type,
            "status": "missing_report",
            "reason": "missing saved report",
        }
        for distance_type in COMPARISON_TYPES
    ]


def _feature_path(round_root: Path, model: str, benchmark: str) -> Path:
    if benchmark == "popular":
        return round_root / "outputs" / "round2_2026_04" / "features" / f"round2-{model}-popular" / "popular.parquet"
    if benchmark == "dash-b":
        return round_root / "outputs" / "round2_2026_04" / "features" / f"round2-{model}-dash-b" / "main.parquet"
    if benchmark == "adversarial":
        return round_root / "outputs" / "round2_2026_04" / "features" / f"round2-{model}-adversarial" / "adversarial.parquet"
    if benchmark == "repope":
        return round_root / "outputs" / "round2_2026_04" / "features" / f"round2-{model}-popular" / "popular.parquet"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _cache_candidates(round_root: Path, model: str, benchmark: str) -> list[Path]:
    if benchmark in {"popular", "repope"}:
        return [round_root / "outputs" / "round2_2026_04" / "cache" / model / "pope" / "popular"]
    if benchmark == "adversarial":
        return [round_root / "outputs" / "round2_2026_04" / "cache" / model / "pope" / "adversarial"]
    if benchmark == "dash-b":
        return [
            round_root / "outputs" / "round2_2026_04" / "cache" / model / "dash-b" / "main",
            round_root / "outputs" / "decisive_round_2026_04" / "cache" / model / "dash-b" / "main",
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _reference_root(round_root: Path, benchmark: str) -> Path:
    if benchmark == "dash-b":
        return round_root / "outputs" / "round2_2026_04" / "reference_banks_dash_b"
    return round_root / "outputs" / "round2_2026_04" / "reference_banks"


def _label_overrides(round_root: Path, benchmark: str) -> Path | None:
    if benchmark == "repope":
        return round_root / "outputs" / "round2_2026_04" / "normalized" / "repope" / "popular.jsonl"
    return None


def _first_existing_path(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _missing_row(*, model: str, benchmark: str, distance_type: str, status: str, reason: str) -> dict[str, object]:
    return {
        "model": model,
        "benchmark": BENCHMARK_LABELS[benchmark],
        "distance_type": distance_type,
        "status": status,
        "reason": reason,
    }


def _evaluation_row(
    *,
    model: str,
    benchmark: str,
    distance_type: str,
    frame: pd.DataFrame,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
) -> dict[str, object]:
    metrics, intervals = evaluate_prompt_frame(
        frame,
        split_strategy=SPLIT_STRATEGY_BY_BENCHMARK[benchmark],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_random_state=bootstrap_random_state,
    )
    return {
        "model": model,
        "benchmark": BENCHMARK_LABELS[benchmark],
        "distance_type": distance_type,
        "roc_auc": float(metrics["roc_auc"]),
        "roc_auc_ci_lower": float(intervals["roc_auc"]["lower"]),
        "roc_auc_ci_upper": float(intervals["roc_auc"]["upper"]),
        "pr_auc": float(metrics["pr_auc"]),
        "pr_auc_ci_lower": float(intervals["pr_auc"]["lower"]),
        "pr_auc_ci_upper": float(intervals["pr_auc"]["upper"]),
        "status": "ok",
        "n_rows": int(len(frame)),
        "split_strategy": SPLIT_STRATEGY_BY_BENCHMARK[benchmark],
    }


def run_model_benchmark(
    *,
    round_root: Path,
    model: str,
    benchmark: str,
    stats_cache_root: Path,
    allow_missing: bool,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
    batch_size: int,
) -> list[dict[str, object]]:
    rows = load_saved_comparison_rows(
        round_root=round_root,
        model=model,
        benchmark=benchmark,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_random_state=bootstrap_random_state,
    )

    feature_path = _feature_path(round_root, model, benchmark)
    label_overrides = _label_overrides(round_root, benchmark)
    if feature_path.exists():
        features = pd.read_parquet(feature_path)
        if label_overrides is not None:
            features = apply_label_overrides_to_frame(features, label_overrides)
        pca_frame = build_prompt_full_curve_from_existing_features(features)
        rows.append(
            _evaluation_row(
                model=model,
                benchmark=benchmark,
                distance_type="euclidean_pca_residual",
                frame=pca_frame,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_random_state=bootstrap_random_state,
            )
        )
    else:
        row = _missing_row(
            model=model,
            benchmark=benchmark,
            distance_type="euclidean_pca_residual",
            status="missing_features",
            reason=f"missing feature file: {feature_path}",
        )
        if not allow_missing:
            raise FileNotFoundError(str(row["reason"]))
        rows.append(row)

    cache_path = _first_existing_path(_cache_candidates(round_root, model, benchmark))
    reference_root = _reference_root(round_root, benchmark)
    if cache_path is None or not reference_root.exists():
        reason = "missing eval cache" if cache_path is None else f"missing reference root: {reference_root}"
        if not allow_missing:
            raise FileNotFoundError(f"{model}/{benchmark}: {reason}")
        rows.extend(
            _missing_row(
                model=model,
                benchmark=benchmark,
                distance_type=distance_type,
                status="missing_cache" if cache_path is None else "missing_reference_bank",
                reason=reason,
            )
            for distance_type in ("centroid_angular", "knn_angular_k10", "centroid_euclidean")
        )
        return rows

    cache_entries = load_cache_entries(cache_path)
    if label_overrides is not None:
        cache_entries = apply_label_overrides_to_entries(cache_entries, label_overrides)
    reference_bank = load_reference_bank(reference_root, model, bank_scope="object")
    for distance_name in ("centroid_angular", "knn_angular_k10", "centroid_euclidean"):
        stats_cache = stats_cache_root / model / benchmark / f"{distance_name}.json"
        distance_stats = load_or_compute_distance_stats(
            reference_bank,
            distance_name=distance_name,
            cache_path=stats_cache,
        )
        frame = compute_distance_feature_frame(
            cache_entries=cache_entries,
            reference_bank=reference_bank,
            distance_stats=distance_stats,
            distance_name=distance_name,
            bank_scope="object",
            batch_size=batch_size,
        )
        rows.append(
            _evaluation_row(
                model=model,
                benchmark=benchmark,
                distance_type=distance_name,
                frame=frame,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_random_state=bootstrap_random_state,
            )
        )
    return rows


def _format_metric_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    if str(row.get("status", "ok")) != "ok":
        reason = str(row.get("reason", "")).strip()
        suffix = f": {reason}" if reason else ""
        return f"{row.get('status')}{suffix}"
    return (
        f"ROC-AUC {float(row['roc_auc']):.4f} "
        f"[{float(row['roc_auc_ci_lower']):.4f}, {float(row['roc_auc_ci_upper']):.4f}]; "
        f"PR-AUC {float(row['pr_auc']):.4f} "
        f"[{float(row['pr_auc_ci_lower']):.4f}, {float(row['pr_auc_ci_upper']):.4f}]"
    )


def write_curvature_tables(
    rows: Sequence[dict[str, object]],
    *,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    row_map = {
        (str(row.get("model")), str(row.get("benchmark")), str(row.get("distance_type"))): dict(row)
        for row in rows
    }
    groups = sorted({(str(row.get("model")), str(row.get("benchmark"))) for row in rows})
    columns = ["model", "benchmark", *TABLE_DISTANCE_COLUMNS]
    lines = [
        "# Experiment 1: Curvature Verification",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for model, benchmark in groups:
        values = [
            model,
            benchmark,
            *[
                _format_metric_cell(row_map.get((model, benchmark, distance_type)))
                for distance_type in TABLE_DISTANCE_COLUMNS
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("status", "ok")) == "ok"]


def _best_angular(row_group: list[dict[str, object]]) -> dict[str, object] | None:
    angular = [row for row in row_group if row["distance_type"] in {"centroid_angular", "knn_angular_k10"}]
    if not angular:
        return None
    return max(angular, key=lambda row: float(row.get("pr_auc", float("-inf"))))


def write_curvature_analysis(rows: Sequence[dict[str, object]], *, output_path: Path) -> None:
    completed = _completed_rows(rows)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in completed:
        groups.setdefault((str(row["model"]), str(row["benchmark"])), []).append(row)

    angular_beats_pca = 0
    angular_beats_no_manifold = 0
    comparable = 0
    no_manifold_comparable = 0
    ranks: dict[str, list[int]] = {distance_type: [] for distance_type in DISTANCE_TYPES}
    hard_deltas: list[float] = []
    popular_deltas: list[float] = []

    for (model, benchmark), group_rows in groups.items():
        by_distance = {str(row["distance_type"]): row for row in group_rows}
        pca = by_distance.get("euclidean_pca_residual")
        best_angular = _best_angular(group_rows)
        if pca is not None and best_angular is not None:
            comparable += 1
            delta = float(best_angular["pr_auc"]) - float(pca["pr_auc"])
            if delta > 0:
                angular_beats_pca += 1
            if benchmark == "POPE popular":
                popular_deltas.append(delta)
            if benchmark in {"DASH-B", "POPE adversarial"}:
                hard_deltas.append(delta)
        no_manifold = by_distance.get("no_manifold")
        if no_manifold is not None and best_angular is not None:
            no_manifold_comparable += 1
            if float(best_angular["pr_auc"]) > float(no_manifold["pr_auc"]):
                angular_beats_no_manifold += 1

        ranked = sorted(
            [row for row in group_rows if row["distance_type"] in DISTANCE_TYPES],
            key=lambda row: (float(row["pr_auc"]), float(row["roc_auc"])),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            ranks[str(row["distance_type"])].append(rank)

    if comparable and angular_beats_pca == 0:
        q1 = "Angular-distance variants do not beat local PCA on any completed comparison."
    else:
        q1 = f"Angular-distance variants beat local PCA on {angular_beats_pca}/{comparable} completed comparisons."
    q2 = (
        "Angular-distance variants beat no_manifold on "
        f"{angular_beats_no_manifold}/{no_manifold_comparable} completed comparisons."
    )
    hard_mean = float(np.mean(hard_deltas)) if hard_deltas else float("nan")
    popular_mean = float(np.mean(popular_deltas)) if popular_deltas else float("nan")
    if hard_deltas and popular_deltas and hard_mean > popular_mean:
        q3 = "Angular distance helps more on the harder benchmark group than on POPE popular."
    elif hard_deltas and popular_deltas:
        q3 = "Angular distance does not help more on the harder benchmark group than on POPE popular."
    else:
        q3 = "The cache coverage is too sparse to compare hard benchmarks against POPE popular cleanly."
    rank_lines = []
    for distance_type, values in ranks.items():
        if values:
            rank_lines.append(f"- {distance_type}: mean rank {float(np.mean(values)):.2f} across {len(values)} comparisons.")
        else:
            rank_lines.append(f"- {distance_type}: no completed comparisons.")
    if comparable and angular_beats_pca == 0:
        verdict = "The evidence points to a mostly linear signal."
    elif comparable and angular_beats_pca > comparable / 2:
        verdict = "The evidence supports more Riemannian geometry work."
    else:
        verdict = "The evidence is mixed; more cache-complete comparisons would be needed."

    missing = [row for row in rows if str(row.get("status", "ok")) != "ok"]
    missing_lines = [
        f"- {row.get('model')} / {row.get('benchmark')} / {row.get('distance_type')}: {row.get('status')} {row.get('reason', '')}".rstrip()
        for row in missing
    ]
    lines = [
        "# Experiment 1 Curvature Analysis",
        "",
        "## Answers",
        "",
        f"1. {q1}",
        f"2. {q2}",
        f"3. {q3}",
        "4. Rank ordering by PR-AUC, then ROC-AUC:",
        *rank_lines,
        f"5. {verdict}",
    ]
    if missing_lines:
        lines.extend(["", "## Missing Coverage", "", *missing_lines])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list(value: str, *, allowed: Sequence[str]) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(allowed)
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}")
    return items


def run_curvature_experiment(
    *,
    round_root: Path,
    models: Sequence[str],
    benchmarks: Sequence[str],
    output_root: Path,
    csv_path: Path,
    markdown_path: Path,
    analysis_path: Path,
    allow_missing: bool,
    bootstrap_resamples: int,
    bootstrap_random_state: int,
    batch_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stats_cache_root = output_root / "distance_stats"
    for model in models:
        for benchmark in benchmarks:
            rows.extend(
                run_model_benchmark(
                    round_root=round_root,
                    model=model,
                    benchmark=benchmark,
                    stats_cache_root=stats_cache_root,
                    allow_missing=allow_missing,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_random_state=bootstrap_random_state,
                    batch_size=batch_size,
                )
            )
            write_curvature_tables(rows, csv_path=csv_path, markdown_path=markdown_path)
            write_curvature_analysis(rows, output_path=analysis_path)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, default=Path("."))
    parser.add_argument("--models", default="all")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geometry_value_2026_04/curvature"))
    parser.add_argument("--csv-path", type=Path, default=Path("docs/tables/experiment_curvature_verification.csv"))
    parser.add_argument("--markdown-path", type=Path, default=Path("docs/tables/experiment_curvature_verification.md"))
    parser.add_argument("--analysis-path", type=Path, default=Path("docs/review/experiment1_curvature_analysis.md"))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-random-state", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = parse_list(args.models, allowed=DEFAULT_MODELS)
    benchmarks = parse_list(args.benchmarks, allowed=tuple(BENCHMARK_LABELS))
    run_curvature_experiment(
        round_root=args.round_root,
        models=models,
        benchmarks=benchmarks,
        output_root=args.output_root,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
        analysis_path=args.analysis_path,
        allow_missing=bool(args.allow_missing),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_random_state=int(args.bootstrap_random_state),
        batch_size=int(args.batch_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
