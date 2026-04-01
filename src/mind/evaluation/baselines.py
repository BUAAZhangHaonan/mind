"""Baseline and ablation helpers for MIND."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from mind.detectors import fit_logistic_detector
from mind.drift import build_drift_features, calibrate_drift_curve
from mind.manifolds import resolve_reference_scope_key

from .metrics import compute_binary_metrics, compute_object_hallucination_label


METADATA_COLUMNS = {
    "sample_id",
    "image_id",
    "label",
    "subset",
    "object_name",
    "ground_truth_label",
    "answer_label",
    "fold",
}

GROUP_COLUMN_BY_STRATEGY = {
    "image_grouped": "image_id",
    "object_heldout": "object_name",
}


def load_reference_bank(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, torch.Tensor]] = {}
    model_root = reference_root / model_name
    for layer_path in model_root.glob("*/layer-*.pt"):
        object_name = layer_path.parent.name
        if bank_scope == "object" and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        layer_index = int(layer_path.stem.split("-")[-1])
        bank.setdefault(object_name, {})[layer_index] = torch.load(layer_path, weights_only=False)
    return bank


def load_reference_stats(
    reference_root: Path,
    model_name: str,
    *,
    bank_scope: str = "object",
) -> dict[str, dict[int, dict[str, float]]]:
    stats_map: dict[str, dict[int, dict[str, float]]] = {}
    model_root = reference_root / model_name
    for stats_path in model_root.glob("*/stats.pt"):
        object_name = stats_path.parent.name
        if bank_scope == "object" and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        payload = torch.load(stats_path, weights_only=False)
        stats_map[object_name] = {
            int(layer_index): {str(key): float(value) for key, value in layer_stats.items()}
            for layer_index, layer_stats in payload.items()
        }
    return stats_map


def load_cache_entries(cache_path: Path) -> list[dict[str, object]]:
    if cache_path.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=False))
        return entries
    return list(torch.load(cache_path, weights_only=False))


def _format_missing_reference_coverage(missing_entries: list[dict[str, object]]) -> str:
    preview = ", ".join(
        f"{entry['sample_id']}[{entry['reason']}]"
        for entry in missing_entries[:5]
    )
    if len(missing_entries) > 5:
        preview += f", ... (+{len(missing_entries) - 5} more)"
    return f"Missing reference coverage for {len(missing_entries)} cache entries: {preview}"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def drift_only_columns(frame: pd.DataFrame) -> list[str]:
    keep = {"raw_max_drift", "raw_mean_drift", "raw_peak_layer_index"}
    return [
        column
        for column in frame.columns
        if column.startswith("raw_drift_") or column in keep
    ]


def build_raw_model_yes_no_baseline(cache_entries: Sequence[dict[str, object]]) -> dict[str, object]:
    rows_total = len(cache_entries)
    parsed_entries = [entry for entry in cache_entries if entry.get("parsed_answer") is not None]
    rows_parsed = len(parsed_entries)
    rows_unparsed = rows_total - rows_parsed
    hallucination_positives = sum(
        compute_object_hallucination_label(
            ground_truth_label=int(entry["label"]),
            answer_label=None if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
        )
        for entry in cache_entries
    )
    yes_no_metrics = compute_binary_metrics(
        y_true=[int(entry["label"]) for entry in parsed_entries],
        y_pred=[int(entry["parsed_answer"]) for entry in parsed_entries],
        y_score=[int(entry["parsed_answer"]) for entry in parsed_entries],
    )
    return {
        "rows_total": rows_total,
        "rows_parsed": rows_parsed,
        "rows_unparsed": rows_unparsed,
        "hallucination_positives": int(hallucination_positives),
        "yes_no": yes_no_metrics,
    }


def _normalized_neighbor_residual(
    query_vector: torch.Tensor,
    reference_vectors: torch.Tensor,
    *,
    k_neighbors: int = 32,
) -> float:
    reference_vectors = reference_vectors.to(dtype=torch.float32)
    query_vector = query_vector.to(dtype=torch.float32)
    distances = torch.norm(reference_vectors - query_vector.unsqueeze(0), dim=1)
    topk = torch.topk(distances, k=min(k_neighbors, reference_vectors.shape[0]), largest=False)
    neighbor_distances = topk.values
    radius = neighbor_distances.mean().clamp_min(1e-8)
    return float(neighbor_distances.min() / radius)


def build_no_manifold_feature_frame(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_bank: dict[str, dict[int, torch.Tensor]],
    reference_stats: dict[str, dict[int, dict[str, float]]],
    bank_scope: str = "object",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_entries: list[dict[str, object]] = []
    for entry in cache_entries:
        object_name = str(entry["object_name"])
        bank_key = resolve_reference_scope_key(object_name, bank_scope)
        if bank_key not in reference_bank:
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": f"missing bank:{bank_key}"}
            )
            continue
        if bank_key not in reference_stats:
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": f"missing stats:{bank_key}"}
            )
            continue
        selected_layers = [int(layer_index) for layer_index in entry["selected_layers"]]
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
            missing_entries.append(
                {"sample_id": entry["sample_id"], "reason": "; ".join(reason_parts)}
            )
            continue
        raw_curve = [
            _normalized_neighbor_residual(
                entry["layer_vectors"][offset],
                reference_bank[bank_key][layer_index],
            )
            for offset, layer_index in enumerate(selected_layers)
        ]
        calibrated_curve = calibrate_drift_curve(
            raw_curve,
            selected_layers=selected_layers,
            layer_stats=reference_stats[bank_key],
            mean_key="neighbor_residual_mean",
            std_key="neighbor_residual_std",
        )
        features = build_drift_features(
            raw_curve=raw_curve,
            calibrated_curve=calibrated_curve,
        )
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "image_id": int(entry.get("image_id", -1)),
                "ground_truth_label": int(entry["label"]),
                "answer_label": -1 if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
                "label": compute_object_hallucination_label(
                    ground_truth_label=int(entry["label"]),
                    answer_label=(
                        None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
                    ),
                ),
                "subset": entry["subset"],
                "object_name": object_name,
                **features,
            }
        )
    if missing_entries:
        raise ValueError(_format_missing_reference_coverage(missing_entries))
    return pd.DataFrame(rows)


def build_linear_probe_frame(cache_entries: Sequence[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in cache_entries:
        flat = entry["layer_vectors"].reshape(-1).to(dtype=torch.float32)
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "image_id": int(entry.get("image_id", -1)),
                "ground_truth_label": int(entry["label"]),
                "answer_label": -1 if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
                "label": compute_object_hallucination_label(
                    ground_truth_label=int(entry["label"]),
                    answer_label=(
                        None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
                    ),
                ),
                "subset": entry["subset"],
                "object_name": entry["object_name"],
                **{f"hidden_{index}": float(value) for index, value in enumerate(flat.tolist())},
            }
        )
    return pd.DataFrame(rows)


def build_train_eval_splits(
    frame: pd.DataFrame,
    *,
    split_strategy: str = "row",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    if frame["label"].nunique() < 2:
        raise ValueError("Input frame must contain both hallucinated and non-hallucinated samples.")
    if split_strategy == "row":
        train_frame, eval_frame = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame["label"],
        )
        return [(0, train_frame.reset_index(drop=True), eval_frame.reset_index(drop=True))]
    if split_strategy not in GROUP_COLUMN_BY_STRATEGY:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    if num_folds < 2:
        raise ValueError("Grouped evaluation requires at least two folds.")
    group_column = GROUP_COLUMN_BY_STRATEGY[split_strategy]
    if group_column not in frame.columns:
        raise ValueError(f"Missing required grouping column: {group_column}")

    splitter = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for fold, (train_index, eval_index) in enumerate(
        splitter.split(frame, frame["label"], frame[group_column])
    ):
        train_frame = frame.iloc[train_index].reset_index(drop=True)
        eval_frame = frame.iloc[eval_index].reset_index(drop=True)
        if train_frame["label"].nunique() < 2 or eval_frame["label"].nunique() < 2:
            raise ValueError(
                f"Fold {fold} under {split_strategy} does not preserve both classes in train and eval."
            )
        splits.append((fold, train_frame, eval_frame))
    return splits


def evaluate_feature_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    split_strategy: str = "row",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
) -> tuple[dict[str, float], pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    for fold, train_frame, eval_frame in build_train_eval_splits(
        frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    ):
        detector = fit_logistic_detector(
            train_frame[list(columns)].to_numpy(),
            train_frame["label"].to_numpy(),
        )
        probabilities = detector.predict_proba(eval_frame[list(columns)].to_numpy())[:, 1]
        predictions = detector.predict(eval_frame[list(columns)].to_numpy())
        result_frames.append(
            eval_frame.assign(prediction=predictions, score=probabilities, fold=fold).reset_index(
                drop=True
            )
        )
    results = pd.concat(result_frames, ignore_index=True)
    if "sample_id" in results.columns:
        results = results.sort_values("sample_id").reset_index(drop=True)
    metrics = compute_binary_metrics(
        y_true=results["label"],
        y_pred=results["prediction"],
        y_score=results["score"],
    )
    return metrics, results
