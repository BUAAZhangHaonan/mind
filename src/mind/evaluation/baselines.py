"""Baseline and ablation helpers for MIND."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mind.detectors import fit_logistic_detector
from mind.drift import standardize_drift_curve
from mind.wavelets import extract_wavelet_features

from .metrics import compute_binary_metrics, compute_object_hallucination_label


METADATA_COLUMNS = {
    "sample_id",
    "label",
    "subset",
    "object_name",
    "ground_truth_label",
    "answer_label",
}


def load_reference_bank(reference_root: Path, model_name: str) -> dict[str, dict[int, torch.Tensor]]:
    bank: dict[str, dict[int, torch.Tensor]] = {}
    model_root = reference_root / model_name
    for layer_path in model_root.glob("*/*.pt"):
        object_name = layer_path.parent.name
        layer_index = int(layer_path.stem.split("-")[-1])
        bank.setdefault(object_name, {})[layer_index] = torch.load(layer_path, weights_only=False)
    return bank


def load_cache_entries(cache_path: Path) -> list[dict[str, object]]:
    if cache_path.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            entries.extend(torch.load(shard_path, weights_only=False))
        return entries
    return list(torch.load(cache_path, weights_only=False))


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def drift_only_columns(frame: pd.DataFrame) -> list[str]:
    keep = {"max_drift", "mean_drift", "peak_layer_index"}
    return [
        column
        for column in frame.columns
        if column.startswith("drift_") or column in keep
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
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in cache_entries:
        object_name = str(entry["object_name"])
        if object_name not in reference_bank:
            continue
        selected_layers = [int(layer_index) for layer_index in entry["selected_layers"]]
        if any(layer_index not in reference_bank[object_name] for layer_index in selected_layers):
            continue
        drift_curve = [
            _normalized_neighbor_residual(
                entry["layer_vectors"][offset],
                reference_bank[object_name][layer_index],
            )
            for offset, layer_index in enumerate(selected_layers)
        ]
        features = extract_wavelet_features(standardize_drift_curve(drift_curve))
        rows.append(
            {
                "sample_id": entry["sample_id"],
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
    return pd.DataFrame(rows)


def build_linear_probe_frame(cache_entries: Sequence[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in cache_entries:
        flat = entry["layer_vectors"].reshape(-1).to(dtype=torch.float32)
        rows.append(
            {
                "sample_id": entry["sample_id"],
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


def evaluate_feature_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    test_size: float = 0.3,
    random_state: int = 13,
) -> tuple[dict[str, float], pd.DataFrame]:
    train_frame, eval_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )
    detector = fit_logistic_detector(
        train_frame[list(columns)].to_numpy(),
        train_frame["label"].to_numpy(),
    )
    probabilities = detector.predict_proba(eval_frame[list(columns)].to_numpy())[:, 1]
    predictions = detector.predict(eval_frame[list(columns)].to_numpy())
    results = eval_frame.assign(prediction=predictions, score=probabilities).reset_index(drop=True)
    metrics = compute_binary_metrics(
        y_true=results["label"],
        y_pred=results["prediction"],
        y_score=results["score"],
    )
    return metrics, results
