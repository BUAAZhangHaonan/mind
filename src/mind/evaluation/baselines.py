"""Baseline and ablation helpers for MIND."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
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

FEATURE_VARIANT_NAMES = (
    "raw_curve_only",
    "raw_plus_calibrated_simple",
    "raw_plus_calibrated_full_curve",
    "raw_plus_calibrated_haar",
)

DEFAULT_FULL_VARIANT = "raw_plus_calibrated_simple"


def _hallucination_label_from_entry(entry: dict[str, object]) -> int:
    return compute_object_hallucination_label(
        ground_truth_label=int(entry["label"]),
        answer_label=None if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
    )


def _metadata_row_from_entry(entry: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": entry["sample_id"],
        "image_id": int(entry.get("image_id", -1)),
        "ground_truth_label": int(entry["label"]),
        "answer_label": -1 if entry.get("parsed_answer") is None else int(entry["parsed_answer"]),
        "label": _hallucination_label_from_entry(entry),
        "subset": entry["subset"],
        "object_name": entry["object_name"],
    }


def _sorted_drift_columns(frame: pd.DataFrame, *, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix)],
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )


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
        if bank_scope in {"object", "shuffled_object"} and object_name == "__shared__":
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
        if bank_scope in {"object", "shuffled_object"} and object_name == "__shared__":
            continue
        if bank_scope == "shared" and object_name != "__shared__":
            continue
        payload = torch.load(stats_path, weights_only=False)
        stats_map[object_name] = {
            int(layer_index): {str(key): float(value) for key, value in layer_stats.items()}
            for layer_index, layer_stats in payload.items()
        }
    return stats_map


def load_label_overrides(overrides: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(overrides, pd.DataFrame):
        override_frame = overrides.copy()
    else:
        override_frame = (
            pd.read_parquet(overrides)
            if overrides.suffix == ".parquet"
            else pd.read_json(overrides, lines=True)
        )
    override_columns = [column for column in ["sample_id", "label"] if column in override_frame.columns]
    if override_columns != ["sample_id", "label"]:
        raise ValueError("Label override file must include sample_id and label columns.")
    normalized = override_frame[["sample_id", "label"]].copy()
    normalized["sample_id"] = normalized["sample_id"].astype(str)
    normalized["label"] = normalized["label"].astype(int)
    return normalized


def _trim_cache_entry(
    entry: dict[str, object],
    *,
    keep_fields: set[str] | None,
) -> dict[str, object]:
    if keep_fields is None:
        return dict(entry)
    return {key: value for key, value in entry.items() if key in keep_fields}


def load_cache_entries(
    cache_path: Path,
    *,
    keep_fields: set[str] | None = None,
) -> list[dict[str, object]]:
    if cache_path.is_dir():
        entries: list[dict[str, object]] = []
        for shard_path in sorted(cache_path.rglob("*.pt")):
            shard_entries = torch.load(shard_path, weights_only=False)
            entries.extend(
                _trim_cache_entry(entry, keep_fields=keep_fields)
                for entry in shard_entries
            )
        return entries
    return [
        _trim_cache_entry(entry, keep_fields=keep_fields)
        for entry in torch.load(cache_path, weights_only=False)
    ]


def apply_label_overrides_to_entries(
    cache_entries: Sequence[dict[str, object]],
    overrides: Path | pd.DataFrame,
) -> list[dict[str, object]]:
    override_frame = load_label_overrides(overrides)
    override_map = {
        str(row["sample_id"]): int(row["label"])
        for row in override_frame.to_dict(orient="records")
    }
    updated_entries: list[dict[str, object]] = []
    for entry in cache_entries:
        updated = dict(entry)
        if str(entry["sample_id"]) in override_map:
            updated["label"] = int(override_map[str(entry["sample_id"])])
        updated_entries.append(updated)
    return updated_entries


def apply_label_overrides_to_frame(
    frame: pd.DataFrame,
    overrides: Path | pd.DataFrame,
) -> pd.DataFrame:
    override_frame = load_label_overrides(overrides)
    working_frame = frame.copy()
    working_frame["sample_id"] = working_frame["sample_id"].astype(str)
    merged = working_frame.drop(columns=["label"], errors="ignore").merge(
        override_frame.rename(columns={"label": "override_label"}),
        on="sample_id",
        how="left",
    )
    if "ground_truth_label" in merged.columns and "answer_label" in merged.columns:
        merged["ground_truth_label"] = (
            merged["override_label"].fillna(merged["ground_truth_label"]).astype(int)
        )
        merged["label"] = [
            compute_object_hallucination_label(
                ground_truth_label=int(ground_truth),
                answer_label=None if int(answer_label) < 0 else int(answer_label),
            )
            for ground_truth, answer_label in zip(
                merged["ground_truth_label"].tolist(),
                merged["answer_label"].tolist(),
            )
        ]
    else:
        merged["label"] = merged["override_label"].fillna(working_frame["label"]).astype(int)
    return merged.drop(columns=["override_label"])


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


def resolve_yes_no_token_ids(tokenizer) -> dict[str, list[int]]:
    token_map: dict[str, list[int]] = {"yes": [], "no": []}
    text_map = {
        "yes": ["yes", " yes", "Yes"],
        "no": ["no", " no", "No"],
    }
    for label, candidates in text_map.items():
        seen: set[int] = set()
        for candidate in candidates:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            token_id = int(token_ids[0])
            if token_id in seen:
                continue
            seen.add(token_id)
            token_map[label].append(token_id)
    if not token_map["yes"] or not token_map["no"]:
        raise ValueError("Could not resolve single-token yes/no ids from tokenizer.")
    return token_map


def _aggregate_candidate_logit(logits: torch.Tensor, token_ids: Sequence[int]) -> float:
    selected = logits[torch.as_tensor(list(token_ids), dtype=torch.long)]
    return float(torch.logsumexp(selected, dim=0))


def _aggregate_candidate_probability(logits: torch.Tensor, token_ids: Sequence[int]) -> float:
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=0)
    selected = probabilities[torch.as_tensor(list(token_ids), dtype=torch.long)]
    return float(selected.sum())


def build_output_baseline_frame(
    cache_entries: Sequence[dict[str, object]],
    *,
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in cache_entries:
        logits = entry.get("first_token_logits")
        if logits is None:
            continue
        logits = torch.as_tensor(logits, dtype=torch.float32)
        p_yes = _aggregate_candidate_probability(logits, yes_token_ids)
        p_no = _aggregate_candidate_probability(logits, no_token_ids)
        answer_label = None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
        if answer_label == 1:
            chosen_answer_confidence = p_yes
        elif answer_label == 0:
            chosen_answer_confidence = p_no
        else:
            chosen_answer_confidence = max(p_yes, p_no)
        rows.append(
            {
                **_metadata_row_from_entry(entry),
                "p_yes": p_yes,
                "yes_logit_margin": _aggregate_candidate_logit(logits, yes_token_ids)
                - _aggregate_candidate_logit(logits, no_token_ids),
                "chosen_answer_confidence": float(chosen_answer_confidence),
            }
        )
    return pd.DataFrame(rows)


def build_feature_variant_frames(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metadata_columns = [column for column in features.columns if column in METADATA_COLUMNS]
    base_frame = features.loc[:, metadata_columns].copy()
    raw_columns = _sorted_drift_columns(features, prefix="raw_drift_")
    calibrated_columns = _sorted_drift_columns(features, prefix="cal_drift_")
    wavelet_columns = sorted(
        [column for column in features.columns if column.startswith("cal_") and "energy" in column]
    )

    calibrated_array = features[calibrated_columns].to_numpy(dtype=np.float32)
    simple_stats = pd.DataFrame(
        {
            "cal_mean_drift": (
                features["cal_mean_drift"].to_numpy(dtype=np.float32)
                if "cal_mean_drift" in features.columns
                else calibrated_array.mean(axis=1)
            ),
            "cal_max_drift": (
                features["cal_max_drift"].to_numpy(dtype=np.float32)
                if "cal_max_drift" in features.columns
                else calibrated_array.max(axis=1)
            ),
            "cal_final_drift": calibrated_array[:, -1],
            "cal_drift_variance": calibrated_array.var(axis=1),
            "cal_drift_slope": np.polyfit(
                np.arange(calibrated_array.shape[1], dtype=np.float32),
                calibrated_array.T,
                deg=1,
            )[0],
        }
    )

    return {
        "raw_curve_only": pd.concat([base_frame, features[raw_columns]], axis=1),
        "raw_plus_calibrated_simple": pd.concat([base_frame, features[raw_columns], simple_stats], axis=1),
        "raw_plus_calibrated_full_curve": pd.concat(
            [base_frame, features[raw_columns], features[calibrated_columns]],
            axis=1,
        ),
        "raw_plus_calibrated_haar": pd.concat(
            [base_frame, features[raw_columns], features[wavelet_columns]],
            axis=1,
        ),
    }


def resolve_feature_variant_frame(features: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    variants = build_feature_variant_frames(features)
    if variant_name not in variants:
        raise ValueError(f"Unsupported feature variant: {variant_name}")
    return variants[variant_name]


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
    if not cache_entries:
        return pd.DataFrame()

    first_flat = torch.as_tensor(
        cache_entries[0]["layer_vectors"],
        dtype=torch.float32,
    ).reshape(-1)
    hidden_matrix = np.empty((len(cache_entries), int(first_flat.numel())), dtype=np.float32)
    metadata_rows: list[dict[str, object]] = []

    for row_index, entry in enumerate(cache_entries):
        flat = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32).reshape(-1)
        if flat.numel() != hidden_matrix.shape[1]:
            raise ValueError(
                "All linear-probe cache entries must share the same flattened hidden-state size."
            )
        hidden_matrix[row_index, :] = flat.cpu().numpy()
        metadata_rows.append(
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
            }
        )

    metadata_frame = pd.DataFrame(metadata_rows)
    hidden_frame = pd.DataFrame(
        hidden_matrix,
        columns=[f"hidden_{index}" for index in range(hidden_matrix.shape[1])],
        dtype=np.float32,
    )
    return pd.concat([metadata_frame, hidden_frame], axis=1)


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


def evaluate_feature_frame_across_random_states(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    split_strategy: str = "row",
    test_size: float = 0.3,
    random_states: Sequence[int],
    num_folds: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for random_state in random_states:
        metrics, _ = evaluate_feature_frame(
            frame,
            columns=columns,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=int(random_state),
            num_folds=num_folds,
        )
        rows.append({"random_state": int(random_state), **metrics})
    return pd.DataFrame(rows)


def resolve_highest_valid_num_folds(
    frames: Sequence[pd.DataFrame],
    *,
    split_strategy: str,
    candidate_folds: Sequence[int] = (5, 4, 3, 2),
    random_state: int = 13,
) -> int:
    for num_folds in candidate_folds:
        try:
            for frame in frames:
                build_train_eval_splits(
                    frame,
                    split_strategy=split_strategy,
                    random_state=random_state,
                    num_folds=int(num_folds),
                )
        except ValueError:
            continue
        return int(num_folds)
    raise ValueError(f"No valid fold count found for {split_strategy}.")


def compute_bootstrap_confidence_intervals(
    results: pd.DataFrame,
    *,
    group_column: str,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 13,
) -> dict[str, dict[str, float]]:
    if group_column not in results.columns:
        raise ValueError(f"Missing grouping column: {group_column}")
    groups = results[group_column].drop_duplicates().tolist()
    if not groups:
        raise ValueError("Bootstrap requires at least one group.")
    y_true = results["label"].to_numpy(dtype=np.int64)
    y_pred = results["prediction"].to_numpy(dtype=np.int64)
    y_score = results["score"].to_numpy(dtype=np.float64)
    grouped_indices = {
        group_value: np.asarray(index_frame, dtype=np.int64)
        for group_value, index_frame in results.groupby(group_column).indices.items()
    }
    rng = np.random.default_rng(random_state)
    alpha = 1.0 - ci_level
    collected: list[dict[str, float]] = []
    attempts = 0
    max_attempts = max(10 * n_resamples, 100)

    while len(collected) < n_resamples and attempts < max_attempts:
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([grouped_indices[group_value] for group_value in sampled_groups])
        attempts += 1
        sampled_true = y_true[sampled_indices]
        if np.unique(sampled_true).size < 2:
            continue
        collected.append(
            compute_binary_metrics(
                y_true=sampled_true,
                y_pred=y_pred[sampled_indices],
                y_score=y_score[sampled_indices],
            )
        )

    if not collected:
        raise ValueError("Bootstrap did not produce any valid resamples.")

    point_metrics = compute_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )
    intervals: dict[str, dict[str, float]] = {}
    for metric_name in point_metrics:
        values = np.asarray([metrics[metric_name] for metrics in collected], dtype=np.float64)
        intervals[metric_name] = {
            "point": float(point_metrics[metric_name]),
            "lower": float(np.quantile(values, alpha / 2.0)),
            "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "n_resamples": float(len(values)),
        }
    return intervals
