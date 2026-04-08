"""Faithful HALP-style pre-generation probing on grouped splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mind.evaluation.baselines import (
    build_train_eval_splits,
    compute_bootstrap_confidence_intervals,
    prepare_object_heldout_frame,
    resolve_highest_valid_num_folds,
)
from mind.evaluation.metrics import compute_binary_metrics, compute_object_hallucination_label


HALP_METADATA_COLUMNS = {
    "sample_id",
    "image_id",
    "ground_truth_label",
    "answer_label",
    "label",
    "subset",
    "object_name",
    "fold",
    "selected_probe",
}


@dataclass(frozen=True)
class HALPProbeConfig:
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    random_state: int = 13


class HALPProbe(nn.Module):
    def __init__(self, input_dim: int, config: HALPProbeConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(config.dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def resolve_halp_layer_indices(total_layers: int) -> list[int]:
    if total_layers < 1:
        raise ValueError("total_layers must be positive")
    return list(range(total_layers))


def _metadata_row_from_readout_entry(entry: dict[str, object]) -> dict[str, object]:
    answer_label = None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
    ground_truth_label = int(entry["label"])
    return {
        "sample_id": str(entry["sample_id"]),
        "image_id": int(entry.get("image_id", -1)),
        "ground_truth_label": ground_truth_label,
        "answer_label": -1 if answer_label is None else answer_label,
        "label": compute_object_hallucination_label(
            ground_truth_label=ground_truth_label,
            answer_label=answer_label,
        ),
        "subset": str(entry.get("subset", "")),
        "object_name": str(entry.get("object_name", "")),
    }


def _feature_row(metadata: dict[str, object], vector: torch.Tensor) -> dict[str, object]:
    row = dict(metadata)
    row.update({f"feature_{index}": float(value) for index, value in enumerate(vector.tolist())})
    return row


def _flatten_vision_features(vision_features: torch.Tensor) -> torch.Tensor:
    if vision_features.ndim == 1:
        return vision_features
    if vision_features.ndim == 2:
        return vision_features.mean(dim=0)
    raise ValueError(f"Unsupported vision feature rank: {vision_features.ndim}")


def _resolve_total_layers(entry: dict[str, object]) -> int:
    if "query_hidden_states" in entry:
        return int(torch.as_tensor(entry["query_hidden_states"]).shape[0])
    return int(torch.as_tensor(entry["full_hidden_states"]).shape[0])


def _resolve_query_hidden_states(entry: dict[str, object]) -> torch.Tensor:
    if "query_hidden_states" in entry:
        return torch.as_tensor(entry["query_hidden_states"], dtype=torch.float32)
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"], dtype=torch.float32)
    query_token_index = int(entry["query_token_index"])
    return full_hidden_states[:, query_token_index, :]


def _resolve_vision_token_hidden_states(entry: dict[str, object]) -> torch.Tensor:
    if "vision_token_hidden_states" in entry:
        return torch.as_tensor(entry["vision_token_hidden_states"], dtype=torch.float32)
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"], dtype=torch.float32)
    vision_token_span = entry.get("vision_token_span")
    if vision_token_span is None:
        raise ValueError(f"Missing vision token span for sample {entry['sample_id']}")
    vision_token_index = int(vision_token_span[-1])
    return full_hidden_states[:, vision_token_index, :]


def resolve_halp_probe_names(
    readout_entries: Sequence[dict[str, object]],
    *,
    layer_indices: Sequence[int] | None = None,
) -> list[str]:
    if not readout_entries:
        raise ValueError("readout_entries must not be empty")
    selected_layers = list(layer_indices or resolve_halp_layer_indices(_resolve_total_layers(readout_entries[0])))
    probe_names = ["vision_only"]
    for layer_index in selected_layers:
        probe_names.append(f"vision_token_layer_{layer_index}")
        probe_names.append(f"query_token_layer_{layer_index}")
    return probe_names


def _parse_halp_probe_name(probe_name: str) -> tuple[str, int | None]:
    if probe_name == "vision_only":
        return "vision_only", None
    if probe_name.startswith("vision_token_layer_"):
        return "vision_token", int(probe_name.rsplit("_", 1)[-1])
    if probe_name.startswith("query_token_layer_"):
        return "query_token", int(probe_name.rsplit("_", 1)[-1])
    raise ValueError(f"Unknown HALP probe name: {probe_name}")


def build_halp_probe_frame(
    readout_entries: Sequence[dict[str, object]],
    probe_name: str,
    *,
    allowed_sample_ids: set[str] | None = None,
) -> pd.DataFrame:
    if not readout_entries:
        raise ValueError("readout_entries must not be empty")

    probe_kind, layer_index = _parse_halp_probe_name(probe_name)
    rows: list[dict[str, object]] = []
    for entry in readout_entries:
        sample_id = str(entry["sample_id"])
        if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
            continue
        metadata = _metadata_row_from_readout_entry(entry)
        if probe_kind == "vision_only":
            vision_features = entry.get("vision_features")
            if vision_features is None:
                raise ValueError(f"Missing vision features for sample {entry['sample_id']}")
            vector = _flatten_vision_features(torch.as_tensor(vision_features, dtype=torch.float32))
        elif probe_kind == "vision_token":
            vision_token_hidden_states = _resolve_vision_token_hidden_states(entry)
            vector = vision_token_hidden_states[layer_index, :]
        else:
            query_hidden_states = _resolve_query_hidden_states(entry)
            vector = query_hidden_states[layer_index, :]
        rows.append(_feature_row(metadata, vector))
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def build_halp_probe_frames(
    readout_entries: Sequence[dict[str, object]],
    *,
    layer_indices: Sequence[int] | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        probe_name: build_halp_probe_frame(readout_entries, probe_name)
        for probe_name in resolve_halp_probe_names(readout_entries, layer_indices=layer_indices)
    }


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in HALP_METADATA_COLUMNS]


def _build_sample_id_splits(
    frame: pd.DataFrame,
    *,
    split_strategy: str,
    test_size: float,
    random_state: int,
    num_folds: int,
) -> list[tuple[int, list[str], list[str]]]:
    splits = build_train_eval_splits(
        frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    )
    return [
        (
            fold,
            [str(sample_id) for sample_id in train_frame["sample_id"].tolist()],
            [str(sample_id) for sample_id in eval_frame["sample_id"].tolist()],
        )
        for fold, train_frame, eval_frame in splits
    ]


def _subset_frame_by_sample_ids(frame: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    indexed = frame.set_index("sample_id")
    subset = indexed.loc[list(sample_ids)].reset_index()
    return subset


def _fit_probe(
    train_frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    config: HALPProbeConfig,
) -> HALPProbe:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    features = torch.tensor(train_frame[list(columns)].to_numpy(dtype=np.float32))
    labels = torch.tensor(train_frame["label"].to_numpy(dtype=np.float32))
    dataset = TensorDataset(features, labels)
    batch_size = max(2, min(config.batch_size, len(dataset)))
    drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    probe = HALPProbe(input_dim=len(columns), config=config)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(config.epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
    return probe


def _predict_probe(
    probe: HALPProbe,
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    probe.eval()
    with torch.inference_mode():
        features = torch.tensor(frame[list(columns)].to_numpy(dtype=np.float32))
        scores = torch.sigmoid(probe(features)).cpu().numpy()
    predictions = (scores >= 0.5).astype(np.int64)
    return predictions, scores


def _evaluate_probe_on_splits(
    frame: pd.DataFrame,
    *,
    splits: Sequence[tuple[int, list[str], list[str]]],
    config: HALPProbeConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    columns = _feature_columns(frame)
    result_frames: list[pd.DataFrame] = []
    for fold, train_ids, eval_ids in splits:
        train_frame = _subset_frame_by_sample_ids(frame, train_ids)
        eval_frame = _subset_frame_by_sample_ids(frame, eval_ids)
        probe = _fit_probe(train_frame, columns=columns, config=config)
        predictions, scores = _predict_probe(probe, eval_frame, columns=columns)
        result_frames.append(
            eval_frame.assign(
                prediction=predictions,
                score=scores,
                fold=fold,
            )
        )
    results = pd.concat(result_frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    metrics = compute_binary_metrics(
        y_true=results["label"],
        y_pred=results["prediction"],
        y_score=results["score"],
    )
    return metrics, results


def _select_best_probe(
    candidate_frames: dict[str, pd.DataFrame],
    *,
    base_train_frame: pd.DataFrame,
    split_strategy: str,
    test_size: float,
    random_state: int,
    inner_candidate_folds: Sequence[int],
    probe_config: HALPProbeConfig,
) -> tuple[str, int, pd.DataFrame]:
    if split_strategy == "row":
        inner_num_folds = 1
    else:
        inner_num_folds = resolve_highest_valid_num_folds(
            [base_train_frame],
            split_strategy=split_strategy,
            candidate_folds=inner_candidate_folds,
            random_state=random_state,
        )
    inner_splits = _build_sample_id_splits(
        base_train_frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=inner_num_folds,
    )

    rows: list[dict[str, object]] = []
    for probe_name, probe_frame in candidate_frames.items():
        inner_frame = _subset_frame_by_sample_ids(probe_frame, base_train_frame["sample_id"].tolist())
        metrics, _ = _evaluate_probe_on_splits(inner_frame, splits=inner_splits, config=probe_config)
        rows.append({"probe_name": probe_name, "inner_num_folds": inner_num_folds, **metrics})
    selection_frame = pd.DataFrame(rows).sort_values(
        by=["roc_auc", "pr_auc", "probe_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return str(selection_frame.iloc[0]["probe_name"]), inner_num_folds, selection_frame


def _select_best_probe_from_readout_entries(
    readout_entries: Sequence[dict[str, object]],
    *,
    probe_names: Sequence[str],
    base_train_frame: pd.DataFrame,
    split_strategy: str,
    test_size: float,
    random_state: int,
    inner_candidate_folds: Sequence[int],
    probe_config: HALPProbeConfig,
) -> tuple[str, int, pd.DataFrame]:
    if split_strategy == "row":
        inner_num_folds = 1
    else:
        inner_num_folds = resolve_highest_valid_num_folds(
            [base_train_frame],
            split_strategy=split_strategy,
            candidate_folds=inner_candidate_folds,
            random_state=random_state,
        )
    inner_splits = _build_sample_id_splits(
        base_train_frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=inner_num_folds,
    )

    train_sample_ids = set(base_train_frame["sample_id"].astype(str).tolist())
    rows: list[dict[str, object]] = []
    for probe_name in probe_names:
        inner_frame = build_halp_probe_frame(
            readout_entries,
            probe_name,
            allowed_sample_ids=train_sample_ids,
        )
        metrics, _ = _evaluate_probe_on_splits(inner_frame, splits=inner_splits, config=probe_config)
        rows.append({"probe_name": probe_name, "inner_num_folds": inner_num_folds, **metrics})
    selection_frame = pd.DataFrame(rows).sort_values(
        by=["roc_auc", "pr_auc", "probe_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return str(selection_frame.iloc[0]["probe_name"]), inner_num_folds, selection_frame


def evaluate_halp_nested(
    candidate_frames: dict[str, pd.DataFrame],
    *,
    split_strategy: str = "image_grouped",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
    inner_candidate_folds: Sequence[int] = (3, 2),
    probe_config: HALPProbeConfig = HALPProbeConfig(),
    supported_object_names: Sequence[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if not candidate_frames:
        raise ValueError("candidate_frames must not be empty")
    base_frame = next(iter(candidate_frames.values())).sort_values("sample_id").reset_index(drop=True)
    if split_strategy == "object_heldout":
        base_frame, _ = prepare_object_heldout_frame(
            base_frame,
            supported_object_names=(
                supported_object_names
                if supported_object_names is not None
                else set(base_frame["object_name"].astype(str).tolist())
            ),
            requested_num_folds=num_folds,
            context="HALP object_heldout",
        )
        allowed_sample_ids = set(base_frame["sample_id"].astype(str).tolist())
        candidate_frames = {
            probe_name: probe_frame[
                probe_frame["sample_id"].astype(str).isin(allowed_sample_ids)
            ].sort_values("sample_id").reset_index(drop=True)
            for probe_name, probe_frame in candidate_frames.items()
        }
    outer_splits = _build_sample_id_splits(
        base_frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    )

    result_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for fold, train_ids, eval_ids in outer_splits:
        outer_train_frame = _subset_frame_by_sample_ids(base_frame, train_ids)
        selected_probe, inner_num_folds, selection_frame = _select_best_probe(
            candidate_frames,
            base_train_frame=outer_train_frame,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=random_state,
            inner_candidate_folds=inner_candidate_folds,
            probe_config=probe_config,
        )
        selection_rows.extend(
            selection_frame.assign(outer_fold=fold, selected_probe=selected_probe).to_dict(orient="records")
        )

        selected_frame = candidate_frames[selected_probe]
        train_frame = _subset_frame_by_sample_ids(selected_frame, train_ids)
        eval_frame = _subset_frame_by_sample_ids(selected_frame, eval_ids)
        probe = _fit_probe(train_frame, columns=_feature_columns(selected_frame), config=probe_config)
        predictions, scores = _predict_probe(probe, eval_frame, columns=_feature_columns(selected_frame))
        result_frames.append(
            eval_frame.assign(
                prediction=predictions,
                score=scores,
                fold=fold,
                selected_probe=selected_probe,
                inner_num_folds=inner_num_folds,
            )
        )

    results = pd.concat(result_frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    metrics = compute_binary_metrics(
        y_true=results["label"],
        y_pred=results["prediction"],
        y_score=results["score"],
    )
    return metrics, results, pd.DataFrame(selection_rows)


def evaluate_halp_nested_from_readout_entries(
    readout_entries: Sequence[dict[str, object]],
    *,
    split_strategy: str = "image_grouped",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
    inner_candidate_folds: Sequence[int] = (3, 2),
    probe_config: HALPProbeConfig = HALPProbeConfig(),
    supported_object_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if not readout_entries:
        raise ValueError("readout_entries must not be empty")

    probe_names = resolve_halp_probe_names(readout_entries, layer_indices=layer_indices)
    base_frame = build_halp_probe_frame(readout_entries, "vision_only")
    if split_strategy == "object_heldout":
        base_frame, _ = prepare_object_heldout_frame(
            base_frame,
            supported_object_names=(
                supported_object_names
                if supported_object_names is not None
                else set(base_frame["object_name"].astype(str).tolist())
            ),
            requested_num_folds=num_folds,
            context="HALP object_heldout",
        )
    outer_splits = _build_sample_id_splits(
        base_frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    )

    result_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for fold, train_ids, eval_ids in outer_splits:
        outer_train_frame = _subset_frame_by_sample_ids(base_frame, train_ids)
        selected_probe, inner_num_folds, selection_frame = _select_best_probe_from_readout_entries(
            readout_entries,
            probe_names=probe_names,
            base_train_frame=outer_train_frame,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=random_state,
            inner_candidate_folds=inner_candidate_folds,
            probe_config=probe_config,
        )
        selection_rows.extend(
            selection_frame.assign(outer_fold=fold, selected_probe=selected_probe).to_dict(orient="records")
        )

        train_frame = build_halp_probe_frame(
            readout_entries,
            selected_probe,
            allowed_sample_ids=set(train_ids),
        )
        eval_frame = build_halp_probe_frame(
            readout_entries,
            selected_probe,
            allowed_sample_ids=set(eval_ids),
        )
        probe = _fit_probe(train_frame, columns=_feature_columns(train_frame), config=probe_config)
        predictions, scores = _predict_probe(probe, eval_frame, columns=_feature_columns(train_frame))
        result_frames.append(
            eval_frame.assign(
                prediction=predictions,
                score=scores,
                fold=fold,
                selected_probe=selected_probe,
                inner_num_folds=inner_num_folds,
            )
        )

    results = pd.concat(result_frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    metrics = compute_binary_metrics(
        y_true=results["label"],
        y_pred=results["prediction"],
        y_score=results["score"],
    )
    return metrics, results, pd.DataFrame(selection_rows)


def summarize_halp_results(
    results: pd.DataFrame,
    *,
    split_strategy: str,
    bootstrap_resamples: int = 1000,
    random_state: int = 13,
) -> dict[str, object]:
    group_column = "sample_id"
    if split_strategy == "image_grouped":
        group_column = "image_id"
    elif split_strategy == "object_heldout":
        group_column = "object_name"
    return {
        **compute_binary_metrics(
            y_true=results["label"],
            y_pred=results["prediction"],
            y_score=results["score"],
        ),
        "confidence_intervals": compute_bootstrap_confidence_intervals(
            results,
            group_column=group_column,
            n_resamples=bootstrap_resamples,
            random_state=random_state,
        ),
    }
