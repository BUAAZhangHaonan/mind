"""GLSim adaptation for queried-object yes/no hallucination benchmarks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mind.evaluation.baselines import (
    build_train_eval_splits,
    compute_bootstrap_confidence_intervals,
    prepare_object_heldout_frame,
    resolve_highest_valid_num_folds,
)
from mind.evaluation.metrics import compute_binary_metrics, compute_object_hallucination_label


GLSIM_METADATA_COLUMNS = {
    "sample_id",
    "image_id",
    "ground_truth_label",
    "answer_label",
    "label",
    "subset",
    "object_name",
    "fold",
    "selected_config",
}


def resolve_glsim_layer_indices(total_layers: int) -> list[int]:
    if total_layers < 1:
        raise ValueError("total_layers must be positive")
    return [
        0,
        total_layers // 4,
        total_layers // 2,
        (3 * total_layers) // 4,
        total_layers - 1,
    ]


def find_subsequence_start(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
    if not subsequence:
        return None
    width = len(subsequence)
    for index in range(0, len(sequence) - width + 1):
        if list(sequence[index : index + width]) == list(subsequence):
            return index
    return None


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


def resolve_readout_total_layers(entry: dict[str, object]) -> int:
    if "query_hidden_states" in entry:
        return int(torch.as_tensor(entry["query_hidden_states"]).shape[0])
    return int(torch.as_tensor(entry["full_hidden_states"]).shape[0])


def _resolve_query_hidden_states(entry: dict[str, object]) -> torch.Tensor:
    if "query_hidden_states" in entry:
        return torch.as_tensor(entry["query_hidden_states"], dtype=torch.float32)
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"], dtype=torch.float32)
    query_token_index = int(entry["query_token_index"])
    return full_hidden_states[:, query_token_index, :]


def _resolve_object_hidden_states(entry: dict[str, object]) -> torch.Tensor:
    if "object_hidden_states" in entry:
        return torch.as_tensor(entry["object_hidden_states"], dtype=torch.float32)
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"], dtype=torch.float32)
    object_token_index = entry.get("object_token_index")
    if object_token_index is None:
        raise ValueError(f"Missing object token index for sample {entry['sample_id']}")
    return full_hidden_states[:, int(object_token_index), :]


def _resolve_object_hidden_states_with_context(
    entry: dict[str, object],
    *,
    context: dict[str, int],
) -> torch.Tensor:
    if "object_hidden_states" in entry:
        return torch.as_tensor(entry["object_hidden_states"], dtype=torch.float32)
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"], dtype=torch.float32)
    object_token_index = entry.get("object_token_index", context["object_token_index"])
    return full_hidden_states[:, int(object_token_index), :]


def _resolve_glsim_visual_slices(
    entry: dict[str, object],
    *,
    layer_indices: Sequence[int],
) -> dict[int, torch.Tensor]:
    if "glsim_vision_hidden_states" in entry:
        cached_layers = [int(value) for value in entry.get("glsim_layer_indices", [])]
        layer_to_index = {layer: index for index, layer in enumerate(cached_layers)}
        missing_layers = [layer for layer in layer_indices if layer not in layer_to_index]
        if missing_layers:
            raise ValueError(
                f"Compact GLSim cache for sample {entry['sample_id']} is missing requested layers: {missing_layers}"
            )
        cached = torch.as_tensor(entry["glsim_vision_hidden_states"], dtype=torch.float32)
        return {layer: cached[layer_to_index[layer]] for layer in layer_indices}
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"], dtype=torch.float32)
    vision_span = entry.get("vision_token_span")
    if vision_span is None:
        raise ValueError(f"Missing vision token span for sample {entry['sample_id']}")
    vision_start, vision_stop = int(vision_span[0]), int(vision_span[1])
    return {
        layer_index: full_hidden_states[layer_index, vision_start : vision_stop + 1, :]
        for layer_index in layer_indices
    }


def _resolve_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def build_object_token_contexts(
    readout_entries: Sequence[dict[str, object]],
    *,
    wrapper,
    processor,
) -> dict[str, dict[str, int]]:
    tokenizer = _resolve_tokenizer(processor)
    contexts: dict[str, dict[str, int]] = {}
    for entry in readout_entries:
        if entry.get("object_token_index") is not None and entry.get("object_token_id") is not None:
            contexts[str(entry["sample_id"])] = {
                "object_token_id": int(entry["object_token_id"]),
                "object_token_index": int(entry["object_token_index"]),
            }
            continue
        model_inputs = wrapper.prepare_inputs(
            processor,
            question=str(entry["question"]),
            image_path=str(entry["image_path"]),
            device="cpu",
        )
        input_ids = model_inputs["input_ids"][0].tolist()
        object_token_ids = tokenizer.encode(str(entry["object_name"]), add_special_tokens=False)
        if not object_token_ids:
            raise ValueError(f"Could not tokenize object name: {entry['object_name']}")
        start_index = find_subsequence_start(input_ids, object_token_ids)
        if start_index is None:
            raise ValueError(f"Could not locate object token span for sample {entry['sample_id']}")
        contexts[str(entry["sample_id"])] = {
            "object_token_id": int(object_token_ids[0]),
            "object_token_index": int(start_index),
        }
    return contexts


def _resolve_output_embedding_device(output_embeddings) -> torch.device:
    try:
        return next(output_embeddings.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _resolve_output_embedding_dtype(output_embeddings) -> torch.dtype:
    try:
        return next(output_embeddings.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return torch.float32


def _object_probabilities(
    output_embeddings,
    hidden_states: torch.Tensor,
    *,
    object_token_id: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    device = _resolve_output_embedding_device(output_embeddings)
    dtype = _resolve_output_embedding_dtype(output_embeddings)
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, hidden_states.shape[0], chunk_size):
            chunk = hidden_states[start : start + chunk_size].to(device=device, dtype=dtype)
            logits = output_embeddings(chunk).to(dtype=torch.float32)
            probabilities.append(torch.softmax(logits, dim=-1)[:, object_token_id].cpu())
    return torch.cat(probabilities, dim=0)


def build_glsim_score_frame(
    readout_entries: Sequence[dict[str, object]],
    *,
    contexts: dict[str, dict[str, int]],
    output_embeddings,
    layer_indices: Sequence[int],
    k_values: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sorted_ks = sorted({int(value) for value in k_values})
    max_k = max(sorted_ks)

    for entry in readout_entries:
        sample_id = str(entry["sample_id"])
        context = contexts[sample_id]
        query_hidden_states = _resolve_query_hidden_states(entry)
        object_hidden_states = _resolve_object_hidden_states_with_context(
            entry,
            context=context,
        )
        visual_slices = _resolve_glsim_visual_slices(entry, layer_indices=layer_indices)
        row = _metadata_row_from_readout_entry(entry)
        for image_layer in layer_indices:
            visual_hidden = visual_slices[image_layer]
            probabilities = _object_probabilities(
                output_embeddings,
                visual_hidden,
                object_token_id=context["object_token_id"],
            )
            top_indices = torch.argsort(probabilities, descending=True)[: min(max_k, len(probabilities))]
            query_hidden = query_hidden_states[image_layer, :]
            query_hidden_norm = F.normalize(query_hidden.unsqueeze(0), dim=-1)[0]
            visual_hidden_norm = F.normalize(visual_hidden, dim=-1)
            for text_layer in layer_indices:
                object_hidden = object_hidden_states[text_layer, :]
                object_hidden_norm = F.normalize(object_hidden.unsqueeze(0), dim=-1)[0]
                global_score = torch.dot(query_hidden_norm, object_hidden_norm).item()
                row[f"global_i{image_layer}_t{text_layer}"] = float(global_score)

                local_cosines = torch.matmul(visual_hidden_norm, object_hidden_norm)
                for k in sorted_ks:
                    selected = top_indices[: min(k, len(top_indices))]
                    row[f"local_i{image_layer}_t{text_layer}_k{k}"] = float(local_cosines[selected].mean().item())
        rows.append(row)
    return pd.DataFrame(rows)


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


def _subset_score_frame(frame: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    return frame.set_index("sample_id").loc[list(sample_ids)].reset_index()


def _config_name(image_layer: int, text_layer: int, k: int, w: float) -> str:
    return f"i{image_layer}_t{text_layer}_k{k}_w{w:.2f}"


def _score_columns(image_layer: int, text_layer: int, k: int) -> tuple[str, str]:
    return (
        f"global_i{image_layer}_t{text_layer}",
        f"local_i{image_layer}_t{text_layer}_k{k}",
    )


def _evaluate_config_on_splits(
    frame: pd.DataFrame,
    *,
    splits: Sequence[tuple[int, list[str], list[str]]],
    image_layer: int,
    text_layer: int,
    k: int,
    w: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    global_column, local_column = _score_columns(image_layer, text_layer, k)
    result_frames: list[pd.DataFrame] = []
    for fold, _train_ids, eval_ids in splits:
        eval_frame = _subset_score_frame(frame, eval_ids)
        scores = w * eval_frame[global_column] + (1.0 - w) * eval_frame[local_column]
        predictions = (scores >= 0.0).astype(int)
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


def evaluate_glsim_nested(
    score_frame: pd.DataFrame,
    *,
    image_layers: Sequence[int],
    text_layers: Sequence[int],
    k_values: Sequence[int],
    w_values: Sequence[float],
    split_strategy: str = "image_grouped",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
    inner_candidate_folds: Sequence[int] = (3, 2),
    supported_object_names: Sequence[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    base_frame = score_frame.sort_values("sample_id").reset_index(drop=True)
    if split_strategy == "object_heldout":
        base_frame, _ = prepare_object_heldout_frame(
            base_frame,
            supported_object_names=(
                supported_object_names
                if supported_object_names is not None
                else set(base_frame["object_name"].astype(str).tolist())
            ),
            requested_num_folds=num_folds,
            context="GLSim object_heldout",
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
        outer_train_frame = _subset_score_frame(base_frame, train_ids)
        if split_strategy == "row":
            inner_num_folds = 1
        else:
            inner_num_folds = resolve_highest_valid_num_folds(
                [outer_train_frame],
                split_strategy=split_strategy,
                candidate_folds=inner_candidate_folds,
                random_state=random_state,
            )
        inner_splits = _build_sample_id_splits(
            outer_train_frame,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=random_state,
            num_folds=inner_num_folds,
        )

        rows: list[dict[str, object]] = []
        for image_layer in image_layers:
            for text_layer in text_layers:
                for k in k_values:
                    for w in w_values:
                        global_column, local_column = _score_columns(int(image_layer), int(text_layer), int(k))
                        if global_column not in outer_train_frame.columns or local_column not in outer_train_frame.columns:
                            continue
                        metrics, _ = _evaluate_config_on_splits(
                            outer_train_frame,
                            splits=inner_splits,
                            image_layer=int(image_layer),
                            text_layer=int(text_layer),
                            k=int(k),
                            w=float(w),
                        )
                        rows.append(
                            {
                                "config_name": _config_name(int(image_layer), int(text_layer), int(k), float(w)),
                                "image_layer": int(image_layer),
                                "text_layer": int(text_layer),
                                "k": int(k),
                                "w": float(w),
                                "inner_num_folds": inner_num_folds,
                                **metrics,
                            }
                        )
        if not rows:
            raise ValueError("No valid GLSim configurations were available for evaluation.")
        selection_frame = pd.DataFrame(rows).sort_values(
            by=["roc_auc", "pr_auc", "config_name"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        best = selection_frame.iloc[0]
        selection_rows.extend(
            selection_frame.assign(
                outer_fold=fold,
                selected_config=str(best["config_name"]),
            ).to_dict(orient="records")
        )
        _, outer_results = _evaluate_config_on_splits(
            base_frame,
            splits=[(fold, train_ids, eval_ids)],
            image_layer=int(best["image_layer"]),
            text_layer=int(best["text_layer"]),
            k=int(best["k"]),
            w=float(best["w"]),
        )
        result_frames.append(
            outer_results.assign(
                selected_config=str(best["config_name"]),
                inner_num_folds=int(best["inner_num_folds"]),
            )
        )

    results = pd.concat(result_frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    metrics = compute_binary_metrics(
        y_true=results["label"],
        y_pred=results["prediction"],
        y_score=results["score"],
    )
    return metrics, results, pd.DataFrame(selection_rows)


def summarize_glsim_results(
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
