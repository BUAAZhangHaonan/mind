#!/usr/bin/env python3
"""Train a CUDA logistic detector for MIND feature parquet files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import pandas as pd
import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

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


RAW_DRIFT_COLUMN_PATTERN = re.compile(r"^raw_drift_\d+$")
CALIBRATED_SUMMARY_COLUMNS = (
    "cal_mean_drift",
    "cal_max_drift",
    "cal_final_drift",
    "cal_drift_slope",
    "cal_drift_variance",
)


@dataclass
class LogisticState:
    weights: torch.Tensor
    bias: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor


def _build_train_eval_splits(
    frame: pd.DataFrame,
    *,
    split_strategy: str = "row",
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    from mind.evaluation.baselines import build_train_eval_splits

    return build_train_eval_splits(
        frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    )


def _device_from_arg(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but CUDA is not available.")
    return device


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor.")


def _sorted_curve_columns(frame: pd.DataFrame, *, prefix: str | None = None) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if RAW_DRIFT_COLUMN_PATTERN.match(str(column)) and (prefix is None or str(column).startswith(prefix))
    ]
    return sorted(columns, key=lambda column: int(str(column).rsplit("_", 1)[1]))


def resolve_columns(frame: pd.DataFrame, columns: str, variant_name: str) -> list[str]:
    requested = columns.strip()
    if requested == "auto":
        requested = variant_name
    if requested == "drift_only":
        resolved = [
            *_sorted_curve_columns(frame, prefix="raw_drift_"),
            *[
                column
                for column in ("raw_max_drift", "raw_mean_drift", "raw_peak_layer_index")
                if column in frame.columns
            ],
        ]
    elif requested == "full_curve":
        resolved = [
            *_sorted_curve_columns(frame, prefix="raw_drift_"),
            *[column for column in CALIBRATED_SUMMARY_COLUMNS if column in frame.columns],
        ]
    elif requested == "all_features":
        resolved = [column for column in frame.columns if column not in METADATA_COLUMNS]
    else:
        resolved = [column.strip() for column in requested.split(",") if column.strip()]
    if not resolved:
        raise ValueError(f"No feature columns resolved for columns={columns!r} variant_name={variant_name!r}.")
    missing = [column for column in resolved if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return resolved


def _tensor_frame(frame: pd.DataFrame, columns: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(frame[list(columns)].to_numpy("float32"), dtype=torch.float32, device=device)
    y = torch.as_tensor(frame["label"].to_numpy("float32"), dtype=torch.float32, device=device)
    return x, y


def fit_logistic_regression_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    require_cuda: bool = False,
) -> LogisticState:
    if require_cuda:
        _require_cuda_tensor("x", x)
        _require_cuda_tensor("y", y)
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_standard = (x - mean) / scale
    weights = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype, requires_grad=True)
    bias = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x_standard @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.5 * float(l2) * torch.dot(weights, weights) / max(1, x.shape[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticState(
        weights=weights.detach(),
        bias=bias.detach(),
        mean=mean.detach(),
        scale=scale.detach(),
    )


def predict_proba_torch(state: LogisticState, x: torch.Tensor, *, require_cuda: bool = False) -> torch.Tensor:
    if require_cuda:
        _require_cuda_tensor("x", x)
    logits = ((x - state.mean) / state.scale) @ state.weights + state.bias
    return torch.sigmoid(logits)


def _average_ranks_torch(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks_sorted = torch.empty(scores.numel(), device=scores.device, dtype=torch.float32)
    if scores.numel() == 0:
        return ranks_sorted
    starts = torch.cat(
        [
            torch.tensor([0], device=scores.device),
            torch.nonzero(sorted_scores[1:] != sorted_scores[:-1], as_tuple=False).flatten() + 1,
        ]
    )
    ends = torch.cat([starts[1:], torch.tensor([scores.numel()], device=scores.device)])
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc_torch(labels: torch.Tensor, scores: torch.Tensor, *, require_cuda: bool = True) -> torch.Tensor:
    if require_cuda:
        _require_cuda_tensor("labels", labels)
        _require_cuda_tensor("scores", scores)
    labels = labels.to(dtype=torch.bool)
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return torch.tensor(float("nan"), device=scores.device)
    rank_sum = _average_ranks_torch(scores)[labels].sum()
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision_torch(labels: torch.Tensor, scores: torch.Tensor, *, require_cuda: bool = True) -> torch.Tensor:
    if require_cuda:
        _require_cuda_tensor("labels", labels)
        _require_cuda_tensor("scores", scores)
    labels = labels.to(dtype=torch.float32)
    positives = labels.sum()
    if positives == 0:
        return torch.tensor(float("nan"), device=scores.device)
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    precision = torch.cumsum(sorted_labels, dim=0) / torch.arange(
        1,
        sorted_labels.numel() + 1,
        device=scores.device,
        dtype=torch.float32,
    )
    return (precision * sorted_labels).sum() / positives


def compute_metrics_torch(
    labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    require_cuda: bool = True,
) -> dict[str, torch.Tensor]:
    return {
        "roc_auc": roc_auc_torch(labels, scores, require_cuda=require_cuda),
        "pr_auc": average_precision_torch(labels, scores, require_cuda=require_cuda),
    }


def bootstrap_metric_intervals_torch(
    labels: torch.Tensor,
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    n_resamples: int,
    random_state: int,
    require_cuda: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    if require_cuda:
        _require_cuda_tensor("labels", labels)
        _require_cuda_tensor("scores", scores)
        _require_cuda_tensor("group_ids", group_ids)
    unique_groups = torch.unique(group_ids)
    if unique_groups.numel() == 0 or n_resamples <= 0:
        point = compute_metrics_torch(labels, scores, require_cuda=require_cuda)
        return {name: {"point": value, "lower": value, "upper": value} for name, value in point.items()}
    generator = torch.Generator(device=labels.device)
    generator.manual_seed(int(random_state))
    _, inverse_groups = torch.unique(group_ids, sorted=True, return_inverse=True)
    group_counts = torch.bincount(inverse_groups)
    max_group_size = int(group_counts.max().item())
    padded_group_rows = torch.full(
        (unique_groups.numel(), max_group_size),
        -1,
        device=labels.device,
        dtype=torch.int64,
    )
    for group_index in range(int(unique_groups.numel())):
        rows = torch.nonzero(inverse_groups == group_index, as_tuple=False).flatten()
        padded_group_rows[group_index, : rows.numel()] = rows
    roc_values: list[torch.Tensor] = []
    pr_values: list[torch.Tensor] = []
    for _ in range(int(n_resamples)):
        sampled = torch.randint(unique_groups.numel(), (unique_groups.numel(),), device=labels.device, generator=generator)
        selected = padded_group_rows[sampled].reshape(-1)
        row_index = selected[selected >= 0]
        sampled_labels = labels[row_index]
        sampled_scores = scores[row_index]
        if torch.unique(sampled_labels).numel() < 2:
            continue
        metric = compute_metrics_torch(sampled_labels, sampled_scores, require_cuda=require_cuda)
        roc_values.append(metric["roc_auc"])
        pr_values.append(metric["pr_auc"])
    point = compute_metrics_torch(labels, scores, require_cuda=require_cuda)
    intervals: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in {"roc_auc": roc_values, "pr_auc": pr_values}.items():
        if values:
            stacked = torch.stack(values)
            intervals[name] = {
                "point": point[name],
                "lower": torch.quantile(stacked, 0.025),
                "upper": torch.quantile(stacked, 0.975),
            }
        else:
            intervals[name] = {"point": point[name], "lower": point[name], "upper": point[name]}
    return intervals


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    split_strategy: str,
    device: torch.device,
    test_size: float = 0.3,
    random_state: int = 13,
    num_folds: int = 5,
    l2: float = 1.0,
    max_iter: int = 100,
    bootstrap_resamples: int = 1000,
) -> tuple[dict[str, float], pd.DataFrame, list[LogisticState]]:
    require_cuda = device.type == "cuda"
    result_frames: list[pd.DataFrame] = []
    states: list[LogisticState] = []
    for fold, train_frame, eval_frame in _build_train_eval_splits(
        frame,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        num_folds=num_folds,
    ):
        train_x, train_y = _tensor_frame(train_frame, columns, device)
        state = fit_logistic_regression_torch(
            train_x,
            train_y,
            l2=l2,
            max_iter=max_iter,
            require_cuda=require_cuda,
        )
        eval_x, _ = _tensor_frame(eval_frame, columns, device)
        scores = predict_proba_torch(state, eval_x, require_cuda=require_cuda)
        predictions = (scores >= 0.5).to(torch.int64)
        states.append(state)
        result_frames.append(
            eval_frame.assign(
                prediction=predictions.cpu().numpy(),
                score=scores.detach().cpu().numpy(),
                fold=fold,
            ).reset_index(drop=True)
        )
    predictions = pd.concat(result_frames, ignore_index=True)
    if "sample_id" in predictions.columns:
        predictions = predictions.sort_values("sample_id").reset_index(drop=True)
    labels = torch.as_tensor(predictions["label"].to_numpy("int64"), device=device)
    scores = torch.as_tensor(predictions["score"].to_numpy("float32"), device=device)
    group_column = "image_id" if split_strategy == "image_grouped" else "object_name" if split_strategy == "object_heldout" else "sample_id"
    if group_column not in predictions.columns:
        group_column = "sample_id"
    group_codes = pd.factorize(predictions[group_column], sort=True)[0]
    group_ids = torch.as_tensor(group_codes, dtype=torch.int64, device=device)
    intervals = bootstrap_metric_intervals_torch(
        labels,
        scores,
        group_ids,
        n_resamples=bootstrap_resamples,
        random_state=random_state,
        require_cuda=require_cuda,
    )
    metrics: dict[str, float] = {}
    for name, interval in intervals.items():
        metrics[name] = float(interval["point"].detach().cpu())
        metrics[f"{name}_ci_lower"] = float(interval["lower"].detach().cpu())
        metrics[f"{name}_ci_upper"] = float(interval["upper"].detach().cpu())
    return metrics, predictions, states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--columns", default="auto", help="Comma list, drift_only, full_curve, or auto.")
    parser.add_argument("--variant-name", choices=["drift_only", "full_curve"], default="full_curve")
    parser.add_argument("--split-strategy", choices=["row", "image_grouped", "object_heldout"], default="image_grouped")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-parquet", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = _device_from_arg(args.device)
    frame = pd.read_parquet(args.features)
    columns = resolve_columns(frame, args.columns, args.variant_name)
    metrics, predictions, _ = evaluate_frame(
        frame,
        columns=columns,
        split_strategy=args.split_strategy,
        device=device,
        test_size=args.test_size,
        random_state=args.random_state,
        num_folds=args.num_folds,
        l2=args.l2,
        max_iter=args.max_iter,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    payload = {
        "features": str(args.features),
        "variant_name": args.variant_name,
        "columns": columns,
        "split_strategy": args.split_strategy,
        "device": str(device),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "n_rows": int(len(frame)),
        **metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.predictions_parquet is not None:
        keep_columns = list(
            dict.fromkeys(
                column
                for column in [*METADATA_COLUMNS, "prediction", "score", "fold"]
                if column in predictions.columns
            )
        )
        args.predictions_parquet.parent.mkdir(parents=True, exist_ok=True)
        predictions[keep_columns].to_parquet(args.predictions_parquet, index=False)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
