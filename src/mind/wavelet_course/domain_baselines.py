"""Official HALP and linear probes on the wavelet-course RePOPE split."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import final_hidden_logreg, mean_layer_hidden_logreg
from .metrics import (
    METRIC_NAMES,
    binary_metrics,
    evaluate_validation_test,
    select_best_f1_threshold,
)
from .population import population_key
from .utils import SPLIT_NAMES, write_csv_rows


HALP_CACHE_IDENTITY_FIELDS = (
    "model_name",
    "dataset_name",
    "subset",
    "sample_id",
)

OFFICIAL_HALP_READOUT_FIELDS = (
    "vision_features",
    "query_hidden_states",
    "vision_token_hidden_states",
    "query_token_index",
    "vision_token_span",
)

OFFICIAL_HALP_REQUIRED_FIELDS = (
    "sample_id",
    "image_id",
    "label",
    "parsed_answer",
    "subset",
    "object_name",
    *OFFICIAL_HALP_READOUT_FIELDS,
)

DOMAIN_BASELINE_FIELDS = (
    "model_name",
    "dataset_name",
    "subset_scope",
    "seed",
    "quick_run",
    "baseline_name",
    "method_family",
    "source",
    "representation",
    "readout",
    "classifier",
    "status",
    "failure_reason",
    "feature_shape",
    "train_samples",
    "validation_samples",
    "test_samples",
    "train_pos",
    "validation_pos",
    "test_pos",
    "best_val_threshold",
    "test_pr_auc",
    "test_average_precision",
    "test_roc_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_balanced_accuracy",
    "test_tpr_at_1pct_fpr",
    "test_fpr_at_95pct_tpr",
    "best_epoch",
    "early_stopped",
    "converged",
    "selected_probe",
    "halp_layer_indices",
    "num_halp_candidates",
    "selection_metric",
    "feature_seconds",
    "train_eval_seconds",
    "total_seconds",
)

HALP_SELECTION_FIELDS = (
    "probe_name",
    "feature_shape",
    "validation_pr_auc",
    "validation_average_precision",
    "validation_roc_auc",
    "validation_f1",
    "validation_precision",
    "validation_recall",
    "selected",
)

HALP_ROW_SELECTION_FIELDS = (
    "probe_name",
    "feature_shape",
    "eval_pr_auc",
    "eval_average_precision",
    "eval_roc_auc",
    "eval_f1",
    "eval_precision",
    "eval_recall",
    "selected",
)

HALP_RESULT_FIELDS = (
    "sample_id",
    "image_id",
    "subset",
    "object_name",
    "label",
    "score",
    "prediction",
    "selected_probe",
)


def run_domain_baselines(
    entries: Sequence[Mapping[str, Any]],
    labels: Sequence[int] | np.ndarray,
    *,
    output_dir: Path | str,
    model_name: str,
    dataset_name: str,
    subsets: Sequence[str],
    seed: int,
    device: str,
    logreg_max_iter: int = 20000,
    halp_hidden_dims: Sequence[int] = (512, 256, 128),
    halp_dropout: float = 0.3,
    halp_learning_rate: float = 1e-3,
    halp_batch_size: int = 32,
    halp_epochs: int = 50,
    quick: bool = False,
) -> dict[str, Any]:
    """Run official HALP and linear probes on one fixed population/split."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    labels_array = _as_labels(labels, expected_size=len(entries))
    split_info = SplitInfo.from_entries(entries, labels_array)
    common = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "subset_scope": ",".join(str(item) for item in subsets),
        "seed": int(seed),
        "quick_run": bool(quick),
    }

    halp_row, selection_rows, result_rows = run_official_halp(
        entries,
        labels_array,
        split_info=split_info,
        seed=seed,
        device=device,
        output_dir=output_path,
        hidden_dims=tuple(int(dim) for dim in halp_hidden_dims),
        dropout=float(halp_dropout),
        learning_rate=float(halp_learning_rate),
        batch_size=int(halp_batch_size),
        epochs=int(halp_epochs),
    )
    halp_row_protocol_row, row_protocol_selection_rows, row_protocol_result_rows = run_official_halp_row_protocol(
        entries,
        output_dir=output_path,
        seed=13,
        device=device,
        hidden_dims=tuple(int(dim) for dim in halp_hidden_dims),
        dropout=float(halp_dropout),
        learning_rate=float(halp_learning_rate),
        batch_size=int(halp_batch_size),
        epochs=int(halp_epochs),
    )
    if row_protocol_selection_rows:
        write_csv_rows(
            output_path / "halp_official_row_selection.csv",
            row_protocol_selection_rows,
            HALP_ROW_SELECTION_FIELDS,
        )
    else:
        write_csv_rows(output_path / "halp_official_row_selection.csv", [], HALP_ROW_SELECTION_FIELDS)
    if row_protocol_result_rows:
        write_csv_rows(output_path / "halp_official_row_results.csv", row_protocol_result_rows, HALP_RESULT_FIELDS)
    else:
        write_csv_rows(output_path / "halp_official_row_results.csv", [], HALP_RESULT_FIELDS)

    rows: list[dict[str, Any]] = [halp_row, halp_row_protocol_row]

    for baseline_name, builder in (
        ("linear_probe_final_hidden_logreg", final_hidden_logreg),
        ("linear_probe_mean_layer_hidden_logreg", mean_layer_hidden_logreg),
    ):
        rows.append(
            _run_feature_logreg(
                baseline_name,
                method_family="linear_probe",
                source="linear_probe",
                representation=baseline_name.replace("linear_probe_", "").replace("_logreg", ""),
                entries=entries,
                labels=labels_array,
                split_info=split_info,
                feature_builder=builder,
                seed=seed,
                max_iter=logreg_max_iter,
            )
        )

    rows = [{**row, **common} for row in rows]
    csv_path = output_path / "domain_baselines.csv"
    summary_path = output_path / "domain_baseline_comparison.md"
    write_domain_baselines_csv(rows, csv_path)
    write_domain_baseline_summary(
        rows,
        summary_path,
        model_name=model_name,
        dataset_name=dataset_name,
        split_info=split_info,
    )
    return {
        "rows": rows,
        "halp_selection_rows": selection_rows,
        "halp_result_rows": result_rows,
        "halp_row_protocol_selection_rows": row_protocol_selection_rows,
        "halp_row_protocol_result_rows": row_protocol_result_rows,
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "best_rows": best_rows_by_family(rows),
    }


class SplitInfo:
    """Boolean masks and label counts for the fixed wavelet-course split."""

    def __init__(self, split_names: np.ndarray, labels: np.ndarray) -> None:
        if split_names.shape[0] != labels.shape[0]:
            raise ValueError("split names and labels must have matching lengths")
        self.split_names = split_names
        self.labels = labels
        self.masks = {split: split_names == split for split in SPLIT_NAMES}
        for split, mask in self.masks.items():
            if int(mask.sum()) <= 0:
                raise ValueError(f"split {split!r} has no samples")

    @classmethod
    def from_entries(cls, entries: Sequence[Mapping[str, Any]], labels: np.ndarray) -> "SplitInfo":
        split_names = np.asarray([str(entry.get("wavelet_split", "")) for entry in entries])
        unknown = sorted(set(split_names.tolist()) - set(SPLIT_NAMES))
        if unknown:
            raise ValueError(f"unexpected wavelet_split values: {unknown}")
        return cls(split_names, labels)

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for split, mask in self.masks.items():
            result[f"{split}_samples"] = int(mask.sum())
            result[f"{split}_pos"] = int(self.labels[mask].sum())
        return result

    def split_arrays(self, features: np.ndarray) -> dict[str, np.ndarray]:
        matrix = _as_2d_features(features, expected_size=self.labels.shape[0])
        return {
            "train_x": matrix[self.masks["train"]],
            "train_y": self.labels[self.masks["train"]],
            "validation_x": matrix[self.masks["validation"]],
            "validation_y": self.labels[self.masks["validation"]],
            "test_x": matrix[self.masks["test"]],
            "test_y": self.labels[self.masks["test"]],
        }


def run_official_halp(
    entries: Sequence[Mapping[str, Any]],
    labels: Sequence[int] | np.ndarray,
    *,
    split_info: SplitInfo,
    seed: int,
    device: str,
    output_dir: Path | str,
    hidden_dims: Sequence[int] = (512, 256, 128),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the official HALP probe family on the fixed train/val/test split.

    This ports the old HALP implementation's probe architecture and feature
    definitions: ``vision_only`` plus query-token and vision-token probes at
    five layer positions.  The only split change is using the wavelet-course
    grouped train/validation/test split so the comparison is on the same data.
    """

    total_start = time.perf_counter()
    output_path = Path(output_dir)
    missing = missing_official_halp_fields(entries)
    if missing:
        _write_empty_halp_detail_files(output_path)
        row = _failure_row(
            "halp_official_mlp",
            method_family="halp_official",
            source="halp_official",
            representation="official_halp_probe_family",
            readout="halp_probe",
            classifier="halp_mlp",
            split_info=split_info,
            failure_reason="missing_official_halp_fields: " + ",".join(missing),
            total_seconds=time.perf_counter() - total_start,
        )
        return row, [], []

    labels_array = _as_labels(labels, expected_size=len(entries))
    probe_names = resolve_halp_probe_names(entries)
    layer_indices = resolve_halp_layer_indices(_resolve_total_layers(entries[0]))
    selection_rows: list[dict[str, Any]] = []
    candidate_payloads: list[dict[str, Any]] = []
    feature_start = time.perf_counter()
    for probe_name in probe_names:
        features = build_halp_feature_matrix(entries, probe_name)
        arrays = split_info.split_arrays(features)
        _require_two_classes(arrays["train_y"], name="HALP train_y")
        model = _fit_halp_probe(
            arrays["train_x"],
            arrays["train_y"],
            input_dim=features.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        validation_scores = _score_halp_probe(model, arrays["validation_x"], device=device)
        test_scores = _score_halp_probe(model, arrays["test_x"], device=device)
        threshold = select_best_f1_threshold(arrays["validation_y"], validation_scores).threshold
        validation_metrics = binary_metrics(arrays["validation_y"], validation_scores, threshold=threshold)
        selection_row = {
            "probe_name": probe_name,
            "feature_shape": "x".join(str(int(dim)) for dim in features.shape),
            "validation_pr_auc": validation_metrics["pr_auc"],
            "validation_average_precision": validation_metrics["average_precision"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_f1": validation_metrics["f1"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "selected": False,
        }
        selection_rows.append(selection_row)
        candidate_payloads.append(
            {
                "probe_name": probe_name,
                "feature_shape": features.shape,
                "validation_scores": validation_scores,
                "test_scores": test_scores,
                "arrays": arrays,
                "selection_row": selection_row,
            }
        )
    feature_seconds = time.perf_counter() - feature_start

    selected = max(
        candidate_payloads,
        key=lambda payload: (
            _metric_value(payload["selection_row"]["validation_roc_auc"]),
            _metric_value(payload["selection_row"]["validation_pr_auc"]),
            str(payload["probe_name"]),
        ),
    )
    for row in selection_rows:
        row["selected"] = row["probe_name"] == selected["probe_name"]

    arrays = selected["arrays"]
    evaluation = evaluate_validation_test(
        arrays["validation_y"],
        selected["validation_scores"],
        arrays["test_y"],
        selected["test_scores"],
    )
    threshold = float(evaluation["threshold"])
    result_rows = _halp_result_rows(
        entries,
        labels_array,
        split_info=split_info,
        test_scores=selected["test_scores"],
        threshold=threshold,
        selected_probe=str(selected["probe_name"]),
    )
    write_csv_rows(output_path / "halp_selection.csv", selection_rows, HALP_SELECTION_FIELDS)
    write_csv_rows(output_path / "halp_results.csv", result_rows, HALP_RESULT_FIELDS)
    row = _success_row(
        "halp_official_mlp",
        method_family="halp_official",
        source="halp_official",
        representation="official_halp_probe_family",
        readout="halp_probe",
        classifier="halp_mlp",
        split_info=split_info,
        feature_shape=selected["feature_shape"],
        evaluation=evaluation,
        feature_seconds=feature_seconds,
        train_eval_seconds=time.perf_counter() - total_start - feature_seconds,
        total_seconds=time.perf_counter() - total_start,
        extra={
            "selected_probe": selected["probe_name"],
            "halp_layer_indices": ",".join(str(index) for index in layer_indices),
            "num_halp_candidates": len(probe_names),
            "selection_metric": "validation_roc_auc_then_pr_auc",
        },
    )
    return row, selection_rows, result_rows


def run_official_halp_row_protocol(
    entries: Sequence[Mapping[str, Any]],
    labels: Sequence[int] | np.ndarray | None = None,
    *,
    output_dir: Path | str | None = None,
    seed: int = 13,
    device: str = "cpu",
    hidden_dims: Sequence[int] = (512, 256, 128),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    test_size: float = 0.2,
    layer_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the old MIND official HALP row-split protocol.

    This mirrors the legacy ``run_halp.py --split-strategy row`` protocol:
    object-hallucination labels are recomputed from each entry's
    ``label``/``parsed_answer`` fields, rows are split with a stratified
    train/eval split, all 11 HALP probes are trained on train and evaluated on
    the same eval split, the best probe is selected by ``(roc_auc, pr_auc,
    probe_name)``, and predictions use the fixed threshold 0.5.
    """

    del output_dir
    total_start = time.perf_counter()
    if labels is not None:
        _as_labels(labels, expected_size=len(entries))
    missing = missing_official_halp_fields(entries)
    split_counts = _empty_row_protocol_counts()
    if missing:
        return (
            _row_protocol_failure_row(
                failure_reason="missing_official_halp_fields: " + ",".join(missing),
                split_counts=split_counts,
                total_seconds=time.perf_counter() - total_start,
            ),
            [],
            [],
        )

    object_labels = _object_hallucination_labels(entries)
    train_indices, eval_indices = _stratified_row_indices(
        object_labels,
        test_size=float(test_size),
        random_state=int(seed),
    )
    split_counts = _row_protocol_counts(object_labels, train_indices=train_indices, eval_indices=eval_indices)
    probe_names = _resolve_halp_probe_names_for_row_protocol(entries, layer_indices=layer_indices)
    selection_rows: list[dict[str, Any]] = []
    candidate_payloads: list[dict[str, Any]] = []
    feature_start = time.perf_counter()
    for probe_name in probe_names:
        features = build_halp_feature_matrix(entries, probe_name)
        train_x = features[train_indices]
        train_y = object_labels[train_indices]
        eval_x = features[eval_indices]
        eval_y = object_labels[eval_indices]
        _require_two_classes(train_y, name="HALP row protocol train_y")
        model = _fit_halp_probe(
            train_x,
            train_y,
            input_dim=features.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        eval_scores = _score_halp_probe(model, eval_x, device=device)
        eval_metrics = binary_metrics(eval_y, eval_scores, threshold=0.5)
        selection_row = {
            "probe_name": probe_name,
            "feature_shape": "x".join(str(int(dim)) for dim in features.shape),
            "eval_pr_auc": eval_metrics["pr_auc"],
            "eval_average_precision": eval_metrics["average_precision"],
            "eval_roc_auc": eval_metrics["roc_auc"],
            "eval_f1": eval_metrics["f1"],
            "eval_precision": eval_metrics["precision"],
            "eval_recall": eval_metrics["recall"],
            "selected": False,
        }
        selection_rows.append(selection_row)
        candidate_payloads.append(
            {
                "probe_name": probe_name,
                "feature_shape": features.shape,
                "eval_scores": eval_scores,
                "eval_y": eval_y,
                "selection_row": selection_row,
            }
        )
    feature_seconds = time.perf_counter() - feature_start

    selected = max(
        candidate_payloads,
        key=lambda payload: (
            _metric_value(payload["selection_row"]["eval_roc_auc"]),
            _metric_value(payload["selection_row"]["eval_pr_auc"]),
            str(payload["probe_name"]),
        ),
    )
    for row in selection_rows:
        row["selected"] = row["probe_name"] == selected["probe_name"]

    evaluation = {"threshold": 0.5, "test": binary_metrics(selected["eval_y"], selected["eval_scores"], threshold=0.5)}
    result_rows = _halp_row_protocol_result_rows(
        entries,
        object_labels,
        eval_indices=eval_indices,
        eval_scores=selected["eval_scores"],
        selected_probe=str(selected["probe_name"]),
    )
    row = _row_protocol_success_row(
        split_counts=split_counts,
        feature_shape=selected["feature_shape"],
        evaluation=evaluation,
        feature_seconds=feature_seconds,
        train_eval_seconds=time.perf_counter() - total_start - feature_seconds,
        total_seconds=time.perf_counter() - total_start,
        selected_probe=str(selected["probe_name"]),
        layer_indices=_resolve_halp_layer_indices_for_row_protocol(entries, layer_indices=layer_indices),
        num_candidates=len(probe_names),
    )
    return row, selection_rows, result_rows


def write_domain_baselines_csv(rows: Sequence[Mapping[str, Any]], output: Path | str) -> Path:
    path = Path(output)
    write_csv_rows(path, rows, DOMAIN_BASELINE_FIELDS)
    return path


def load_halp_readout_cache(path: Path | str) -> list[dict[str, Any]]:
    """Load an external official-HALP readout cache.

    Supported payloads are ``list[dict]`` and ``{"entries": list[dict]}``
    stored as a PyTorch ``.pt`` file.  A directory containing direct
    ``shard-*.pt`` files or partition subdirectories with ``shard-*.pt`` files
    is also supported for the extractor output.  Validation is intentionally
    strict: malformed cache rows fail before any baseline runs with partially
    merged fields.
    """

    cache_path = Path(path)
    if cache_path.is_dir():
        shard_paths = sorted(cache_path.rglob("shard-*.pt"))
        if not shard_paths:
            raise ValueError(f"HALP readout cache directory contains no shard-*.pt files: {cache_path}")
        rows: list[dict[str, Any]] = []
        for shard_path in shard_paths:
            rows.extend(_load_halp_readout_cache_file(shard_path))
        if not rows:
            raise ValueError(f"HALP readout cache directory is empty: {cache_path}")
        return rows
    return _load_halp_readout_cache_file(cache_path)


def _load_halp_readout_cache_file(cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.suffix != ".pt":
        raise ValueError(f"HALP readout cache must be a .pt file, got {cache_path}")
    import torch

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        if "entries" not in payload:
            raise ValueError("HALP readout cache mapping must contain an 'entries' field")
        payload = payload["entries"]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("HALP readout cache payload must be a sequence of row mappings")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"HALP readout cache row {index} must be a mapping")
        normalized = dict(row)
        _validate_halp_cache_row(normalized, context=f"HALP readout cache row {index}")
        rows.append(normalized)
    if not rows:
        raise ValueError("HALP readout cache must not be empty")
    return rows


def load_halp_readout_cache_entries(path: Path | str) -> list[dict[str, Any]]:
    """Backward-compatible public name for loading HALP readout cache entries."""

    return load_halp_readout_cache(path)


def merge_halp_readout_cache(
    entries: Sequence[Mapping[str, Any]],
    readout_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge external HALP readouts into wavelet-course population entries.

    Cache order is deliberately ignored.  Rows are matched by the same stable
    key used by the wavelet-course population:
    ``[model_name, dataset_name, subset, sample_id]``.
    """

    readout_by_key: dict[str, Mapping[str, Any]] = {}
    for index, readout in enumerate(readout_rows):
        if not isinstance(readout, Mapping):
            raise ValueError(f"HALP readout row {index} must be a mapping")
        _validate_halp_cache_row(readout, context=f"HALP readout row {index}")
        key = population_key(readout)
        if key in readout_by_key:
            raise ValueError(f"duplicate HALP readout cache key: {key}")
        readout_by_key[key] = readout

    merged: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        key = population_key(entry)
        if key not in readout_by_key:
            raise ValueError(f"missing HALP readout cache key for primary entry {index}: {key}")
        readout = readout_by_key[key]
        _validate_halp_metadata_consistency(entry, readout, key=key)
        enriched = dict(entry)
        for field in OFFICIAL_HALP_READOUT_FIELDS:
            enriched[field] = readout[field]
        for optional_field in ("readout_format", "total_layers"):
            if optional_field in readout:
                enriched[optional_field] = readout[optional_field]
        _validate_halp_cache_row(enriched, context=f"merged HALP entry {index}")
        merged.append(enriched)
    return merged


def best_rows_by_family(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for baseline_name in ("halp_official_mlp", "halp_official_row_protocol"):
        candidates = [
            dict(row)
            for row in rows
            if row.get("baseline_name") == baseline_name and row.get("status") == "success"
        ]
        best = _best_by_metric(candidates, "test_pr_auc")
        if best is not None:
            result.append({"rank_name": f"best_{baseline_name}", **best})
    for family in ("linear_probe",):
        candidates = [
            dict(row)
            for row in rows
            if row.get("method_family") == family and row.get("status") == "success"
        ]
        best = _best_by_metric(candidates, "test_pr_auc")
        if best is not None:
            result.append({"rank_name": f"best_{family}", **best})
    overall = _best_by_metric([dict(row) for row in rows if row.get("status") == "success"], "test_pr_auc")
    if overall is not None:
        result.append({"rank_name": "best_domain_overall", **overall})
    return result


def write_domain_baseline_summary(
    rows: Sequence[Mapping[str, Any]],
    output: Path | str,
    *,
    model_name: str,
    dataset_name: str,
    split_info: SplitInfo,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = split_info.counts()
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") == "failure"]
    best_rows = best_rows_by_family(rows)
    lines = [
        "# Domain Baseline Comparison",
        "",
        (
            "这些结果使用小波课程 v2 的同一份 RePOPE readout cache。"
            "HALP 的两个协议分开报告：course-grouped 行使用课程 image_id grouped "
            "train/validation/test split；official-row 行使用 HALP 旧 row-stratified train/eval split。"
        ),
        "",
        f"- model: {model_name}",
        f"- dataset: {dataset_name}",
        f"- primary_population: {int(split_info.labels.shape[0])}",
        f"- hard_hallucinations: {int(split_info.labels.sum())}",
        f"- train: {counts['train_samples']} samples, {counts['train_pos']} positives",
        f"- validation: {counts['validation_samples']} samples, {counts['validation_pos']} positives",
        f"- test: {counts['test_samples']} samples, {counts['test_pos']} positives",
        f"- success_rows: {len(successes)}",
        f"- failure_rows: {len(failures)}",
        "",
        "## Baselines",
        "",
        "- `halp_official_mlp`: course-grouped HALP; validation selects probe and threshold; test reports metrics.",
        "- `halp_official_row_protocol`: old official row protocol; eval selects probe and reports metrics; threshold=0.5.",
        "- Linear probe: balanced logistic regression on final hidden state and mean-layer hidden state.",
        "",
        "## Best Rows",
        "",
    ]
    if best_rows:
        for row in best_rows:
            lines.append(
                "- {rank}: {name} PR-AUC={pr_auc} F1={f1} AP={ap}".format(
                    rank=row.get("rank_name", ""),
                    name=row.get("baseline_name", ""),
                    pr_auc=_fmt(row.get("test_pr_auc")),
                    f1=_fmt(row.get("test_f1")),
                    ap=_fmt(row.get("test_average_precision")),
                )
            )
    else:
        lines.append("- No successful domain baseline rows.")
    lines.extend(["", "## Official HALP Status", ""])
    halp_rows = [row for row in rows if row.get("method_family") == "halp_official"]
    if halp_rows:
        for row in halp_rows:
            baseline_name = str(row.get("baseline_name", ""))
            source = str(row.get("source", ""))
            is_row_protocol = (
                baseline_name == "halp_official_row_protocol"
                or source == "halp_official_legacy_row_protocol"
            )
            protocol = "official-row" if is_row_protocol else "course-grouped"
            if row.get("status") == "success":
                lines.append(
                    (
                        "- {name}: protocol={protocol}, selected_probe={probe}, threshold={threshold}, "
                        "selection_metric={metric}, candidates={count}, layer_indices={layers}, PR-AUC={pr_auc}"
                    ).format(
                        name=baseline_name,
                        protocol=protocol,
                        probe=row.get("selected_probe", ""),
                        threshold=_fmt(row.get("best_val_threshold")),
                        metric=row.get("selection_metric", ""),
                        count=row.get("num_halp_candidates", ""),
                        layers=row.get("halp_layer_indices", ""),
                        pr_auc=_fmt(row.get("test_pr_auc")),
                    )
                )
            else:
                lines.append(
                    "- {name}: protocol={protocol}, failure_reason={reason}".format(
                        name=baseline_name,
                        protocol=protocol,
                        reason=row.get("failure_reason"),
                    )
                )
    else:
        lines.append("- No official HALP row was produced.")
    lines.extend(["", "## All Successful Rows", ""])
    for row in sorted(successes, key=lambda item: _metric_value(item.get("test_pr_auc")), reverse=True):
        lines.append(
            "- {family} / {name}: PR-AUC={pr_auc}, F1={f1}, ROC-AUC={roc}".format(
                family=row.get("method_family", ""),
                name=row.get("baseline_name", ""),
                pr_auc=_fmt(row.get("test_pr_auc")),
                f1=_fmt(row.get("test_f1")),
                roc=_fmt(row.get("test_roc_auc")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def missing_official_halp_fields(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    if not entries:
        raise ValueError("entries must not be empty")
    missing: set[str] = set()
    for entry in entries:
        for field in OFFICIAL_HALP_REQUIRED_FIELDS:
            if field not in entry or entry[field] is None:
                missing.add(field)
    return sorted(missing)


def _validate_halp_cache_row(row: Mapping[str, Any], *, context: str) -> None:
    for field in HALP_CACHE_IDENTITY_FIELDS:
        if field not in row or row[field] is None:
            raise ValueError(f"{context}: missing identity field {field!r}")
    for field in OFFICIAL_HALP_READOUT_FIELDS:
        if field not in row or row[field] is None:
            raise ValueError(f"{context}: missing HALP readout field {field!r}")

    vision_features = _as_float_array(row["vision_features"], field=f"{context}.vision_features")
    if vision_features.ndim not in (1, 2):
        raise ValueError(f"{context}: vision_features must be 1D or 2D, got shape {vision_features.shape}")

    query_states = _as_float_array(row["query_hidden_states"], field=f"{context}.query_hidden_states")
    vision_states = _as_float_array(
        row["vision_token_hidden_states"],
        field=f"{context}.vision_token_hidden_states",
    )
    if query_states.ndim != 2:
        raise ValueError(f"{context}: query_hidden_states must have shape (layers, hidden_dim)")
    if vision_states.ndim != 2:
        raise ValueError(f"{context}: vision_token_hidden_states must have shape (layers, hidden_dim)")
    if query_states.shape != vision_states.shape:
        raise ValueError(
            f"{context}: query_hidden_states shape {query_states.shape} does not match "
            f"vision_token_hidden_states shape {vision_states.shape}"
        )

    query_token_index = _as_non_negative_int(row["query_token_index"], field=f"{context}.query_token_index")
    span = _as_vision_token_span(row["vision_token_span"], field=f"{context}.vision_token_span")
    if query_token_index < 0:
        raise ValueError(f"{context}: query_token_index must be non-negative")
    if span[0] < 0 or span[1] < span[0]:
        raise ValueError(f"{context}: vision_token_span must be [start, stop] with 0 <= start <= stop")


def _validate_halp_metadata_consistency(
    primary: Mapping[str, Any],
    readout: Mapping[str, Any],
    *,
    key: str,
) -> None:
    for field in ("label", "parsed_answer"):
        if field not in readout or readout[field] is None:
            continue
        primary_value = _normalise_compare_scalar(primary.get(field), field=f"primary.{field}")
        readout_value = _normalise_compare_scalar(readout[field], field=f"readout.{field}")
        if primary_value != readout_value:
            raise ValueError(
                f"{field} mismatch in HALP readout metadata for key={key}: "
                f"primary={primary_value!r}, readout={readout_value!r}"
            )


def _as_non_negative_int(value: Any, *, field: str) -> int:
    scalar = _scalar_value(value, field=field)
    if isinstance(scalar, bool):
        raise ValueError(f"{field} must be an integer, not bool")
    if isinstance(scalar, float):
        if not math.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{field} must be an integer")
        scalar = int(scalar)
    if not isinstance(scalar, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    result = int(scalar)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _as_vision_token_span(value: Any, *, field: str) -> tuple[int, int]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (2,):
        raise ValueError(f"{field} must contain exactly two integers")
    return (
        _as_non_negative_int(array[0], field=f"{field}[0]"),
        _as_non_negative_int(array[1], field=f"{field}[1]"),
    )


def _normalise_compare_scalar(value: Any, *, field: str) -> Any:
    scalar = _scalar_value(value, field=field)
    if scalar is None:
        return None
    if isinstance(scalar, bool):
        return int(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, float):
        if not math.isfinite(scalar):
            raise ValueError(f"{field} must be finite")
        if scalar.is_integer():
            return int(scalar)
        return float(scalar)
    return str(scalar)


def _scalar_value(value: Any, *, field: str) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
        if int(value.numel()) != 1:
            raise ValueError(f"{field} must be scalar")
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{field} must be scalar")
        return value.reshape(()).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_halp_layer_indices(total_layers: int) -> list[int]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    selected: list[int] = []
    for raw_index in (0, total_layers // 4, total_layers // 2, (3 * total_layers) // 4, total_layers - 1):
        index = max(0, min(int(total_layers) - 1, int(raw_index)))
        if index not in selected:
            selected.append(index)
    return selected


def resolve_halp_probe_names(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    if not entries:
        raise ValueError("entries must not be empty")
    layer_indices = resolve_halp_layer_indices(_resolve_total_layers(entries[0]))
    names = ["vision_only"]
    for layer_index in layer_indices:
        names.append(f"vision_token_layer_{layer_index}")
        names.append(f"query_token_layer_{layer_index}")
    return names


def build_halp_feature_matrix(entries: Sequence[Mapping[str, Any]], probe_name: str) -> np.ndarray:
    if not entries:
        raise ValueError("entries must not be empty")
    matrix = np.stack([_halp_feature_vector(entry, probe_name) for entry in entries], axis=0)
    return _as_2d_features(matrix, expected_size=len(entries))


def _halp_feature_vector(entry: Mapping[str, Any], probe_name: str) -> np.ndarray:
    kind, layer_index = _parse_halp_probe_name(probe_name)
    if kind == "vision_only":
        return _flatten_vision_features(_as_float_array(entry["vision_features"], field="vision_features"))
    if kind == "vision_token":
        states = _as_float_array(entry["vision_token_hidden_states"], field="vision_token_hidden_states")
        return _layer_vector(states, layer_index=layer_index, field="vision_token_hidden_states")
    if kind == "query_token":
        states = _as_float_array(entry["query_hidden_states"], field="query_hidden_states")
        return _layer_vector(states, layer_index=layer_index, field="query_hidden_states")
    raise AssertionError(f"unhandled HALP probe kind: {kind}")


def _parse_halp_probe_name(probe_name: str) -> tuple[str, int]:
    if probe_name == "vision_only":
        return "vision_only", -1
    if probe_name.startswith("vision_token_layer_"):
        return "vision_token", int(probe_name.rsplit("_", 1)[-1])
    if probe_name.startswith("query_token_layer_"):
        return "query_token", int(probe_name.rsplit("_", 1)[-1])
    raise ValueError(f"unknown HALP probe name: {probe_name}")


def _resolve_total_layers(entry: Mapping[str, Any]) -> int:
    return int(_as_float_array(entry["query_hidden_states"], field="query_hidden_states").shape[0])


def _flatten_vision_features(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        return array.mean(axis=0, dtype=np.float32)
    raise ValueError(f"vision_features must be 1D or 2D, got shape {array.shape}")


def _layer_vector(array: np.ndarray, *, layer_index: int, field: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{field} must have shape (layers, hidden_dim)")
    if layer_index < 0 or layer_index >= array.shape[0]:
        raise ValueError(f"{field} layer index {layer_index} out of range for {array.shape[0]} layers")
    return array[layer_index].astype(np.float32, copy=False)


def _as_float_array(value: Any, *, field: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{field} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains NaN or Inf")
    return array


def _fit_halp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device: str,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0:
        raise ValueError("HALP epochs must be positive")
    if batch_size <= 0:
        raise ValueError("HALP batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("HALP learning_rate must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("HALP dropout must be in [0, 1)")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    target = torch.device(device)
    features = torch.tensor(train_x, dtype=torch.float32)
    labels = torch.tensor(train_y.astype(np.float32, copy=False), dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    effective_batch = max(2, min(int(batch_size), len(dataset)))
    drop_last = len(dataset) > effective_batch and len(dataset) % effective_batch == 1
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=target.type == "cuda",
    )

    model = _HALPProbe(input_dim=int(input_dim), hidden_dims=hidden_dims, dropout=dropout).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _epoch in range(int(epochs)):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(target, non_blocking=target.type == "cuda")
            batch_y = batch_y.to(target, non_blocking=target.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


class _HALPProbe:  # populated lazily to avoid importing torch at module import time
    pass


def _make_halp_probe_class() -> type[Any]:
    import torch
    from torch import nn

    class HALPProbe(nn.Module):
        def __init__(self, *, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            current_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                hidden = int(hidden_dim)
                if hidden <= 0:
                    raise ValueError("HALP hidden dimensions must be positive")
                layers.append(nn.Linear(current_dim, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden))
                layers.append(nn.Dropout(float(dropout)))
                current_dim = hidden
            layers.append(nn.Linear(current_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features.to(dtype=torch.float32)).squeeze(-1)

    return HALPProbe


_HALPProbe = _make_halp_probe_class()


def _score_halp_probe(model: Any, values: np.ndarray, *, device: str) -> np.ndarray:
    import torch

    matrix = _as_2d_features(values, expected_size=values.shape[0])
    target = torch.device(device)
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(matrix, dtype=torch.float32, device=target)
        scores = torch.sigmoid(model(tensor)).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("HALP scores contain NaN or Inf")
    return scores


def _halp_result_rows(
    entries: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    *,
    split_info: SplitInfo,
    test_scores: np.ndarray,
    threshold: float,
    selected_probe: str,
) -> list[dict[str, Any]]:
    test_indices = np.flatnonzero(split_info.masks["test"])
    scores = np.asarray(test_scores, dtype=np.float64)
    if scores.shape[0] != test_indices.shape[0]:
        raise ValueError("HALP test score count must match test split size")
    rows: list[dict[str, Any]] = []
    for local_index, entry_index in enumerate(test_indices):
        entry = entries[int(entry_index)]
        score = float(scores[local_index])
        rows.append(
            {
                "sample_id": str(entry.get("sample_id", "")),
                "image_id": str(entry.get("image_id", "")),
                "subset": str(entry.get("subset", "")),
                "object_name": str(entry.get("object_name", "")),
                "label": int(labels[int(entry_index)]),
                "score": score,
                "prediction": int(score >= threshold),
                "selected_probe": selected_probe,
            }
        )
    return rows


def _halp_row_protocol_result_rows(
    entries: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    *,
    eval_indices: np.ndarray,
    eval_scores: np.ndarray,
    selected_probe: str,
) -> list[dict[str, Any]]:
    scores = np.asarray(eval_scores, dtype=np.float64)
    if scores.shape[0] != eval_indices.shape[0]:
        raise ValueError("HALP row protocol eval score count must match eval split size")
    rows: list[dict[str, Any]] = []
    for local_index, entry_index in enumerate(eval_indices):
        entry = entries[int(entry_index)]
        score = float(scores[local_index])
        rows.append(
            {
                "sample_id": str(entry.get("sample_id", "")),
                "image_id": str(entry.get("image_id", "")),
                "subset": str(entry.get("subset", "")),
                "object_name": str(entry.get("object_name", "")),
                "label": int(labels[int(entry_index)]),
                "score": score,
                "prediction": int(score >= 0.5),
                "selected_probe": selected_probe,
            }
        )
    return rows


def _object_hallucination_labels(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not entries:
        raise ValueError("entries must not be empty")
    labels: list[int] = []
    for index, entry in enumerate(entries):
        if "label" not in entry or entry["label"] is None:
            raise ValueError(f"entry {index} missing label")
        ground_truth = int(_normalise_compare_scalar(entry["label"], field=f"entry[{index}].label"))
        answer = entry.get("parsed_answer")
        answer_label = None if answer is None else int(_normalise_compare_scalar(answer, field=f"entry[{index}].parsed_answer"))
        labels.append(int(answer_label == 1 and ground_truth == 0))
    result = np.asarray(labels, dtype=np.int64)
    if not set(np.unique(result).tolist()).issubset({0, 1}):
        raise ValueError("object hallucination labels must be binary")
    return result


def _stratified_row_indices(
    labels: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    _require_two_classes(labels, name="HALP row protocol labels")
    indices = np.arange(labels.shape[0], dtype=np.int64)
    train_indices, eval_indices = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=labels,
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    eval_indices = np.asarray(eval_indices, dtype=np.int64)
    _require_two_classes(labels[train_indices], name="HALP row protocol train_y")
    _require_two_classes(labels[eval_indices], name="HALP row protocol eval_y")
    return train_indices, eval_indices


def _resolve_halp_layer_indices_for_row_protocol(
    entries: Sequence[Mapping[str, Any]],
    *,
    layer_indices: Sequence[int] | None,
) -> list[int]:
    if layer_indices is None:
        return resolve_halp_layer_indices(_resolve_total_layers(entries[0]))
    total_layers = _resolve_total_layers(entries[0])
    resolved: list[int] = []
    for raw_index in layer_indices:
        index = int(raw_index)
        if index < 0 or index >= total_layers:
            raise ValueError(f"HALP row protocol layer index {index} out of range for {total_layers} layers")
        resolved.append(index)
    if not resolved:
        raise ValueError("HALP row protocol layer_indices must not be empty")
    return resolved


def _resolve_halp_probe_names_for_row_protocol(
    entries: Sequence[Mapping[str, Any]],
    *,
    layer_indices: Sequence[int] | None,
) -> list[str]:
    selected_layers = _resolve_halp_layer_indices_for_row_protocol(entries, layer_indices=layer_indices)
    probe_names = ["vision_only"]
    for layer_index in selected_layers:
        probe_names.append(f"vision_token_layer_{layer_index}")
        probe_names.append(f"query_token_layer_{layer_index}")
    return probe_names


def _empty_row_protocol_counts() -> dict[str, int]:
    return {
        "train_samples": 0,
        "validation_samples": 0,
        "test_samples": 0,
        "train_pos": 0,
        "validation_pos": 0,
        "test_pos": 0,
    }


def _row_protocol_counts(
    labels: np.ndarray,
    *,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, int]:
    return {
        "train_samples": int(train_indices.shape[0]),
        "validation_samples": 0,
        "test_samples": int(eval_indices.shape[0]),
        "train_pos": int(labels[train_indices].sum()),
        "validation_pos": 0,
        "test_pos": int(labels[eval_indices].sum()),
    }


def _row_protocol_success_row(
    *,
    split_counts: Mapping[str, int],
    feature_shape: Sequence[int],
    evaluation: Mapping[str, Any],
    feature_seconds: float,
    train_eval_seconds: float,
    total_seconds: float,
    selected_probe: str,
    layer_indices: Sequence[int],
    num_candidates: int,
) -> dict[str, Any]:
    row = _row_protocol_base_row(split_counts)
    row.update(
        {
            "baseline_name": "halp_official_row_protocol",
            "method_family": "halp_official",
            "source": "halp_official_legacy_row_protocol",
            "representation": "official_halp_probe_family",
            "readout": "halp_probe",
            "classifier": "halp_mlp",
            "status": "success",
            "failure_reason": "",
            "feature_shape": "x".join(str(int(dim)) for dim in feature_shape),
            "best_val_threshold": 0.5,
            "feature_seconds": float(feature_seconds),
            "train_eval_seconds": float(train_eval_seconds),
            "total_seconds": float(total_seconds),
            "selected_probe": selected_probe,
            "halp_layer_indices": ",".join(str(index) for index in layer_indices),
            "num_halp_candidates": int(num_candidates),
            "selection_metric": "eval_roc_auc_then_pr_auc",
        }
    )
    for metric in METRIC_NAMES:
        row[f"test_{metric}"] = dict(evaluation["test"]).get(metric, float("nan"))
    return row


def _row_protocol_failure_row(
    *,
    failure_reason: str,
    split_counts: Mapping[str, int],
    total_seconds: float,
) -> dict[str, Any]:
    row = _row_protocol_base_row(split_counts)
    row.update(
        {
            "baseline_name": "halp_official_row_protocol",
            "method_family": "halp_official",
            "source": "halp_official_legacy_row_protocol",
            "representation": "official_halp_probe_family",
            "readout": "halp_probe",
            "classifier": "halp_mlp",
            "status": "failure",
            "failure_reason": str(failure_reason),
            "feature_shape": "",
            "best_val_threshold": float("nan"),
            "feature_seconds": float("nan"),
            "train_eval_seconds": float("nan"),
            "total_seconds": float(total_seconds),
            "selected_probe": "",
            "halp_layer_indices": "",
            "num_halp_candidates": "",
            "selection_metric": "",
        }
    )
    for metric in METRIC_NAMES:
        row[f"test_{metric}"] = float("nan")
    return row


def _row_protocol_base_row(split_counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "model_name": "",
        "dataset_name": "",
        "subset_scope": "",
        "seed": "",
        "quick_run": "",
        "baseline_name": "",
        "method_family": "",
        "source": "",
        "representation": "",
        "readout": "",
        "classifier": "",
        "train_samples": int(split_counts.get("train_samples", 0)),
        "validation_samples": int(split_counts.get("validation_samples", 0)),
        "test_samples": int(split_counts.get("test_samples", 0)),
        "train_pos": int(split_counts.get("train_pos", 0)),
        "validation_pos": int(split_counts.get("validation_pos", 0)),
        "test_pos": int(split_counts.get("test_pos", 0)),
        "best_epoch": "",
        "early_stopped": "",
        "converged": "",
    }


def _write_empty_halp_detail_files(output_dir: Path) -> None:
    write_csv_rows(output_dir / "halp_selection.csv", [], HALP_SELECTION_FIELDS)
    write_csv_rows(output_dir / "halp_results.csv", [], HALP_RESULT_FIELDS)


def _run_feature_logreg(
    baseline_name: str,
    *,
    method_family: str,
    source: str,
    representation: str,
    entries: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    split_info: SplitInfo,
    feature_builder: Any,
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    try:
        feature_start = time.perf_counter()
        features = feature_builder(entries)
        feature_seconds = time.perf_counter() - feature_start
        train_start = time.perf_counter()
        row = _train_logreg_row(
            baseline_name,
            method_family=method_family,
            source=source,
            representation=representation,
            features=features,
            labels=labels,
            split_info=split_info,
            seed=seed,
            max_iter=max_iter,
            feature_seconds=feature_seconds,
            total_start=total_start,
        )
        row["train_eval_seconds"] = time.perf_counter() - train_start
        row["total_seconds"] = time.perf_counter() - total_start
        return row
    except Exception as error:
        return _failure_row(
            baseline_name,
            method_family=method_family,
            source=source,
            representation=representation,
            readout="logreg",
            classifier="logreg",
            split_info=split_info,
            failure_reason=str(error),
            total_seconds=time.perf_counter() - total_start,
        )


def _train_logreg_row(
    baseline_name: str,
    *,
    method_family: str,
    source: str,
    representation: str,
    features: np.ndarray,
    labels: np.ndarray,
    split_info: SplitInfo,
    seed: int,
    max_iter: int,
    feature_seconds: float,
    total_start: float,
) -> dict[str, Any]:
    from .baselines import train_logistic_regression

    arrays = split_info.split_arrays(features)
    result = train_logistic_regression(
        arrays["train_x"],
        arrays["train_y"],
        validation_x=arrays["validation_x"],
        test_x=arrays["test_x"],
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    evaluation = evaluate_validation_test(
        arrays["validation_y"],
        result.scores.validation,
        arrays["test_y"],
        result.scores.test,
    )
    return _success_row(
        baseline_name,
        method_family=method_family,
        source=source,
        representation=representation,
        readout="logreg",
        classifier="logreg",
        split_info=split_info,
        feature_shape=np.asarray(features).shape,
        evaluation=evaluation,
        feature_seconds=feature_seconds,
        train_eval_seconds=0.0,
        total_seconds=time.perf_counter() - total_start,
    )


def _success_row(
    baseline_name: str,
    *,
    method_family: str,
    source: str,
    representation: str,
    readout: str,
    classifier: str,
    split_info: SplitInfo,
    feature_shape: Sequence[int],
    evaluation: Mapping[str, Any],
    feature_seconds: float,
    train_eval_seconds: float,
    total_seconds: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    test_metrics = dict(evaluation["test"])
    row = _base_row(
        baseline_name,
        method_family=method_family,
        source=source,
        representation=representation,
        readout=readout,
        classifier=classifier,
        split_info=split_info,
    )
    row.update(
        {
            "status": "success",
            "failure_reason": "",
            "feature_shape": "x".join(str(int(dim)) for dim in feature_shape),
            "best_val_threshold": evaluation["threshold"],
            "feature_seconds": float(feature_seconds),
            "train_eval_seconds": float(train_eval_seconds),
            "total_seconds": float(total_seconds),
        }
    )
    if extra:
        row.update(dict(extra))
    for metric in METRIC_NAMES:
        row[f"test_{metric}"] = test_metrics.get(metric, float("nan"))
    return row


def _failure_row(
    baseline_name: str,
    *,
    method_family: str,
    source: str,
    representation: str,
    readout: str,
    classifier: str,
    split_info: SplitInfo,
    failure_reason: str,
    total_seconds: float,
) -> dict[str, Any]:
    row = _base_row(
        baseline_name,
        method_family=method_family,
        source=source,
        representation=representation,
        readout=readout,
        classifier=classifier,
        split_info=split_info,
    )
    row.update(
        {
            "status": "failure",
            "failure_reason": str(failure_reason),
            "feature_shape": "",
            "best_val_threshold": float("nan"),
            "feature_seconds": float("nan"),
            "train_eval_seconds": float("nan"),
            "total_seconds": float(total_seconds),
            "best_epoch": "",
            "early_stopped": "",
            "converged": "",
            "selected_probe": "",
            "halp_layer_indices": "",
            "num_halp_candidates": "",
            "selection_metric": "",
        }
    )
    for metric in METRIC_NAMES:
        row[f"test_{metric}"] = float("nan")
    return row


def _base_row(
    baseline_name: str,
    *,
    method_family: str,
    source: str,
    representation: str,
    readout: str,
    classifier: str,
    split_info: SplitInfo,
) -> dict[str, Any]:
    counts = split_info.counts()
    return {
        "baseline_name": baseline_name,
        "method_family": method_family,
        "source": source,
        "representation": representation,
        "readout": readout,
        "classifier": classifier,
        "train_samples": counts["train_samples"],
        "validation_samples": counts["validation_samples"],
        "test_samples": counts["test_samples"],
        "train_pos": counts["train_pos"],
        "validation_pos": counts["validation_pos"],
        "test_pos": counts["test_pos"],
        "best_epoch": "",
        "early_stopped": "",
        "converged": "",
        "selected_probe": "",
        "halp_layer_indices": "",
        "num_halp_candidates": "",
        "selection_metric": "",
    }


def _as_labels(labels: Sequence[int] | np.ndarray, *, expected_size: int) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("labels must be a 1D vector")
    if array.shape[0] != expected_size:
        raise ValueError("labels length must match entries")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")
    return array


def _as_2d_features(features: np.ndarray, *, expected_size: int) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D array")
    if matrix.shape[0] != expected_size:
        raise ValueError("features sample count must match labels")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must have non-empty sample and feature dimensions")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain NaN or Inf")
    return matrix


def _require_two_classes(labels: np.ndarray, *, name: str) -> None:
    if np.unique(labels).shape[0] < 2:
        raise ValueError(f"{name} must contain both classes")


def _best_by_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_value = -float("inf")
    for row in rows:
        value = _metric_value(row.get(metric))
        if value > best_value:
            best_value = value
            best = dict(row)
    return best


def _metric_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -float("inf")
    if not math.isfinite(number):
        return -float("inf")
    return number


def _fmt(value: Any) -> str:
    number = _metric_value(value)
    if number == -float("inf"):
        return "undefined"
    return f"{number:.6f}"
