#!/usr/bin/env python3
"""Run Stage A representation-space diagnostics from Stage 0 cache tensors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(REPO_SRC), str(SCRIPT_DIR)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from stage_a_build_family_splits import build_family_splits
from stage_a_preflight import run_preflight
from mind.trajectory.stage_a_metrics import bootstrap_binary_metrics, binary_diagnostic_metrics
from mind.trajectory.stage_a_population import (
    PopulationClass,
    classify_entry,
    stream_stage0_cache_entries,
    summarize_population,
)
from mind.trajectory.stage_a_readouts import (
    compute_knn_scores,
    train_logistic_diagnostic,
    train_lstm_diagnostic,
)
from mind.trajectory.stage_a_representations import (
    DEFAULT_STAGE_A_SEED,
    build_lstm_trajectory,
    build_representation_matrix,
    set_deterministic_seed,
    shuffled_layer_permutation,
)


VARIANTS = (
    "Raw-Static",
    "Sphere-Static",
    "Norm-Static",
    "Raw-Traj-MeanPool",
    "Sphere-Traj-MeanPool",
    "Norm-Traj",
    "Sphere-Traj-LSTM-v0",
    "Sphere-Traj-Shuffled-LSTM",
)
NON_LSTM_VARIANTS = VARIANTS[:6]
LSTM_VARIANTS = VARIANTS[6:]
READOUTS = ("Diag-Classifier", "Diag-KNN")
SPLITS = ("encoder_train", "bank", "cal", "test")
PRIMARY_MODELS = ("qwen3-vl-8b", "internvl3.5-8b")
REQUIRED_STAGE_A_SUBSETS = ("popular", "random", "adversarial")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageA"))
    parser.add_argument("--models", nargs="+", default=["qwen3-vl-8b"], choices=PRIMARY_MODELS)
    parser.add_argument("--subsets", nargs="+", default=["popular", "random", "adversarial"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_STAGE_A_SEED)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--include-internvl-after-qwen-pass", action="store_true")
    parser.add_argument("--limit-per-subset", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_deterministic_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    requested_models = _ordered_requested_models(args.models)

    preflight = run_preflight(stage0_root=args.stage0_root, output_root=args.output_root)
    if preflight["status"] != "passed":
        _write_stage_a_summary(
            args.output_root,
            preflight,
            {},
            [],
            "fail",
            dry_run=args.dry_run,
            requested_models=requested_models,
            requested_subsets=args.subsets,
            limit_per_subset=args.limit_per_subset,
            skip_lstm=args.skip_lstm,
        )
        print("Stage A preflight failed; experiments were not started", file=sys.stderr)
        return 2

    split_manifest = build_family_splits(
        stage0_root=args.stage0_root,
        output_root=args.output_root,
        dataset_names=("pope",),
        seed=args.seed,
    )
    _write_population_audits(
        stage0_root=args.stage0_root,
        output_root=args.output_root,
        subsets=args.subsets,
        split_manifest=split_manifest,
        limit_per_subset=args.limit_per_subset,
    )

    if args.dry_run:
        summary = _write_stage_a_summary(
            args.output_root,
            preflight,
            {},
            [],
            "dry_run",
            dry_run=True,
            requested_models=requested_models,
            requested_subsets=args.subsets,
            limit_per_subset=args.limit_per_subset,
            skip_lstm=args.skip_lstm,
        )
        print(f"Stage A dry run complete: {summary}")
        return 0

    completed_models: list[str] = []
    skipped_models: dict[str, dict[str, object]] = {}
    gates: dict[str, dict[str, object]] = {}
    qwen_decision: str | None = None
    for model_name in requested_models:
        if model_name == "internvl3.5-8b" and qwen_decision == "fail" and not _only_internvl_requested(args.models):
            reason = "qwen3-vl-8b Stage A decision was fail"
            _write_not_run(
                args.output_root,
                model_name,
                reason=reason,
                skip_type="qwen_gate",
            )
            skipped_models[model_name] = {"reason": reason, "skip_type": "qwen_gate"}
            continue
        if model_name == "internvl3.5-8b" and not args.include_internvl_after_qwen_pass and "qwen3-vl-8b" in requested_models:
            reason = "--include-internvl-after-qwen-pass was not set"
            _write_not_run(
                args.output_root,
                model_name,
                reason=reason,
                skip_type="deferred_until_qwen_pass",
            )
            skipped_models[model_name] = {"reason": reason, "skip_type": "deferred_until_qwen_pass"}
            continue
        model_result = run_model_stage_a(
            model_name=model_name,
            stage0_root=args.stage0_root,
            output_root=args.output_root,
            subsets=args.subsets,
            split_manifest=split_manifest,
            device=args.device,
            seed=args.seed,
            bootstrap=args.bootstrap,
            lstm_epochs=args.lstm_epochs,
            knn_k=args.knn_k,
            limit_per_subset=args.limit_per_subset,
            skip_lstm=args.skip_lstm,
            save_embeddings=args.save_embeddings,
        )
        gates[model_name] = model_result
        completed_models.append(model_name)
        if model_name == "qwen3-vl-8b":
            qwen_decision = str(model_result["overall_decision"])

    overall = _overall_stage_a_decision(gates)
    _write_stage_a_summary(
        args.output_root,
        preflight,
        gates,
        completed_models,
        overall,
        dry_run=False,
        requested_models=requested_models,
        requested_subsets=args.subsets,
        limit_per_subset=args.limit_per_subset,
        skip_lstm=args.skip_lstm,
        skipped_models=skipped_models,
    )
    print(f"Stage A complete overall_decision={overall}")
    return 0 if overall != "fail" or completed_models else 2


def run_model_stage_a(
    *,
    model_name: str,
    stage0_root: Path,
    output_root: Path,
    subsets: Sequence[str],
    split_manifest: Mapping[str, object],
    device: str,
    seed: int,
    bootstrap: int,
    lstm_epochs: int,
    knn_k: int,
    limit_per_subset: int | None,
    skip_lstm: bool,
    save_embeddings: bool,
) -> dict[str, object]:
    entries = _load_model_entries(
        stage0_root=stage0_root,
        model_name=model_name,
        subsets=subsets,
        split_manifest=split_manifest,
        limit_per_subset=limit_per_subset,
    )
    primary_entries = [
        row for row in entries
        if classify_entry(row) in {PopulationClass.CORRECT, PopulationClass.HARD_HALLUCINATION}
    ]
    excluded_counts = _excluded_counts(entries)
    if sum(1 for row in primary_entries if classify_entry(row) == PopulationClass.HARD_HALLUCINATION) < 50:
        raise ValueError(f"{model_name} has fewer than 50 hard hallucination samples in POPE primary population")
    if not primary_entries:
        raise ValueError(f"{model_name} has no Stage A primary population entries")

    report_dir = output_root / "reports" / model_name
    report_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    score_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for variant in NON_LSTM_VARIANTS:
        _run_non_lstm_variant(
            variant,
            primary_entries,
            metric_rows=metric_rows,
            score_cache=score_cache,
            excluded_counts=excluded_counts,
            knn_device=device,
            seed=seed,
            bootstrap=bootstrap,
            knn_k=knn_k,
        )

    if not skip_lstm:
        for variant in LSTM_VARIANTS:
            _run_lstm_variant(
                variant,
                primary_entries,
                metric_rows=metric_rows,
                score_cache=score_cache,
                excluded_counts=excluded_counts,
                knn_device=device,
                device=device,
                seed=seed,
                bootstrap=bootstrap,
                lstm_epochs=lstm_epochs,
                knn_k=knn_k,
            )

    metrics_path = report_dir / "stageA_metrics.csv"
    _write_csv(metrics_path, metric_rows)
    if save_embeddings:
        (report_dir / "embedding_note.txt").write_text(
            "Stage A save-embeddings was requested, but embeddings are intentionally not persisted by default.\n",
            encoding="utf-8",
        )

    gate = _gate_model(metric_rows, skip_lstm=skip_lstm)
    gate["model_name"] = model_name
    gate["metrics_path"] = str(metrics_path)
    gate["num_primary_population"] = len(primary_entries)
    gate["population"] = summarize_population(entries)
    gate_path = report_dir / "stageA_gate.json"
    _write_json(gate_path, gate)
    _write_model_summary(report_dir / "stageA_summary.md", gate, metrics_path)
    return gate


def _run_non_lstm_variant(
    variant: str,
    entries: Sequence[Mapping[str, object]],
    *,
    metric_rows: list[dict[str, object]],
    score_cache: dict[tuple[str, str], dict[str, np.ndarray]],
    excluded_counts: Mapping[tuple[str, str], Mapping[str, int]],
    knn_device: str,
    seed: int,
    bootstrap: int,
    knn_k: int,
) -> None:
    features = build_representation_matrix(entries, variant).values
    labels = _labels(entries)
    splits = _splits(entries)
    subsets = _subsets(entries)
    train_mask = splits == "encoder_train"
    if np.unique(labels[train_mask]).size < 2:
        raise ValueError(f"{variant} encoder_train is missing one class")
    classifier = train_logistic_diagnostic(
        features[train_mask],
        labels[train_mask],
        seed=seed,
        max_iter=1000,
    )
    classifier_scores = classifier.model.predict_proba(features)[:, 1].astype(np.float32)
    _append_metric_rows(
        metric_rows,
        entries,
        labels,
        splits,
        subsets,
        variant=variant,
        readout="Diag-Classifier",
        scores=classifier_scores,
        bootstrap=bootstrap,
        seed=seed,
        num_bank_correct=_bank_correct_count(entries),
        excluded_counts=excluded_counts,
    )
    score_cache[(variant, "Diag-Classifier")] = {"scores": classifier_scores, "labels": labels}

    knn_features = _knn_features(variant, features, train_mask)
    bank_mask = (splits == "bank") & _correct_mask(entries)
    metric = _knn_metric_for_variant(variant)
    knn_scores = _compute_stage_a_knn_scores(
        knn_features[bank_mask],
        knn_features,
        k=knn_k,
        metric=metric,
        device=knn_device,
    )
    _append_metric_rows(
        metric_rows,
        entries,
        labels,
        splits,
        subsets,
        variant=variant,
        readout="Diag-KNN",
        scores=knn_scores,
        bootstrap=bootstrap,
        seed=seed,
        num_bank_correct=int(bank_mask.sum()),
        excluded_counts=excluded_counts,
    )
    score_cache[(variant, "Diag-KNN")] = {"scores": knn_scores, "labels": labels}


def _run_lstm_variant(
    variant: str,
    entries: Sequence[Mapping[str, object]],
    *,
    metric_rows: list[dict[str, object]],
    score_cache: dict[tuple[str, str], dict[str, np.ndarray]],
    excluded_counts: Mapping[tuple[str, str], Mapping[str, int]],
    knn_device: str,
    device: str,
    seed: int,
    bootstrap: int,
    lstm_epochs: int,
    knn_k: int,
) -> None:
    labels = _labels(entries)
    splits = _splits(entries)
    subsets = _subsets(entries)
    first = entries[0]["layer_vectors"]
    num_layers = int(first.shape[0])  # type: ignore[attr-defined]
    hidden_dim = int(first.shape[1])  # type: ignore[attr-defined]
    layer_order = None
    if variant == "Sphere-Traj-Shuffled-LSTM":
        layer_order = shuffled_layer_permutation(num_layers, seed=seed)
    trajectories = np.stack(
        [build_lstm_trajectory(row, layer_order=layer_order) for row in entries],
        axis=0,
    ).astype(np.float32, copy=False)
    train_mask = splits == "encoder_train"
    if np.unique(labels[train_mask]).size < 2:
        raise ValueError(f"{variant} encoder_train is missing one class")
    result = train_lstm_diagnostic(
        trajectories[train_mask],
        labels[train_mask],
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        epochs=lstm_epochs,
        batch_size=128,
        device=device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu",
        seed=seed,
        patience=3,
    )
    lstm_device = next(result.model.parameters()).device
    scores, embeddings = _score_lstm_in_chunks(result.model, trajectories, device=lstm_device)
    _append_metric_rows(
        metric_rows,
        entries,
        labels,
        splits,
        subsets,
        variant=variant,
        readout="Diag-Classifier",
        scores=scores,
        bootstrap=bootstrap,
        seed=seed,
        num_bank_correct=_bank_correct_count(entries),
        excluded_counts=excluded_counts,
        extra={"layer_permutation": json.dumps(layer_order) if layer_order is not None else ""},
    )
    score_cache[(variant, "Diag-Classifier")] = {"scores": scores, "labels": labels}

    bank_mask = (splits == "bank") & _correct_mask(entries)
    embeddings = _l2_normalize(embeddings)
    knn_scores = _compute_stage_a_knn_scores(
        embeddings[bank_mask],
        embeddings,
        k=knn_k,
        metric="angular",
        device=knn_device,
    )
    _append_metric_rows(
        metric_rows,
        entries,
        labels,
        splits,
        subsets,
        variant=variant,
        readout="Diag-KNN",
        scores=knn_scores,
        bootstrap=bootstrap,
        seed=seed,
        num_bank_correct=int(bank_mask.sum()),
        excluded_counts=excluded_counts,
        extra={"layer_permutation": json.dumps(layer_order) if layer_order is not None else ""},
    )
    score_cache[(variant, "Diag-KNN")] = {"scores": knn_scores, "labels": labels}


def _append_metric_rows(
    rows: list[dict[str, object]],
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    splits: np.ndarray,
    subsets: np.ndarray,
    *,
    variant: str,
    readout: str,
    scores: np.ndarray,
    bootstrap: int,
    seed: int,
    num_bank_correct: int,
    excluded_counts: Mapping[tuple[str, str], Mapping[str, int]],
    extra: Mapping[str, object] | None = None,
) -> None:
    extra = dict(extra or {})
    for split_name in SPLITS:
        mask = splits == split_name
        rows.append(
            _metric_row(
                entries,
                labels,
                scores,
                mask,
                variant=variant,
                readout=readout,
                eval_split=split_name,
                eval_scope="pooled",
                bootstrap=bootstrap,
                seed=seed,
                num_bank_correct=num_bank_correct,
                excluded_counts=excluded_counts,
                extra=extra,
            )
        )
    for subset in sorted(set(subsets.tolist())):
        mask = (splits == "test") & (subsets == subset)
        rows.append(
            _metric_row(
                entries,
                labels,
                scores,
                mask,
                variant=variant,
                readout=readout,
                eval_split="test",
                eval_scope=str(subset),
                bootstrap=bootstrap,
                seed=seed,
                num_bank_correct=num_bank_correct,
                excluded_counts=excluded_counts,
                extra=extra,
            )
        )


def _metric_row(
    entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    variant: str,
    readout: str,
    eval_split: str,
    eval_scope: str,
    bootstrap: int,
    seed: int,
    num_bank_correct: int,
    excluded_counts: Mapping[tuple[str, str], Mapping[str, int]],
    extra: Mapping[str, object],
) -> dict[str, object]:
    split_entries = [entry for entry, keep in zip(entries, mask, strict=True) if bool(keep)]
    y = labels[mask]
    split_scores = scores[mask]
    undefined_reason = ""
    if y.size == 0:
        undefined_reason = "no samples in evaluation scope"
    elif np.unique(y).size < 2:
        undefined_reason = "one class present in evaluation scope"
    if undefined_reason:
        metrics = _undefined_metrics()
        intervals = _undefined_intervals(metrics)
    else:
        metrics = binary_diagnostic_metrics(y, split_scores)
        intervals = bootstrap_binary_metrics(y, split_scores, num_bootstrap=bootstrap, seed=seed)
    row: dict[str, object] = {
        "variant": variant,
        "readout": readout,
        "eval_split": eval_split,
        "eval_scope": eval_scope,
        "metric_status": "undefined" if undefined_reason else "passed",
        "failure_reason": undefined_reason,
        "pr_auc": metrics["pr_auc"],
        "pr_auc_ci_low": intervals["pr_auc"].lower,
        "pr_auc_ci_high": intervals["pr_auc"].upper,
        "roc_auc": metrics["roc_auc"],
        "roc_auc_ci_low": intervals["roc_auc"].lower,
        "roc_auc_ci_high": intervals["roc_auc"].upper,
        "tpr_at_1pct_fpr": metrics["tpr_at_1pct_fpr"],
        "fpr_at_95pct_tpr": metrics["fpr_at_95pct_tpr"],
        "average_precision": metrics["average_precision"],
        "num_test": int(y.size),
        "num_test_correct": int(np.sum(y == 0)),
        "num_test_hard_hallucination": int(np.sum(y == 1)),
        "num_bank_correct": int(num_bank_correct),
        "num_encoder_train": int(sum(1 for entry in entries if entry["stage_a_split"] == "encoder_train")),
        "num_encoder_train_hallucination": int(
            sum(
                1
                for entry in entries
                if entry["stage_a_split"] == "encoder_train"
                and classify_entry(entry) == PopulationClass.HARD_HALLUCINATION
            )
        ),
        "num_excluded_false_negative": int(
            excluded_counts.get((eval_split, eval_scope), {}).get("false_negative", 0)
        ),
        "num_excluded_parsed_none": int(
            excluded_counts.get((eval_split, eval_scope), {}).get("parsed_none", 0)
        ),
    }
    row.update(extra)
    return row


def _load_model_entries(
    *,
    stage0_root: Path,
    model_name: str,
    subsets: Sequence[str],
    split_manifest: Mapping[str, object],
    limit_per_subset: int | None,
) -> list[dict[str, object]]:
    assignment_by_key = {
        (
            str(row["model_name"]),
            str(row["dataset_name"]),
            str(row["subset"]),
            str(row["sample_id"]),
        ): str(row["split"])
        for row in split_manifest.get("assignments", [])
        if isinstance(row, Mapping)
    }
    subset_filter = set(subsets)
    subset_counts: Counter[str] = Counter()
    entries: list[dict[str, object]] = []
    for row in stream_stage0_cache_entries(
        stage0_root,
        dataset_names=("pope",),
        model_names=(model_name,),
    ):
        subset = str(row.get("subset", ""))
        if subset not in subset_filter:
            continue
        if limit_per_subset is not None and subset_counts[subset] >= limit_per_subset:
            continue
        split = assignment_by_key.get((model_name, "pope", subset, str(row.get("sample_id", ""))))
        if split is None:
            raise ValueError(f"missing Stage A split for {model_name}/pope/{subset}/{row.get('sample_id')}")
        row["stage_a_split"] = split
        entries.append(row)
        subset_counts[subset] += 1
    return entries


def _write_population_audits(
    *,
    stage0_root: Path,
    output_root: Path,
    subsets: Sequence[str],
    split_manifest: Mapping[str, object],
    limit_per_subset: int | None,
) -> None:
    assignment_by_key = {
        (
            str(row["model_name"]),
            str(row["dataset_name"]),
            str(row["subset"]),
            str(row["sample_id"]),
        ): str(row["split"])
        for row in split_manifest.get("assignments", [])
        if isinstance(row, Mapping)
    }
    subset_filter = set(subsets)
    rows_by_subset: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    rows_by_split: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    subset_counts: Counter[tuple[str, str]] = Counter()
    for row in stream_stage0_cache_entries(
        stage0_root,
        dataset_names=("pope",),
        include_tensors=False,
    ):
        model_name = str(row.get("model_name", ""))
        subset = str(row.get("subset", ""))
        if subset not in subset_filter:
            continue
        limit_key = (model_name, subset)
        if limit_per_subset is not None and subset_counts[limit_key] >= limit_per_subset:
            continue
        split = assignment_by_key.get((model_name, "pope", subset, str(row.get("sample_id", ""))))
        row["stage_a_split"] = split or ""
        key = (model_name, "pope", subset)
        rows_by_subset[key].append(row)
        rows_by_split[(model_name, "pope", subset, str(row["stage_a_split"]))].append(row)
        subset_counts[limit_key] += 1

    balance_rows = []
    for key, values in sorted(rows_by_subset.items()):
        summary = summarize_population(values)
        balance_rows.append(_population_row(key, values, summary))
    audit_rows = []
    for key, values in sorted(rows_by_split.items()):
        model_name, dataset_name, subset, split = key
        summary = summarize_population(values)
        audit_rows.append({"split": split, **_population_row((model_name, dataset_name, subset), values, summary)})

    audit_dir = output_root / "audit"
    _write_csv(audit_dir / "cache_label_balance.csv", balance_rows)
    _write_csv(audit_dir / "stageA_population_audit.csv", audit_rows)


def _population_row(
    key: tuple[str, str, str],
    values: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> dict[str, object]:
    model_name, dataset_name, subset = key
    num_entries = int(summary["num_entries"])
    hallucination_count = int(summary["num_hard_hallucination"])
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "subset": subset,
        "num_entries": num_entries,
        "num_gt_yes": sum(1 for row in values if _binary_value(row.get("label")) == 1),
        "num_gt_no": sum(1 for row in values if _binary_value(row.get("label")) == 0),
        "num_parsed_yes": sum(1 for row in values if _binary_value(row.get("parsed_answer")) == 1),
        "num_parsed_no": sum(1 for row in values if _binary_value(row.get("parsed_answer")) == 0),
        "num_parsed_none": summary["num_parsed_none"],
        "num_correct": summary["num_correct"],
        "num_hard_hallucination": hallucination_count,
        "num_false_negative_error": summary["num_false_negative_error"],
        "num_primary_population": summary["num_primary_population"],
        "hallucination_rate_in_primary_population": summary["hallucination_rate_in_primary_population"],
        "low_positive_count": hallucination_count < 20,
    }


def _excluded_counts(entries: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"false_negative": 0, "parsed_none": 0})
    for row in entries:
        split = str(row["stage_a_split"])
        subset = str(row.get("subset", ""))
        population_class = classify_entry(row)
        for scope in ("pooled", subset):
            if population_class == PopulationClass.FALSE_NEGATIVE_ERROR:
                counts[(split, scope)]["false_negative"] += 1
            elif population_class == PopulationClass.PARSED_NONE:
                counts[(split, scope)]["parsed_none"] += 1
    return dict(counts)


def _binary_value(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in {0, 1} else int(value > 0)
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "present"}:
        return 1
    if text in {"no", "n", "0", "false", "absent"}:
        return 0
    return None


def _gate_model(metric_rows: Sequence[Mapping[str, object]], *, skip_lstm: bool) -> dict[str, object]:
    sphere_gate = _sphere_gate(metric_rows)
    trajectory_gate = _trajectory_gate(metric_rows)
    layer_order_gate = {"status": "mixed", "reason": "LSTM variants skipped"}
    if not skip_lstm:
        layer_order_gate = _layer_order_gate(metric_rows)
    norm_gate = _norm_gate(metric_rows)
    severe_issue = False
    if sphere_gate["status"] == "pass" and trajectory_gate["status"] == "pass" and not severe_issue:
        overall = "pass"
    elif (
        {sphere_gate["status"], trajectory_gate["status"]} <= {"pass", "mixed"}
        and "pass" in {sphere_gate["status"], trajectory_gate["status"]}
        and not severe_issue
    ):
        overall = "mixed_positive"
    else:
        overall = "fail"
    conclusion_notes = []
    if layer_order_gate["status"] in {"mixed", "fail"}:
        conclusion_notes.append("multi-layer aggregation may be useful, but sequence order is not yet proven.")
    if norm_gate["status"] == "strong":
        conclusion_notes.append("magnitude carries useful signal and cannot be dismissed as pure noise.")
    return {
        "sphere_gate": sphere_gate,
        "trajectory_gate": trajectory_gate,
        "layer_order_gate": layer_order_gate,
        "norm_diagnostic_gate": norm_gate,
        "overall_decision": overall,
        "conclusion_notes": conclusion_notes,
    }


def _sphere_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    comparisons = [
        ("Sphere-Static", "Raw-Static"),
        ("Sphere-Traj-MeanPool", "Raw-Traj-MeanPool"),
    ]
    return _pair_gate(rows, comparisons, min_pr_delta=0.01, max_roc_drop=0.02)


def _trajectory_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    comparisons = [
        ("Sphere-Traj-MeanPool", "Sphere-Static"),
        ("Raw-Traj-MeanPool", "Raw-Static"),
    ]
    return _pair_gate(rows, comparisons, min_pr_delta=0.01, max_roc_drop=0.05)


def _pair_gate(
    rows: Sequence[Mapping[str, object]],
    comparisons: Sequence[tuple[str, str]],
    *,
    min_pr_delta: float,
    max_roc_drop: float,
) -> dict[str, object]:
    successes: list[dict[str, object]] = []
    worse_count = 0
    total = 0
    for better, base in comparisons:
        for readout in READOUTS:
            better_row = _find_metric(rows, better, readout, "test", "pooled")
            base_row = _find_metric(rows, base, readout, "test", "pooled")
            if better_row is None or base_row is None:
                continue
            total += 1
            pr_delta = float(better_row["pr_auc"]) - float(base_row["pr_auc"])
            roc_delta = float(better_row["roc_auc"]) - float(base_row["roc_auc"])
            if pr_delta >= min_pr_delta and roc_delta >= -max_roc_drop:
                successes.append(
                    {
                        "variant": better,
                        "baseline": base,
                        "readout": readout,
                        "pr_auc_delta": pr_delta,
                        "roc_auc_delta": roc_delta,
                    }
                )
            if pr_delta < 0.0 and roc_delta < 0.0:
                worse_count += 1
    if successes:
        status = "pass" if len({item["readout"] for item in successes}) > 1 else "mixed"
    elif total > 0 and worse_count == total:
        status = "fail"
    else:
        status = "mixed"
    return {"status": status, "successes": successes, "total_comparisons": total}


def _layer_order_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ordered = _find_metric(rows, "Sphere-Traj-LSTM-v0", "Diag-Classifier", "test", "pooled")
    shuffled = _find_metric(rows, "Sphere-Traj-Shuffled-LSTM", "Diag-Classifier", "test", "pooled")
    if ordered is None or shuffled is None:
        return {"status": "mixed", "reason": "missing LSTM metric rows"}
    pr_delta = float(ordered["pr_auc"]) - float(shuffled["pr_auc"])
    roc_delta = float(ordered["roc_auc"]) - float(shuffled["roc_auc"])
    intervals_overlap = not (
        float(ordered["pr_auc_ci_low"]) > float(shuffled["pr_auc_ci_high"])
        or float(shuffled["pr_auc_ci_low"]) > float(ordered["pr_auc_ci_high"])
    )
    if pr_delta >= 0.01 and roc_delta >= 0.005:
        status = "pass"
    elif intervals_overlap:
        status = "mixed"
    elif pr_delta < 0.0 and roc_delta < 0.0:
        status = "fail"
    else:
        status = "mixed"
    return {"status": status, "pr_auc_delta": pr_delta, "roc_auc_delta": roc_delta}


def _norm_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    test_rows = [
        row for row in rows
        if row["eval_split"] == "test" and row["eval_scope"] == "pooled"
    ]
    norm_best = max(
        (float(row["pr_auc"]) for row in test_rows if str(row["variant"]).startswith("Norm-")),
        default=float("nan"),
    )
    non_norm_best = max(
        (float(row["pr_auc"]) for row in test_rows if not str(row["variant"]).startswith("Norm-")),
        default=float("nan"),
    )
    strong = np.isfinite(norm_best) and np.isfinite(non_norm_best) and norm_best >= 0.9 * non_norm_best
    return {
        "status": "strong" if strong else "weak",
        "best_norm_pr_auc": norm_best,
        "best_non_norm_pr_auc": non_norm_best,
    }


def _find_metric(
    rows: Sequence[Mapping[str, object]],
    variant: str,
    readout: str,
    eval_split: str,
    eval_scope: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if (
            row.get("variant") == variant
            and row.get("readout") == readout
            and row.get("eval_split") == eval_split
            and row.get("eval_scope") == eval_scope
        ):
            return row
    return None


def _knn_features(variant: str, features: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    if variant.startswith("Norm-"):
        scaler = StandardScaler()
        scaler.fit(features[train_mask])
        return scaler.transform(features).astype(np.float32)
    return features


def _knn_metric_for_variant(variant: str) -> str:
    if variant.startswith("Sphere-"):
        return "angular"
    return "euclidean"


def _undefined_metrics() -> dict[str, float]:
    return {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
        "fpr_at_95pct_tpr": float("nan"),
        "average_precision": float("nan"),
    }


def _undefined_intervals(metrics: Mapping[str, float]) -> dict[str, object]:
    class _Interval:
        def __init__(self, value: float) -> None:
            self.lower = value
            self.upper = value

    return {
        name: _Interval(value)
        for name, value in metrics.items()
    }


def _compute_stage_a_knn_scores(
    bank: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
    metric: str,
    device: str,
) -> np.ndarray:
    backend = "torch" if _knn_device_available(device) else "numpy"
    return compute_knn_scores(
        bank,
        query,
        k=k,
        metric=metric,
        backend=backend,
        device=device if backend == "torch" else None,
        chunk_size=4096,
    )


def _knn_device_available(device: str) -> bool:
    if not device.startswith("cuda"):
        return True
    return torch.cuda.is_available()


def _score_lstm_in_chunks(
    model: torch.nn.Module,
    trajectories: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, trajectories.shape[0], batch_size):
            batch = torch.from_numpy(trajectories[start : start + batch_size]).to(device)
            emb, logits = model.embed_and_score(batch)  # type: ignore[attr-defined]
            scores.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
            embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0), np.concatenate(embeddings, axis=0)


def _labels(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [1 if classify_entry(row) == PopulationClass.HARD_HALLUCINATION else 0 for row in entries],
        dtype=np.int64,
    )


def _splits(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([str(row["stage_a_split"]) for row in entries])


def _subsets(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([str(row["subset"]) for row in entries])


def _correct_mask(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([classify_entry(row) == PopulationClass.CORRECT for row in entries], dtype=bool)


def _bank_correct_count(entries: Sequence[Mapping[str, object]]) -> int:
    return int(
        sum(
            1
            for row in entries
            if row["stage_a_split"] == "bank" and classify_entry(row) == PopulationClass.CORRECT
        )
    )


def _l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, eps)).astype(np.float32)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_not_run(output_root: Path, model_name: str, *, reason: str, skip_type: str) -> None:
    _write_json(
        output_root / "reports" / model_name / "not_run.json",
        {
            "model_name": model_name,
            "status": "not_run",
            "reason": reason,
            "skip_type": skip_type,
        },
    )


def _write_model_summary(path: Path, gate: Mapping[str, object], metrics_path: Path) -> None:
    notes = gate.get("conclusion_notes", [])
    note_text = "\n".join(f"- {note}" for note in notes) if notes else "- No extra gate notes."
    path.write_text(
        "\n".join(
            [
                f"# Stage A Summary: {gate['model_name']}",
                "",
                f"- Overall decision: {gate['overall_decision']}",
                f"- Metrics: `{metrics_path}`",
                f"- Sphere gate: {gate['sphere_gate']['status']}",  # type: ignore[index]
                f"- Trajectory gate: {gate['trajectory_gate']['status']}",  # type: ignore[index]
                f"- Layer-order gate: {gate['layer_order_gate']['status']}",  # type: ignore[index]
                f"- Norm diagnostic: {gate['norm_diagnostic_gate']['status']}",  # type: ignore[index]
                "",
                "Stage A tests representation hypotheses only. It does not validate the final MIND detector.",
                "",
                note_text,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_stage_a_summary(
    output_root: Path,
    preflight: Mapping[str, object],
    gates: Mapping[str, Mapping[str, object]],
    completed_models: Sequence[str],
    overall: str,
    *,
    dry_run: bool,
    requested_models: Sequence[str] | None = None,
    requested_subsets: Sequence[str] | None = None,
    limit_per_subset: int | None = None,
    skip_lstm: bool = False,
    skipped_models: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    path = output_root / "manifests" / "stageA_summary.json"
    current_completed_models = list(completed_models)
    current_requested_models = list(
        current_completed_models if requested_models is None else requested_models
    )
    current_skipped_models = {
        str(model_name): dict(details)
        for model_name, details in (skipped_models or {}).items()
    }
    run_scope = _stage_a_run_scope(
        stage0_acceptance=preflight.get("status"),
        requested_models=current_requested_models,
        completed_models=current_completed_models,
        skipped_models=current_skipped_models,
        requested_subsets=REQUIRED_STAGE_A_SUBSETS if requested_subsets is None else requested_subsets,
        dry_run=dry_run,
        limit_per_subset=limit_per_subset,
        skip_lstm=skip_lstm,
    )
    if preflight.get("status") != "passed":
        summary_status = "failed"
    elif dry_run:
        summary_status = "dry_run"
    elif run_scope["full_stage_a_run"]:
        summary_status = "completed"
    else:
        summary_status = "partial"
    model_decisions = {
        model_name: gate.get("overall_decision")
        for model_name, gate in gates.items()
    }
    merged_overall = _decision_from_values([str(value) for value in model_decisions.values()])
    if not model_decisions:
        merged_overall = overall
    payload = {
        "stage": "stageA",
        "status": summary_status,
        "stage0_acceptance": preflight.get("status"),
        "completed_models": current_completed_models,
        "model_decisions": model_decisions,
        "overall_decision": merged_overall,
        "stage_b_started": False,
        "dry_run": dry_run,
        "full_stage_a_run": run_scope["full_stage_a_run"],
        "run_scope": run_scope,
    }
    _write_json(path, payload)
    return str(path)


def _stage_a_run_scope(
    *,
    stage0_acceptance: object,
    requested_models: Sequence[str],
    completed_models: Sequence[str],
    skipped_models: Mapping[str, Mapping[str, object]],
    requested_subsets: Sequence[str],
    dry_run: bool,
    limit_per_subset: int | None,
    skip_lstm: bool,
) -> dict[str, object]:
    reasons: list[str] = []
    subset_values = list(requested_subsets)
    skipped = {str(model_name): dict(details) for model_name, details in skipped_models.items()}

    if dry_run:
        reasons.append("dry_run")
    if stage0_acceptance != "passed":
        reasons.append("stage0_acceptance_not_passed")
    if limit_per_subset is not None:
        reasons.append("limit_per_subset")
    if skip_lstm:
        reasons.append("skip_lstm")
    if not _is_required_stage_a_subset_scope(subset_values):
        reasons.append("required_subsets")
    handled_models = set(completed_models)
    handled_models.update(
        model_name
        for model_name, details in skipped.items()
        if details.get("skip_type") == "qwen_gate"
    )
    if any(model_name not in handled_models for model_name in requested_models):
        reasons.append("requested_models_not_completed_or_qwen_gate_skipped")

    return {
        "full_stage_a_run": not reasons,
        "reasons": reasons,
        "requested_models": list(requested_models),
        "completed_models": list(completed_models),
        "skipped_models": skipped,
        "requested_subsets": subset_values,
        "required_subsets": list(REQUIRED_STAGE_A_SUBSETS),
        "limit_per_subset": limit_per_subset,
        "skip_lstm": skip_lstm,
        "dry_run": dry_run,
    }


def _is_required_stage_a_subset_scope(subsets: Sequence[str]) -> bool:
    return (
        len(subsets) == len(REQUIRED_STAGE_A_SUBSETS)
        and set(subsets) == set(REQUIRED_STAGE_A_SUBSETS)
    )


def _ordered_requested_models(models: Sequence[str]) -> list[str]:
    result = []
    for model in PRIMARY_MODELS:
        if model in models:
            result.append(model)
    for model in models:
        if model not in result:
            result.append(model)
    return result


def _only_internvl_requested(models: Sequence[str]) -> bool:
    return list(models) == ["internvl3.5-8b"]


def _overall_stage_a_decision(gates: Mapping[str, Mapping[str, object]]) -> str:
    if not gates:
        return "fail"
    return _decision_from_values([str(gate.get("overall_decision")) for gate in gates.values()])


def _decision_from_values(decisions: Sequence[str]) -> str:
    if "pass" in decisions:
        return "pass"
    if "mixed_positive" in decisions:
        return "mixed_positive"
    return "fail"


if __name__ == "__main__":
    raise SystemExit(main())
