#!/usr/bin/env python3
"""Run Stage B2 Proxy Anchor negative-budget diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) in sys.path:
    sys.path.remove(str(REPO_SRC))
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mind.trajectory.stage_a_closeout import (  # noqa: E402
    FAMILY_SUBSETS,
    build_closeout_family_split,
    write_csv_rows,
    write_split_manifest,
)
from mind.trajectory.stage_a_metrics import binary_diagnostic_metrics, bootstrap_binary_metrics  # noqa: E402
from mind.trajectory.stage_a_population import PopulationClass, classify_entry  # noqa: E402
from mind.trajectory.stage_a_readouts import train_logistic_diagnostic  # noqa: E402
from mind.trajectory.stage_a_representations import build_lstm_trajectory  # noqa: E402
from mind.trajectory.stage_b_glm_qc import (  # noqa: E402
    GLM_MODEL_ALIAS,
    apply_glm_qc_exclusion,
    scan_glm_cache_rows,
    write_glm_qc_reports,
)
from mind.trajectory.stage_b_manifest import stream_stage_b_full_cache_entries  # noqa: E402
from mind.trajectory.stage_b_objectives import STAGE_B_ENCODER_FAMILY  # noqa: E402
from mind.trajectory.stage_b_training import score_stage_b_lstm, train_stage_b_lstm  # noqa: E402
from mind.trajectory.stage_b_vmf import fit_single_vmf_prototype, score_single_vmf_prototype  # noqa: E402
from mind.trajectory.stage_b2_budget import (  # noqa: E402
    REQUIRED_STAGE_B2_RATIOS,
    REQUIRED_STAGE_B2_SEEDS,
    STAGE_B2_OBJECTIVE,
    subsample_stage_b2_training_indices,
    validate_stage_b2_budget_plan,
)
from mind.trajectory.stage_b2_knn import (  # noqa: E402
    compute_stage_b_knn_scores,
    generate_stage_b_k_candidates,
    select_stage_b2_knn_k,
)
from mind.trajectory.stage_b2_manifest import build_stage_b2_preflight, load_stage_b2_panel  # noqa: E402
from mind.trajectory.stage_b2_status import validate_stage_b2_summary  # noqa: E402


DATASET_OUTPUT_NAMES = {
    "repope": "repope_family_split_manifest.json",
    "pope": "pope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}
READOUTS = ("Diag-kNN-tuned", "Diag-Classifier", "Diag-vMF")
GLM_EXCLUSION_REASON = "GLM answer_text is not parseable under frozen yes/no population rules"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageB2"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["repope", "pope", "dash-b"])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(REQUIRED_STAGE_B2_RATIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REQUIRED_STAGE_B2_SEEDS))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--glm-qc-limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root: Path = args.output_root
    preflight_dir = output_root / "preflight"
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    for path in (preflight_dir, manifest_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    datasets = _validate_datasets(args.datasets)
    plan = validate_stage_b2_budget_plan(
        ratios=args.ratios,
        seeds=args.seeds,
        objective=STAGE_B2_OBJECTIVE,
        encoder_family=STAGE_B_ENCODER_FAMILY,
    )
    manifest = load_stage_b2_panel(args.full_cache_root)
    panel_models = [str(row["model_alias"]) for row in manifest.models]
    model_rows = _select_model_rows(manifest.models, requested=args.models)

    split_maps = _build_splits(
        manifest.models[0],
        full_cache_root=args.full_cache_root,
        output_root=output_root,
        datasets=datasets,
        seed=REQUIRED_STAGE_B2_SEEDS[0],
    )

    qc_rows = scan_glm_cache_rows(
        manifest,
        args.full_cache_root,
        dataset_families=tuple(dataset for dataset in datasets if dataset in FAMILY_SUBSETS),
        limit=args.glm_qc_limit,
    )
    write_glm_qc_reports(
        qc_rows,
        json_path=preflight_dir / "glm_answer_qc.json",
        markdown_path=preflight_dir / "glm_answer_qc.md",
    )
    excluded_models = dict(
        apply_glm_qc_exclusion(panel_models=panel_models, qc_rows=qc_rows)["excluded_models"]
    )
    if GLM_MODEL_ALIAS in panel_models:
        excluded_models[GLM_MODEL_ALIAS] = GLM_EXCLUSION_REASON

    preflight = build_stage_b2_preflight(
        manifest,
        excluded_models=excluded_models,
        split_ready=True,
        primary_dataset_available="repope" in datasets,
    )
    preflight["budget_plan"] = plan
    _write_json(preflight_dir / "stageB2_preflight.json", preflight)
    (preflight_dir / "stageB2_preflight.md").write_text(
        _render_preflight_markdown(preflight),
        encoding="utf-8",
    )

    metric_rows: list[dict[str, object]] = []
    selected_k_rows: list[dict[str, object]] = []
    vmf_summary_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    model_failures: dict[str, str] = {}

    included_model_rows = [
        row for row in model_rows if str(row.get("model_alias", "")) not in excluded_models
    ]
    device = _training_device(str(args.device))
    for index, model_row in enumerate(included_model_rows, start=1):
        model_alias = str(model_row["model_alias"])
        print(f"[{index}/{len(included_model_rows)}] Stage B2 model={model_alias}", flush=True)
        try:
            result = _run_model_stage_b2(
                model_row,
                full_cache_root=args.full_cache_root,
                datasets=datasets,
                split_maps=split_maps,
                ratios=tuple(float(value) for value in plan["ratios"]),
                seeds=tuple(int(value) for value in plan["seeds"]),
                bootstrap=int(args.bootstrap),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                device=device,
                limit_per_family=args.limit_per_family,
            )
        except Exception as exc:  # noqa: BLE001 - failures must be recorded.
            reason = f"{type(exc).__name__}: {exc}"
            model_failures[model_alias] = reason
            metric_rows.extend(_failed_metric_rows(model_alias, datasets, tuple(plan["ratios"]), tuple(plan["seeds"]), reason))
            print(f"  failed: {reason}", flush=True)
            continue
        metric_rows.extend(result["metric_rows"])
        selected_k_rows.extend(result["selected_k_rows"])
        vmf_summary_rows.extend(result["vmf_summary_rows"])
        training_rows.extend(result["training_rows"])
        skipped_rows.extend(result["skipped_rows"])

    metrics_path = report_dir / "stageB2_metrics_long.csv"
    selected_k_path = report_dir / "knn_selected_k.csv"
    vmf_summary_path = report_dir / "vmf_prototype_summary.csv"
    training_path = report_dir / "training_history.csv"
    skipped_path = report_dir / "budget_skips.csv"
    write_csv_rows(metrics_path, metric_rows)
    write_csv_rows(selected_k_path, selected_k_rows)
    write_csv_rows(vmf_summary_path, vmf_summary_rows)
    write_csv_rows(training_path, training_rows)
    write_csv_rows(skipped_path, skipped_rows)
    table_paths = _write_stage_b2_tables(report_dir, metric_rows)
    per_model_path = report_dir / "per_model_negative_budget_summary.csv"
    per_model_rows = _per_model_negative_budget_summary(
        metric_rows,
        panel_models=[str(row["model_alias"]) for row in model_rows],
        excluded_models=excluded_models,
        model_failures=model_failures,
    )
    write_csv_rows(per_model_path, per_model_rows)

    status_exclusions = dict(excluded_models)
    status_exclusions.update(model_failures)
    evaluated_models = sorted(
        {
            str(row["model_alias"])
            for row in metric_rows
            if row.get("metric_status") in {"passed", "undefined"}
        }
    )
    verdict = _negative_budget_verdict(metric_rows)
    summary = validate_stage_b2_summary(
        {
            "stage": "stage_b2",
            "stage_c_started": False,
            "detector_selected": False,
            "full_cache_manifest": str(manifest.path),
            "panel_models": [str(row["model_alias"]) for row in model_rows],
            "evaluated_models": evaluated_models,
            "excluded_models": status_exclusions,
            "objective": STAGE_B2_OBJECTIVE,
            "encoder_family": STAGE_B_ENCODER_FAMILY,
            "negative_budget_ratios": list(plan["ratios"]),
            "negative_budget_seeds": list(plan["seeds"]),
            "primary_decision_metric": "RePOPE pooled/test auto-tuned geodesic kNN PR-AUC",
            "verdict": verdict,
            "preflight_path": str(preflight_dir / "stageB2_preflight.json"),
            "split_manifest_paths": {
                dataset: str(manifest_dir / DATASET_OUTPUT_NAMES[dataset]) for dataset in datasets
            },
            "metrics_long_path": str(metrics_path),
            "repope_knn_table_path": str(table_paths["repope_knn"]),
            "repope_classifier_table_path": str(table_paths["repope_classifier"]),
            "repope_vmf_table_path": str(table_paths["repope_vmf"]),
            "pope_secondary_table_path": str(table_paths["pope_secondary"]),
            "dash_b_secondary_table_path": str(table_paths["dash_b_secondary"]),
            "selected_k_path": str(selected_k_path),
            "vmf_summary_path": str(vmf_summary_path),
            "per_model_negative_budget_summary_path": str(per_model_path),
            "budget_skips_path": str(skipped_path),
            "training_history_path": str(training_path),
            "stage_b2_scope": "Proxy Anchor negative-budget efficiency only; no final detector selected.",
        }
    )
    summary_path = report_dir / "STAGE_B2_SUMMARY.json"
    summary_md_path = report_dir / "STAGE_B2_SUMMARY.md"
    _write_json(summary_path, summary)
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    print(f"Stage B2 summary={summary_path} verdict={verdict['verdict']}", flush=True)
    return 0


def _run_model_stage_b2(
    model_row: Mapping[str, object],
    *,
    full_cache_root: Path,
    datasets: Sequence[str],
    split_maps: Mapping[str, Mapping[str, str]],
    ratios: Sequence[float],
    seeds: Sequence[int],
    bootstrap: int,
    epochs: int,
    batch_size: int,
    device: str,
    limit_per_family: int | None,
) -> dict[str, list[dict[str, object]]]:
    model_alias = str(model_row["model_alias"])
    family_cache: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        entries = _load_family_entries(
            model_row,
            full_cache_root=full_cache_root,
            dataset_family=dataset,
            split_map=split_maps[dataset],
            limit=limit_per_family,
        )
        primary = _primary_family_data(entries)
        family_cache[dataset] = {
            "entries": primary["entries"],
            "all_entries": entries,
            "labels": primary["labels"],
            "splits": primary["splits"],
            "trajectories": primary["trajectories"],
        }

    repope = family_cache["repope"]
    train_mask = np.asarray(repope["splits"]) == "encoder_train"
    train_labels_all = np.asarray(repope["labels"], dtype=np.int64)[train_mask]
    trajectories_train_all = np.asarray(repope["trajectories"], dtype=np.float32)[train_mask]
    num_layers = int(trajectories_train_all.shape[1])
    hidden_dim = int(trajectories_train_all.shape[2])
    available_correct = int(np.sum(train_labels_all == 0))
    available_hard = int(np.sum(train_labels_all == 1))

    metric_rows: list[dict[str, object]] = []
    selected_k_rows: list[dict[str, object]] = []
    vmf_summary_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for ratio in ratios:
        for seed in seeds:
            try:
                selected_train_indices = subsample_stage_b2_training_indices(
                    train_labels_all,
                    ratio=float(ratio),
                    seed=int(seed),
                )
            except ValueError as exc:
                skip = {
                    "model_alias": model_alias,
                    "negative_budget_ratio": float(ratio),
                    "negative_budget_seed": int(seed),
                    "budget_status": "skipped",
                    "budget_skip_reason": str(exc),
                    "num_encoder_train_correct": available_correct,
                    "num_encoder_train_hard_hallucination_available": available_hard,
                    "num_encoder_train_hard_hallucination_used": 0,
                }
                skipped_rows.append(skip)
                metric_rows.extend(_skipped_metric_rows(model_alias, datasets, float(ratio), int(seed), skip))
                continue

            train_labels = train_labels_all[selected_train_indices]
            train_trajectories = trajectories_train_all[selected_train_indices]
            used_hard = int(np.sum(train_labels == 1))
            trained = train_stage_b_lstm(
                train_trajectories,
                train_labels,
                objective=STAGE_B2_OBJECTIVE,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=int(seed),
                patience=5,
            )
            for history_row in trained.history:
                row = dict(history_row)
                row.update(
                    {
                        "model_alias": model_alias,
                        "objective": STAGE_B2_OBJECTIVE,
                        "negative_budget_ratio": float(ratio),
                        "negative_budget_seed": int(seed),
                    }
                )
                training_rows.append(row)

            family_embeddings: dict[str, np.ndarray] = {}
            for dataset, data in family_cache.items():
                embeddings, _ = score_stage_b_lstm(
                    trained.model,
                    np.asarray(data["trajectories"], dtype=np.float32),
                    batch_size=batch_size,
                )
                family_embeddings[dataset] = embeddings

            classifier_scores = _classifier_scores(
                train_embeddings=family_embeddings["repope"][train_mask],
                train_labels=np.asarray(repope["labels"], dtype=np.int64)[train_mask],
                selected_train_indices=selected_train_indices,
                family_embeddings=family_embeddings,
                seed=int(seed),
            )
            selected_k, _tuning_rows = _select_knn_for_budget(
                model_alias=model_alias,
                ratio=float(ratio),
                seed=int(seed),
                data=repope,
                embeddings=family_embeddings["repope"],
            )
            selected_k_rows.append(selected_k)

            for dataset, data in family_cache.items():
                rows = _evaluate_budget_family(
                    model_alias=model_alias,
                    dataset_family=dataset,
                    data=data,
                    embeddings=family_embeddings[dataset],
                    classifier_scores=np.asarray(classifier_scores[dataset], dtype=np.float32),
                    selected_k=int(selected_k["selected_k"]),
                    bootstrap=bootstrap,
                    seed=int(seed),
                    ratio=float(ratio),
                    available_correct=available_correct,
                    available_hard=available_hard,
                    used_hard=used_hard,
                )
                metric_rows.extend(rows)
                vmf_summary_rows.append(
                    _vmf_summary_row(
                        model_alias=model_alias,
                        dataset_family=dataset,
                        data=data,
                        embeddings=family_embeddings[dataset],
                        ratio=float(ratio),
                        seed=int(seed),
                    )
                )

    return {
        "metric_rows": metric_rows,
        "selected_k_rows": selected_k_rows,
        "vmf_summary_rows": vmf_summary_rows,
        "training_rows": training_rows,
        "skipped_rows": skipped_rows,
    }


def _build_splits(
    split_source_model: Mapping[str, object],
    *,
    full_cache_root: Path,
    output_root: Path,
    datasets: Sequence[str],
    seed: int,
) -> dict[str, dict[str, str]]:
    split_maps: dict[str, dict[str, str]] = {}
    for dataset in datasets:
        rows = list(
            stream_stage_b_full_cache_entries(
                split_source_model,
                full_cache_root,
                dataset_family=dataset,
                include_tensors=False,
            )
        )
        manifest = build_closeout_family_split(rows, family=dataset, seed=seed)
        manifest["stage"] = "stage_b2"
        manifest["split_source_model"] = split_source_model["model_alias"]
        manifest["split_application"] = "image_id assignments are applied to every panel model"
        path = output_root / "manifests" / DATASET_OUTPUT_NAMES[dataset]
        write_split_manifest(manifest, path)
        split_maps[dataset] = {
            str(row["image_id"]): str(row["split"]) for row in manifest["assignments"]
        }
    return split_maps


def _load_family_entries(
    model_row: Mapping[str, object],
    *,
    full_cache_root: Path,
    dataset_family: str,
    split_map: Mapping[str, str],
    limit: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in stream_stage_b_full_cache_entries(
        model_row,
        full_cache_root,
        dataset_family=dataset_family,
        include_tensors=True,
    ):
        image_id = str(entry.get("image_id", ""))
        split = split_map.get(image_id)
        if split is None:
            raise ValueError(f"missing Stage B2 split for image_id={image_id} family={dataset_family}")
        row = dict(entry)
        row["stage_b2_split"] = split
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def _primary_family_data(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary_entries: list[dict[str, object]] = []
    labels: list[int] = []
    splits: list[str] = []
    for entry in entries:
        stage_row = dict(entry)
        stage_row["stage_b_split"] = stage_row["stage_b2_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.CORRECT:
            primary_entries.append(dict(entry))
            labels.append(0)
            splits.append(str(entry["stage_b2_split"]))
        elif population == PopulationClass.HARD_HALLUCINATION:
            primary_entries.append(dict(entry))
            labels.append(1)
            splits.append(str(entry["stage_b2_split"]))
    if not primary_entries:
        raise ValueError("no Stage B2 primary population rows")
    trajectories = np.stack([build_lstm_trajectory(row) for row in primary_entries], axis=0).astype(
        np.float32,
        copy=False,
    )
    return {
        "entries": primary_entries,
        "labels": np.asarray(labels, dtype=np.int64),
        "splits": np.asarray(splits),
        "trajectories": trajectories,
    }


def _classifier_scores(
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    selected_train_indices: np.ndarray,
    family_embeddings: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    selected_labels = np.asarray(train_labels, dtype=np.int64)[selected_train_indices]
    selected_embeddings = np.asarray(train_embeddings, dtype=np.float32)[selected_train_indices]
    if np.unique(selected_labels).size < 2:
        raise ValueError("classifier control training labels must contain both classes")
    stacked_eval = np.concatenate(list(family_embeddings.values()), axis=0)
    classifier = train_logistic_diagnostic(
        selected_embeddings,
        selected_labels,
        eval_x=stacked_eval,
        seed=seed,
        max_iter=1000,
    )
    assert classifier.eval_scores is not None
    result: dict[str, np.ndarray] = {}
    offset = 0
    for family, embeddings in family_embeddings.items():
        next_offset = offset + embeddings.shape[0]
        result[family] = classifier.eval_scores[offset:next_offset]
        offset = next_offset
    return result


def _select_knn_for_budget(
    *,
    model_alias: str,
    ratio: float,
    seed: int,
    data: Mapping[str, object],
    embeddings: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    splits = np.asarray(data["splits"])
    bank_mask = (splits == "bank") & (labels == 0)
    cal_mask = splits == "cal"
    candidates = generate_stage_b_k_candidates(num_bank_correct=int(bank_mask.sum()))
    if not candidates:
        raise ValueError(f"{model_alias}/ratio={ratio:g}/seed={seed} has no k candidates")
    if np.unique(labels[cal_mask]).size < 2:
        raise ValueError(f"{model_alias}/ratio={ratio:g}/seed={seed} RePOPE cal missing one class")
    tuning_rows: list[dict[str, object]] = []
    for k_value in candidates:
        scores = compute_stage_b_knn_scores(
            bank_embeddings=embeddings[bank_mask],
            query_embeddings=embeddings[cal_mask],
            k=int(k_value),
        )
        metrics = binary_diagnostic_metrics(labels[cal_mask], scores)
        tuning_rows.append(
            {
                "row_type": "tuning_candidate",
                "model_alias": model_alias,
                "objective": STAGE_B2_OBJECTIVE,
                "dataset_family": "repope",
                "split": "cal",
                "metric_split": "cal",
                "readout": "Diag-kNN-tuned",
                "k": int(k_value),
                "num_bank_correct": int(bank_mask.sum()),
                "bank_size": int(bank_mask.sum()),
                "negative_budget_ratio": float(ratio),
                "negative_budget_seed": int(seed),
                "metric_status": "passed",
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    selected = select_stage_b2_knn_k(tuning_rows)
    selected_row = {
        "row_type": "selected",
        "model_alias": model_alias,
        "objective": STAGE_B2_OBJECTIVE,
        "selected_k": int(selected["k"]),
        "selected_on": "repope/cal",
        "frozen_for_test": True,
        "selection_metric": "pr_auc",
        "selection_pr_auc": float(selected["pr_auc"]),
        "selection_roc_auc": float(selected["roc_auc"]),
        "num_bank_correct": int(bank_mask.sum()),
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
        "metric_status": "passed",
    }
    return selected_row, tuning_rows


def _evaluate_budget_family(
    *,
    model_alias: str,
    dataset_family: str,
    data: Mapping[str, object],
    embeddings: np.ndarray,
    classifier_scores: np.ndarray,
    selected_k: int,
    bootstrap: int,
    seed: int,
    ratio: float,
    available_correct: int,
    available_hard: int,
    used_hard: int,
) -> list[dict[str, object]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    splits = np.asarray(data["splits"])
    entries = data["entries"]
    all_entries = data["all_entries"]
    bank_mask = (splits == "bank") & (labels == 0)
    rows = [
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            readout="Diag-Classifier",
            labels=labels,
            splits=splits,
            scores=classifier_scores,
            entries=entries,
            all_entries=all_entries,
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_k="",
            num_bank_correct=int(bank_mask.sum()),
            available_correct=available_correct,
            available_hard=available_hard,
            used_hard=used_hard,
        )
    ]
    knn_scores = compute_stage_b_knn_scores(
        bank_embeddings=embeddings[bank_mask],
        query_embeddings=embeddings,
        k=int(selected_k),
    )
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            readout="Diag-kNN-tuned",
            labels=labels,
            splits=splits,
            scores=knn_scores,
            entries=entries,
            all_entries=all_entries,
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_k=int(selected_k),
            num_bank_correct=int(bank_mask.sum()),
            available_correct=available_correct,
            available_hard=available_hard,
            used_hard=used_hard,
        )
    )
    prototype = fit_single_vmf_prototype(embeddings[bank_mask])
    vmf_scores = score_single_vmf_prototype(prototype, embeddings)
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            readout="Diag-vMF",
            labels=labels,
            splits=splits,
            scores=vmf_scores,
            entries=entries,
            all_entries=all_entries,
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_k="",
            num_bank_correct=int(bank_mask.sum()),
            available_correct=available_correct,
            available_hard=available_hard,
            used_hard=used_hard,
        )
    )
    return rows


def _metric_row(
    *,
    model_alias: str,
    dataset_family: str,
    readout: str,
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
    entries: Sequence[Mapping[str, object]],
    all_entries: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
    ratio: float,
    selected_k: int | str,
    num_bank_correct: int,
    available_correct: int,
    available_hard: int,
    used_hard: int,
) -> dict[str, object]:
    mask = splits == "test"
    y = labels[mask]
    split_scores = np.asarray(scores, dtype=np.float32)[mask]
    undefined_reason = ""
    if y.size == 0:
        undefined_reason = "no samples in test split"
    elif np.unique(y).size < 2:
        undefined_reason = "one class present in test split"
    elif not np.isfinite(split_scores).all():
        undefined_reason = "non-finite scores"

    if undefined_reason:
        metrics = _undefined_metrics()
        ci_low = {"pr_auc": float("nan"), "roc_auc": float("nan")}
        ci_high = {"pr_auc": float("nan"), "roc_auc": float("nan")}
    else:
        metrics = binary_diagnostic_metrics(y, split_scores)
        intervals = bootstrap_binary_metrics(y, split_scores, num_bootstrap=bootstrap, seed=seed)
        ci_low = {name: intervals[name].lower for name in ("pr_auc", "roc_auc")}
        ci_high = {name: intervals[name].upper for name in ("pr_auc", "roc_auc")}
    excluded = _excluded_counts(all_entries, split="test")
    return {
        "model_alias": model_alias,
        "model_name": model_alias,
        "dataset_family": dataset_family,
        "encoder_family": STAGE_B_ENCODER_FAMILY,
        "objective": STAGE_B2_OBJECTIVE,
        "readout": readout,
        "eval_split": "test",
        "metric_split": "test",
        "eval_scope": "pooled",
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
        "selected_k": selected_k,
        "budget_status": "evaluated",
        "budget_skip_reason": "",
        "metric_status": "undefined" if undefined_reason else "passed",
        "failure_reason": undefined_reason,
        "pr_auc": metrics["pr_auc"],
        "pr_auc_ci_low": ci_low["pr_auc"],
        "pr_auc_ci_high": ci_high["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "roc_auc_ci_low": ci_low["roc_auc"],
        "roc_auc_ci_high": ci_high["roc_auc"],
        "average_precision": metrics["average_precision"],
        "tpr_at_1pct_fpr": metrics["tpr_at_1pct_fpr"],
        "fpr_at_95pct_tpr": metrics["fpr_at_95pct_tpr"],
        "num_test": int(y.size),
        "num_test_correct": int(np.sum(y == 0)),
        "num_test_hard_hallucination": int(np.sum(y == 1)),
        "num_bank_correct": int(num_bank_correct),
        "num_encoder_train_correct": int(available_correct),
        "num_encoder_train_hard_hallucination_available": int(available_hard),
        "num_encoder_train_hard_hallucination_used": int(used_hard),
        "num_encoder_train": int(available_correct + used_hard),
        "num_encoder_train_hallucination": int(used_hard),
        "num_excluded_false_negative": excluded["false_negative"],
        "num_excluded_parsed_none": excluded["parsed_none"],
    }


def _vmf_summary_row(
    *,
    model_alias: str,
    dataset_family: str,
    data: Mapping[str, object],
    embeddings: np.ndarray,
    ratio: float,
    seed: int,
) -> dict[str, object]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    splits = np.asarray(data["splits"])
    bank_mask = (splits == "bank") & (labels == 0)
    prototype = fit_single_vmf_prototype(embeddings[bank_mask])
    return {
        "model_alias": model_alias,
        "dataset_family": dataset_family,
        "objective": STAGE_B2_OBJECTIVE,
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
        "num_bank_correct": int(bank_mask.sum()),
        "embedding_dim": prototype["embedding_dim"],
        "mean_resultant_length": prototype["mean_resultant_length"],
        "concentration_proxy": prototype["concentration_proxy"],
    }


def _skipped_metric_rows(
    model_alias: str,
    datasets: Sequence[str],
    ratio: float,
    seed: int,
    skip: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for readout in READOUTS:
            rows.append(
                {
                    "model_alias": model_alias,
                    "model_name": model_alias,
                    "dataset_family": dataset,
                    "encoder_family": STAGE_B_ENCODER_FAMILY,
                    "objective": STAGE_B2_OBJECTIVE,
                    "readout": readout,
                    "eval_split": "test",
                    "metric_split": "test",
                    "eval_scope": "pooled",
                    "negative_budget_ratio": float(ratio),
                    "negative_budget_seed": int(seed),
                    "selected_k": "",
                    "budget_status": "skipped",
                    "budget_skip_reason": str(skip["budget_skip_reason"]),
                    "metric_status": "skipped",
                    "failure_reason": str(skip["budget_skip_reason"]),
                    **_undefined_metrics(),
                    "pr_auc_ci_low": float("nan"),
                    "pr_auc_ci_high": float("nan"),
                    "roc_auc_ci_low": float("nan"),
                    "roc_auc_ci_high": float("nan"),
                    "num_test": 0,
                    "num_test_correct": 0,
                    "num_test_hard_hallucination": 0,
                    "num_bank_correct": 0,
                    "num_encoder_train_correct": int(skip["num_encoder_train_correct"]),
                    "num_encoder_train_hard_hallucination_available": int(
                        skip["num_encoder_train_hard_hallucination_available"]
                    ),
                    "num_encoder_train_hard_hallucination_used": 0,
                    "num_encoder_train": int(skip["num_encoder_train_correct"]),
                    "num_encoder_train_hallucination": 0,
                    "num_excluded_false_negative": 0,
                    "num_excluded_parsed_none": 0,
                }
            )
    return rows


def _failed_metric_rows(
    model_alias: str,
    datasets: Sequence[str],
    ratios: Sequence[float],
    seeds: Sequence[int],
    reason: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ratio in ratios:
        for seed in seeds:
            rows.extend(
                _skipped_metric_rows(
                    model_alias,
                    datasets,
                    float(ratio),
                    int(seed),
                    {
                        "budget_skip_reason": reason,
                        "num_encoder_train_correct": 0,
                        "num_encoder_train_hard_hallucination_available": 0,
                    },
                )
            )
    for row in rows:
        row["budget_status"] = "failed"
        row["metric_status"] = "failed"
    return rows


def _write_stage_b2_tables(
    report_dir: Path,
    metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    paths = {
        "repope_knn": report_dir / "repope_negative_budget_knn.csv",
        "repope_classifier": report_dir / "repope_negative_budget_classifier.csv",
        "repope_vmf": report_dir / "repope_negative_budget_vmf.csv",
        "pope_secondary": report_dir / "pope_secondary_table.csv",
        "dash_b_secondary": report_dir / "dash_b_secondary_table.csv",
    }
    write_csv_rows(
        paths["repope_knn"],
        [
            row
            for row in metric_rows
            if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-kNN-tuned"
        ],
    )
    write_csv_rows(
        paths["repope_classifier"],
        [
            row
            for row in metric_rows
            if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-Classifier"
        ],
    )
    write_csv_rows(
        paths["repope_vmf"],
        [
            row
            for row in metric_rows
            if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-vMF"
        ],
    )
    write_csv_rows(paths["pope_secondary"], [row for row in metric_rows if row.get("dataset_family") == "pope"])
    write_csv_rows(paths["dash_b_secondary"], [row for row in metric_rows if row.get("dataset_family") == "dash-b"])
    return paths


def _per_model_negative_budget_summary(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    panel_models: Sequence[str],
    excluded_models: Mapping[str, str],
    model_failures: Mapping[str, str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in metric_rows:
        if (
            row.get("dataset_family") == "repope"
            and row.get("readout") == "Diag-kNN-tuned"
            and row.get("metric_status") == "passed"
        ):
            value = _finite_float(row.get("pr_auc"))
            if value is not None:
                grouped[(str(row["model_alias"]), float(row["negative_budget_ratio"]))].append(value)
    rows: list[dict[str, object]] = []
    for model in panel_models:
        if model in excluded_models:
            rows.append({"model_alias": model, "status": "excluded", "reason": excluded_models[model]})
            continue
        if model in model_failures:
            rows.append({"model_alias": model, "status": "failed", "reason": model_failures[model]})
            continue
        for ratio in REQUIRED_STAGE_B2_RATIOS:
            values = grouped.get((model, float(ratio)), [])
            rows.append(
                {
                    "model_alias": model,
                    "status": "evaluated" if values else "missing_or_skipped",
                    "negative_budget_ratio": float(ratio),
                    "repope_knn_pr_auc_mean": "" if not values else float(np.mean(values)),
                    "repope_knn_pr_auc_std": "" if not values else float(np.std(values, ddof=0)),
                    "repope_knn_pr_auc_min": "" if not values else float(np.min(values)),
                    "repope_knn_pr_auc_max": "" if not values else float(np.max(values)),
                    "num_seeds": len(values),
                    "reason": "",
                }
            )
    return rows


def _negative_budget_verdict(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ratio_scores: dict[float, list[float]] = defaultdict(list)
    for row in metric_rows:
        if (
            row.get("dataset_family") == "repope"
            and row.get("readout") == "Diag-kNN-tuned"
            and row.get("metric_status") == "passed"
        ):
            value = _finite_float(row.get("pr_auc"))
            if value is not None:
                ratio_scores[float(row["negative_budget_ratio"])].append(value)
    means = {
        f"{ratio:g}": float(np.mean(values))
        for ratio, values in sorted(ratio_scores.items(), reverse=True)
        if values
    }
    if not means or "1" not in means:
        return {
            "verdict": "negative_budget_inconclusive",
            "per_ratio_repope_knn_pr_auc_mean": means,
            "reason": "missing baseline 1.0 ratio or no finite primary rows",
        }
    baseline = means["1"]
    threshold = baseline * 0.95
    stable = [
        float(ratio)
        for ratio, value in means.items()
        if float(value) >= threshold
    ]
    if not stable:
        label = "negative_budget_inconclusive"
        material = None
    else:
        lowest = min(stable)
        label = f"negative_budget_stable_to_{_ratio_label(lowest)}"
        material = _next_lower_ratio(lowest, means)
    return {
        "verdict": label,
        "baseline_ratio": 1.0,
        "baseline_repope_knn_pr_auc_mean": baseline,
        "per_ratio_repope_knn_pr_auc_mean": means,
        "material_degradation_below_ratio": material,
        "stability_rule": "stable means panel average RePOPE kNN PR-AUC is at least 95% of ratio 1.0",
    }


def _next_lower_ratio(ratio: float, means: Mapping[str, float]) -> float | None:
    lower = sorted(float(value) for value in means if float(value) < ratio)
    return max(lower) if lower else None


def _ratio_label(ratio: float) -> str:
    return f"{int(round(float(ratio) * 100)):02d}pct"


def _excluded_counts(entries: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    counts = {"false_negative": 0, "parsed_none": 0}
    for row in entries:
        if str(row.get("stage_b2_split", "")) != split:
            continue
        stage_row = dict(row)
        stage_row["stage_b_split"] = stage_row["stage_b2_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.FALSE_NEGATIVE_ERROR:
            counts["false_negative"] += 1
        elif population == PopulationClass.PARSED_NONE:
            counts["parsed_none"] += 1
    return counts


def _validate_datasets(values: Sequence[str]) -> list[str]:
    datasets = [str(value) for value in values]
    invalid = [dataset for dataset in datasets if dataset not in FAMILY_SUBSETS]
    if invalid:
        raise SystemExit("unsupported Stage B2 dataset(s): " + ", ".join(invalid))
    if "repope" not in datasets:
        raise SystemExit("Stage B2 requires repope for budget evaluation")
    return datasets


def _select_model_rows(
    model_rows: Sequence[Mapping[str, object]],
    *,
    requested: Sequence[str] | None,
) -> list[dict[str, object]]:
    if not requested:
        return [dict(row) for row in model_rows]
    requested_set = {str(model) for model in requested}
    selected = [dict(row) for row in model_rows if str(row.get("model_alias", "")) in requested_set]
    found = {str(row["model_alias"]) for row in selected}
    missing = sorted(requested_set - found)
    if missing:
        raise SystemExit("requested Stage B2 models not found in unified manifest: " + ", ".join(missing))
    return selected


def _training_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _undefined_metrics() -> dict[str, float]:
    return {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
        "fpr_at_95pct_tpr": float("nan"),
    }


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _render_preflight_markdown(preflight: Mapping[str, object]) -> str:
    lines = [
        "# Stage B2 Preflight",
        "",
        "Stage B2 tests Proxy Anchor negative-budget efficiency only.",
        "",
        f"- total_panel_models: {preflight.get('total_panel_models')}",
        f"- evaluable_models: {preflight.get('evaluable_models')}",
        f"- cache_root_readiness: {preflight.get('cache_root_readiness')}",
        f"- split_readiness: {preflight.get('split_readiness')}",
        "",
        "## Excluded Models",
        "",
    ]
    excluded = preflight.get("excluded_models", {})
    if isinstance(excluded, Mapping) and excluded:
        for model, reason in excluded.items():
            lines.append(f"- {model}: {reason}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _render_summary_markdown(summary: Mapping[str, object]) -> str:
    verdict = summary.get("verdict", {})
    if not isinstance(verdict, Mapping):
        verdict = {}
    lines = [
        "# Stage B2 Summary",
        "",
        "Stage B2 tests negative-budget efficiency for Proxy Anchor only.",
        "It does not choose the final detector. Stage C has not started.",
        "",
        f"- stage_c_started: {str(summary.get('stage_c_started', False)).lower()}",
        f"- objective: {summary.get('objective', STAGE_B2_OBJECTIVE)}",
        f"- verdict: {verdict.get('verdict', 'negative_budget_inconclusive')}",
        "",
        "## RePOPE kNN PR-AUC By Ratio",
        "",
    ]
    means = verdict.get("per_ratio_repope_knn_pr_auc_mean", {})
    if isinstance(means, Mapping) and means:
        for ratio, value in means.items():
            lines.append(f"- {ratio}: {value}")
    else:
        lines.append("- no finite primary rows")
    lines.extend(["", "## Excluded Models", ""])
    excluded = summary.get("excluded_models", {})
    if isinstance(excluded, Mapping) and excluded:
        for model, reason in excluded.items():
            lines.append(f"- {model}: {reason}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
