#!/usr/bin/env python3
"""Run Stage B1 objective-family diagnostics on the full-cache panel."""

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
    DEFAULT_GLM_QC_DATASETS,
    GLM_MODEL_ALIAS,
    apply_glm_qc_exclusion,
    scan_glm_cache_rows,
    write_glm_qc_reports,
)
from mind.trajectory.stage_b_knn import (  # noqa: E402
    compute_stage_b_knn_scores,
    generate_stage_b_k_candidates,
    select_stage_b_knn_k,
)
from mind.trajectory.stage_b_manifest import (  # noqa: E402
    load_stage_b_panel_manifest,
    stream_stage_b_full_cache_entries,
)
from mind.trajectory.stage_b_objectives import (  # noqa: E402
    ALLOWED_STAGE_B_OBJECTIVES,
    STAGE_B_ENCODER_FAMILY,
    validate_stage_b_objective_plan,
)
from mind.trajectory.stage_b_status import (  # noqa: E402
    decide_stage_b_verdict,
    render_stage_b_summary_markdown,
    summarize_stage_b_status,
    validate_stage_b_status,
)
from mind.trajectory.stage_b_training import score_stage_b_lstm, train_stage_b_lstm  # noqa: E402
from mind.trajectory.stage_b_vmf import fit_single_vmf_prototype, score_single_vmf_prototype  # noqa: E402


DATASET_OUTPUT_NAMES = {
    "repope": "repope_family_split_manifest.json",
    "pope": "pope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}
READOUTS = ("Diag-kNN-tuned", "Diag-vMF", "Diag-Classifier")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageB"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["repope", "pope", "dash-b"])
    parser.add_argument("--objectives", nargs="+", default=list(ALLOWED_STAGE_B_OBJECTIVES))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--glm-qc-limit", type=int, default=None)
    parser.add_argument("--skip-glm-qc", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root: Path = args.output_root
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    preflight_dir = output_root / "preflight"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir.mkdir(parents=True, exist_ok=True)

    datasets = _validate_datasets(args.datasets)
    objective_plan = validate_stage_b_objective_plan(
        objectives=args.objectives,
        encoder_family=STAGE_B_ENCODER_FAMILY,
        representation_branches=[STAGE_B_ENCODER_FAMILY],
        optimize_detector=False,
    )
    manifest = load_stage_b_panel_manifest(args.full_cache_root)
    panel_models = [str(row["model_alias"]) for row in manifest.models]
    model_rows = _select_model_rows(manifest.models, requested=args.models)

    split_maps = _build_or_load_splits(
        manifest.models[0],
        full_cache_root=args.full_cache_root,
        output_root=output_root,
        datasets=datasets,
        seed=args.seed,
    )

    qc_json_path = preflight_dir / "glm_answer_qc.json"
    qc_md_path = preflight_dir / "glm_answer_qc.md"
    excluded_models: dict[str, str] = {}
    if not args.skip_glm_qc:
        qc_rows = scan_glm_cache_rows(
            manifest,
            args.full_cache_root,
            dataset_families=tuple(dataset for dataset in datasets if dataset in DEFAULT_GLM_QC_DATASETS),
            limit=args.glm_qc_limit,
        )
        write_glm_qc_reports(qc_rows, json_path=qc_json_path, markdown_path=qc_md_path)
        excluded_models = dict(
            apply_glm_qc_exclusion(panel_models=panel_models, qc_rows=qc_rows)["excluded_models"]
        )
    else:
        write_glm_qc_reports([], json_path=qc_json_path, markdown_path=qc_md_path)

    metric_rows: list[dict[str, object]] = []
    selected_k_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    vmf_summary_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    model_failures: dict[str, str] = {}

    included_model_rows = [
        row for row in model_rows if str(row.get("model_alias", "")) not in excluded_models
    ]
    for index, model_row in enumerate(included_model_rows, start=1):
        model_alias = str(model_row["model_alias"])
        print(f"[{index}/{len(included_model_rows)}] Stage B1 model={model_alias}", flush=True)
        try:
            result = _run_model_stage_b1(
                model_row,
                full_cache_root=args.full_cache_root,
                datasets=datasets,
                split_maps=split_maps,
                objectives=tuple(objective_plan["objectives"]),
                bootstrap=args.bootstrap,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=_training_device(args.device),
                seed=args.seed,
                limit_per_family=args.limit_per_family,
            )
        except Exception as exc:  # noqa: BLE001 - failures must be reported, not hidden.
            reason = f"{type(exc).__name__}: {exc}"
            model_failures[model_alias] = reason
            metric_rows.extend(_failed_metric_rows(model_alias, datasets, tuple(objective_plan["objectives"]), reason))
            print(f"  failed: {reason}", flush=True)
            continue
        metric_rows.extend(result["metric_rows"])
        selected_k_rows.extend(result["selected_k_rows"])
        tuning_rows.extend(result["tuning_rows"])
        vmf_summary_rows.extend(result["vmf_summary_rows"])
        training_rows.extend(result["training_rows"])

    metrics_path = report_dir / "stageB1_metrics_long.csv"
    write_csv_rows(metrics_path, metric_rows)
    table_paths = _write_stage_b_tables(report_dir, metric_rows)
    selected_k_path = report_dir / "knn_selected_k.csv"
    tuning_path = report_dir / "repope_cal_knn_tuning.csv"
    vmf_summary_path = report_dir / "vmf_prototype_summary.csv"
    training_path = report_dir / "training_history.csv"
    per_model_path = report_dir / "per_model_objective_summary.csv"
    write_csv_rows(selected_k_path, selected_k_rows)
    write_csv_rows(tuning_path, tuning_rows)
    write_csv_rows(vmf_summary_path, vmf_summary_rows)
    write_csv_rows(training_path, training_rows)
    per_model_rows = _per_model_objective_summary(
        panel_models=[str(row["model_alias"]) for row in model_rows],
        metric_rows=metric_rows,
        excluded_models=excluded_models,
        model_failures=model_failures,
    )
    write_csv_rows(per_model_path, per_model_rows)

    status_exclusions = dict(excluded_models)
    status_exclusions.update(model_failures)
    status = summarize_stage_b_status(
        panel_models=[str(row["model_alias"]) for row in model_rows],
        metric_rows=metric_rows,
        excluded_models=status_exclusions,
    )
    primary_rows = [
        row
        for row in metric_rows
        if row.get("dataset_family") == "repope"
        and row.get("readout") == "Diag-kNN-tuned"
        and row.get("metric_status") == "passed"
    ]
    verdict = decide_stage_b_verdict(primary_rows, objectives=tuple(objective_plan["objectives"]))
    summary = validate_stage_b_status(
        {
            "stage": "stage_b1",
            "stage_b_started": True,
            "stage_c_started": False,
            "final_detector_selected": False,
            "full_cache_manifest": str(manifest.path),
            "panel_models": [str(row["model_alias"]) for row in model_rows],
            "evaluated_models": [
                str(row["model_alias"])
                for row in model_rows
                if str(row["model_alias"]) not in status_exclusions
            ],
            "excluded_models": status_exclusions,
            "objectives": list(objective_plan["objectives"]),
            "encoder_family": STAGE_B_ENCODER_FAMILY,
            "primary_decision_metric": "RePOPE pooled/test geodesic kNN PR-AUC",
            "status": status,
            "verdict": verdict,
            "glm_qc_json_path": str(qc_json_path),
            "glm_qc_markdown_path": str(qc_md_path),
            "split_manifest_paths": {
                dataset: str(manifest_dir / DATASET_OUTPUT_NAMES[dataset]) for dataset in datasets
            },
            "metrics_long_path": str(metrics_path),
            "repope_knn_table_path": str(table_paths["repope_knn"]),
            "repope_vmf_table_path": str(table_paths["repope_vmf"]),
            "repope_classifier_table_path": str(table_paths["repope_classifier"]),
            "pope_secondary_table_path": str(table_paths["pope_secondary"]),
            "dash_b_secondary_table_path": str(table_paths["dash_b_secondary"]),
            "selected_k_path": str(selected_k_path),
            "repope_cal_knn_tuning_path": str(tuning_path),
            "vmf_summary_path": str(vmf_summary_path),
            "per_model_objective_summary_path": str(per_model_path),
            "training_history_path": str(training_path),
            "no_stage_c_started": True,
        }
    )
    summary_path = report_dir / "STAGE_B1_SUMMARY.json"
    summary_md_path = report_dir / "STAGE_B1_SUMMARY.md"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    summary_md_path.write_text(render_stage_b_summary_markdown(summary), encoding="utf-8")
    print(f"Stage B1 summary={summary_path} verdict={verdict['verdict']}", flush=True)
    return 0


def _run_model_stage_b1(
    model_row: Mapping[str, object],
    *,
    full_cache_root: Path,
    datasets: Sequence[str],
    split_maps: Mapping[str, Mapping[str, str]],
    objectives: Sequence[str],
    bootstrap: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    limit_per_family: int | None,
) -> dict[str, list[dict[str, object]]]:
    model_alias = str(model_row["model_alias"])
    repope_entries = _load_family_entries(
        model_row,
        full_cache_root=full_cache_root,
        dataset_family="repope",
        split_map=split_maps["repope"],
        limit=limit_per_family,
    )
    repope_data = _primary_family_data(repope_entries)
    train_mask = repope_data["splits"] == "encoder_train"
    repope_train_labels = repope_data["labels"][train_mask]
    if np.unique(repope_train_labels).size < 2:
        raise ValueError(f"{model_alias}/repope encoder_train is missing one class")

    trajectories_train = repope_data["trajectories"][train_mask]
    num_layers = int(trajectories_train.shape[1])
    hidden_dim = int(trajectories_train.shape[2])

    metric_rows: list[dict[str, object]] = []
    selected_k_rows: list[dict[str, object]] = []
    tuning_all_rows: list[dict[str, object]] = []
    vmf_summary_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for objective in objectives:
        trained = train_stage_b_lstm(
            trajectories_train,
            repope_train_labels,
            objective=str(objective),
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed,
            patience=5,
        )
        for history_row in trained.history:
            row = dict(history_row)
            row.update({"model_alias": model_alias, "objective": objective})
            training_rows.append(row)

        family_scores: dict[str, dict[str, object]] = {}
        repope_embeddings, _ = score_stage_b_lstm(
            trained.model,
            repope_data["trajectories"],
            batch_size=batch_size,
        )
        family_scores["repope"] = {
            "entries": repope_data["entries"],
            "all_entries": repope_entries,
            "labels": repope_data["labels"],
            "splits": repope_data["splits"],
            "embeddings": repope_embeddings,
        }

        repope_embeddings_train = repope_embeddings[train_mask]
        classifier_scores = _classifier_scores(
            train_embeddings=repope_embeddings_train,
            train_labels=repope_train_labels,
            family_embeddings={"repope": repope_embeddings},
            seed=seed,
        )
        selected_k, tuning_rows = _select_knn_for_objective(
            model_alias=model_alias,
            objective=str(objective),
            family_data=family_scores["repope"],
        )
        selected_k_rows.append(selected_k)
        tuning_all_rows.extend(tuning_rows)

        for dataset in datasets:
            if dataset == "repope":
                data = family_scores["repope"]
                embeddings = repope_embeddings
            else:
                entries = _load_family_entries(
                    model_row,
                    full_cache_root=full_cache_root,
                    dataset_family=dataset,
                    split_map=split_maps[dataset],
                    limit=limit_per_family,
                )
                primary = _primary_family_data(entries)
                embeddings, _ = score_stage_b_lstm(
                    trained.model,
                    primary["trajectories"],
                    batch_size=batch_size,
                )
                data = {
                    "entries": primary["entries"],
                    "all_entries": entries,
                    "labels": primary["labels"],
                    "splits": primary["splits"],
                    "embeddings": embeddings,
                }
                classifier_scores.update(
                    _classifier_scores(
                        train_embeddings=repope_embeddings_train,
                        train_labels=repope_train_labels,
                        family_embeddings={dataset: embeddings},
                        seed=seed,
                    )
                )

            metric_rows.extend(
                _evaluate_family_objective(
                    model_alias=model_alias,
                    dataset_family=dataset,
                    objective=str(objective),
                    data=data,
                    classifier_scores=np.asarray(classifier_scores[dataset], dtype=np.float32),
                    selected_k=int(selected_k["selected_k"]),
                    bootstrap=bootstrap,
                    seed=seed,
                )
            )
            vmf_summary_rows.append(
                _vmf_summary_row(
                    model_alias=model_alias,
                    dataset_family=dataset,
                    objective=str(objective),
                    data=data,
                )
            )

    return {
        "metric_rows": metric_rows,
        "selected_k_rows": selected_k_rows,
        "tuning_rows": tuning_all_rows,
        "vmf_summary_rows": vmf_summary_rows,
        "training_rows": training_rows,
    }


def _build_or_load_splits(
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
        manifest["stage"] = "stage_b1"
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
            raise ValueError(f"missing Stage B split for image_id={image_id} family={dataset_family}")
        row = dict(entry)
        row["stage_b_split"] = split
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def _primary_family_data(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary_entries: list[dict[str, object]] = []
    labels: list[int] = []
    splits: list[str] = []
    for entry in entries:
        population = classify_entry(entry)
        if population == PopulationClass.CORRECT:
            primary_entries.append(dict(entry))
            labels.append(0)
            splits.append(str(entry["stage_b_split"]))
        elif population == PopulationClass.HARD_HALLUCINATION:
            primary_entries.append(dict(entry))
            labels.append(1)
            splits.append(str(entry["stage_b_split"]))
    if not primary_entries:
        raise ValueError("no Stage B primary population rows")
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
    family_embeddings: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    if np.unique(train_labels).size < 2:
        raise ValueError("classifier control training labels must contain both classes")
    stacked_eval = np.concatenate(list(family_embeddings.values()), axis=0)
    classifier = train_logistic_diagnostic(
        train_embeddings,
        train_labels,
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


def _select_knn_for_objective(
    *,
    model_alias: str,
    objective: str,
    family_data: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    labels = np.asarray(family_data["labels"], dtype=np.int64)
    splits = np.asarray(family_data["splits"])
    embeddings = np.asarray(family_data["embeddings"], dtype=np.float32)
    bank_mask = (splits == "bank") & (labels == 0)
    cal_mask = splits == "cal"
    if int(bank_mask.sum()) <= 0:
        raise ValueError(f"{model_alias}/{objective} has no RePOPE correct bank rows")
    if np.unique(labels[cal_mask]).size < 2:
        raise ValueError(f"{model_alias}/{objective} RePOPE cal split is missing one class")
    candidates = generate_stage_b_k_candidates(num_bank_correct=int(bank_mask.sum()))
    if not candidates:
        raise ValueError(f"{model_alias}/{objective} has no valid Stage B k candidates")
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
                "model_alias": model_alias,
                "objective": objective,
                "dataset_family": "repope",
                "split": "cal",
                "metric_split": "cal",
                "readout": "Diag-kNN-tuned",
                "k": int(k_value),
                "num_bank_correct": int(bank_mask.sum()),
                "bank_size": int(bank_mask.sum()),
                "metric_status": "passed",
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    selected = select_stage_b_knn_k(tuning_rows)
    selected_row = {
        "model_alias": model_alias,
        "objective": objective,
        "selected_k": int(selected["k"]),
        "selected_on": "repope/cal",
        "selection_metric": "pr_auc",
        "selection_pr_auc": float(selected["pr_auc"]),
        "selection_roc_auc": float(selected["roc_auc"]),
        "num_bank_correct": int(bank_mask.sum()),
        "metric_status": "passed",
    }
    return selected_row, tuning_rows


def _evaluate_family_objective(
    *,
    model_alias: str,
    dataset_family: str,
    objective: str,
    data: Mapping[str, object],
    classifier_scores: np.ndarray,
    selected_k: int,
    bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    splits = np.asarray(data["splits"])
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    entries = data["entries"]
    all_entries = data["all_entries"]
    rows = [
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            objective=objective,
            readout="Diag-Classifier",
            labels=labels,
            splits=splits,
            scores=classifier_scores,
            entries=entries,
            all_entries=all_entries,
            bootstrap=bootstrap,
            seed=seed,
            num_bank_correct=int(np.sum((splits == "bank") & (labels == 0))),
            selected_k="",
        )
    ]
    bank_mask = (splits == "bank") & (labels == 0)
    knn_scores = compute_stage_b_knn_scores(
        bank_embeddings=embeddings[bank_mask],
        query_embeddings=embeddings,
        k=int(selected_k),
    )
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            objective=objective,
            readout="Diag-kNN-tuned",
            labels=labels,
            splits=splits,
            scores=knn_scores,
            entries=entries,
            all_entries=all_entries,
            bootstrap=bootstrap,
            seed=seed,
            num_bank_correct=int(bank_mask.sum()),
            selected_k=int(selected_k),
        )
    )
    prototype = fit_single_vmf_prototype(embeddings[bank_mask])
    vmf_scores = score_single_vmf_prototype(prototype, embeddings)
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            objective=objective,
            readout="Diag-vMF",
            labels=labels,
            splits=splits,
            scores=vmf_scores,
            entries=entries,
            all_entries=all_entries,
            bootstrap=bootstrap,
            seed=seed,
            num_bank_correct=int(bank_mask.sum()),
            selected_k="",
        )
    )
    return rows


def _metric_row(
    *,
    model_alias: str,
    dataset_family: str,
    objective: str,
    readout: str,
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
    entries: Sequence[Mapping[str, object]],
    all_entries: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
    num_bank_correct: int,
    selected_k: int | str,
) -> dict[str, object]:
    mask = splits == "test"
    y = labels[mask]
    split_scores = scores[mask]
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
        "objective": objective,
        "readout": readout,
        "eval_split": "test",
        "metric_split": "test",
        "eval_scope": "pooled",
        "selected_k": selected_k,
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
        "num_encoder_train": int(np.sum(splits == "encoder_train")),
        "num_encoder_train_hallucination": int(
            np.sum((splits == "encoder_train") & (labels == 1))
        ),
        "num_excluded_false_negative": excluded["false_negative"],
        "num_excluded_parsed_none": excluded["parsed_none"],
    }


def _vmf_summary_row(
    *,
    model_alias: str,
    dataset_family: str,
    objective: str,
    data: Mapping[str, object],
) -> dict[str, object]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    splits = np.asarray(data["splits"])
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    bank_mask = (splits == "bank") & (labels == 0)
    prototype = fit_single_vmf_prototype(embeddings[bank_mask])
    return {
        "model_alias": model_alias,
        "dataset_family": dataset_family,
        "objective": objective,
        "num_bank_correct": int(bank_mask.sum()),
        "embedding_dim": prototype["embedding_dim"],
        "mean_resultant_length": prototype["mean_resultant_length"],
        "concentration_proxy": prototype["concentration_proxy"],
    }


def _failed_metric_rows(
    model_alias: str,
    datasets: Sequence[str],
    objectives: Sequence[str],
    reason: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for objective in objectives:
            for readout in READOUTS:
                rows.append(
                    {
                        "model_alias": model_alias,
                        "model_name": model_alias,
                        "dataset_family": dataset,
                        "encoder_family": STAGE_B_ENCODER_FAMILY,
                        "objective": objective,
                        "readout": readout,
                        "eval_split": "test",
                        "metric_split": "test",
                        "eval_scope": "pooled",
                        "selected_k": "",
                        "metric_status": "failed",
                        "failure_reason": reason,
                        **_undefined_metrics(),
                        "pr_auc_ci_low": float("nan"),
                        "pr_auc_ci_high": float("nan"),
                        "roc_auc_ci_low": float("nan"),
                        "roc_auc_ci_high": float("nan"),
                        "num_test": 0,
                        "num_test_correct": 0,
                        "num_test_hard_hallucination": 0,
                        "num_bank_correct": 0,
                        "num_encoder_train": 0,
                        "num_encoder_train_hallucination": 0,
                        "num_excluded_false_negative": 0,
                        "num_excluded_parsed_none": 0,
                    }
                )
    return rows


def _write_stage_b_tables(
    report_dir: Path,
    metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    paths = {
        "repope_knn": report_dir / "repope_objective_table_knn.csv",
        "repope_vmf": report_dir / "repope_objective_table_vmf.csv",
        "repope_classifier": report_dir / "repope_objective_table_classifier.csv",
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
        paths["repope_vmf"],
        [
            row
            for row in metric_rows
            if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-vMF"
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
    write_csv_rows(paths["pope_secondary"], [row for row in metric_rows if row.get("dataset_family") == "pope"])
    write_csv_rows(paths["dash_b_secondary"], [row for row in metric_rows if row.get("dataset_family") == "dash-b"])
    return paths


def _per_model_objective_summary(
    *,
    panel_models: Sequence[str],
    metric_rows: Sequence[Mapping[str, object]],
    excluded_models: Mapping[str, str],
    model_failures: Mapping[str, str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-kNN-tuned":
            grouped[(str(row.get("model_alias")), str(row.get("objective")))].append(row)
    rows: list[dict[str, object]] = []
    for model in panel_models:
        if model in excluded_models:
            rows.append({"model_alias": model, "status": "excluded", "reason": excluded_models[model]})
            continue
        if model in model_failures:
            rows.append({"model_alias": model, "status": "failed", "reason": model_failures[model]})
            continue
        for objective in ALLOWED_STAGE_B_OBJECTIVES:
            values = [
                float(row["pr_auc"])
                for row in grouped.get((model, objective), [])
                if row.get("metric_status") == "passed" and np.isfinite(float(row.get("pr_auc", float("nan"))))
            ]
            rows.append(
                {
                    "model_alias": model,
                    "status": "evaluated",
                    "objective": objective,
                    "repope_knn_pr_auc": "" if not values else sum(values) / len(values),
                    "reason": "",
                }
            )
    return rows


def _excluded_counts(entries: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    counts = {"false_negative": 0, "parsed_none": 0}
    for row in entries:
        if str(row.get("stage_b_split", "")) != split:
            continue
        population = classify_entry(row)
        if population == PopulationClass.FALSE_NEGATIVE_ERROR:
            counts["false_negative"] += 1
        elif population == PopulationClass.PARSED_NONE:
            counts["parsed_none"] += 1
    return counts


def _validate_datasets(values: Sequence[str]) -> list[str]:
    datasets = [str(value) for value in values]
    invalid = [dataset for dataset in datasets if dataset not in FAMILY_SUBSETS]
    if invalid:
        raise SystemExit("unsupported Stage B dataset(s): " + ", ".join(invalid))
    if "repope" not in datasets:
        raise SystemExit("Stage B1 requires repope for objective selection")
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
        raise SystemExit("requested Stage B models not found in unified manifest: " + ", ".join(missing))
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


if __name__ == "__main__":
    raise SystemExit(main())
