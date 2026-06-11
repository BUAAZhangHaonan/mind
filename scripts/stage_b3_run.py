#!/usr/bin/env python3
"""Run Stage B3 Proxy Anchor kNN scale-stability diagnostics."""

from __future__ import annotations

import argparse
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
from mind.trajectory.stage_b_glm_qc import GLM_MODEL_ALIAS  # noqa: E402
from mind.trajectory.stage_b_manifest import stream_stage_b_full_cache_entries  # noqa: E402
from mind.trajectory.stage_b_objectives import STAGE_B_ENCODER_FAMILY  # noqa: E402
from mind.trajectory.stage_b_training import score_stage_b_lstm, train_stage_b_lstm  # noqa: E402
from mind.trajectory.stage_b_vmf import fit_single_vmf_prototype, score_single_vmf_prototype  # noqa: E402
from mind.trajectory.stage_b2_budget import subsample_stage_b2_training_indices  # noqa: E402
from mind.trajectory.stage_b3_knn import (  # noqa: E402
    build_stage_b3_knn_scale_grid,
    build_stage_b3_stability_band,
    select_stage_b3_knn_k,
)
from mind.trajectory.stage_b3_manifest import (  # noqa: E402
    REQUIRED_STAGE_B3_RATIO,
    REQUIRED_STAGE_B3_SEEDS,
    STAGE_B3_GLM_EXCLUSION_REASON,
    STAGE_B3_OBJECTIVE,
    build_stage_b3_preflight,
    load_stage_b3_panel,
    validate_stage_b3_plan,
)
from mind.trajectory.stage_b3_status import (  # noqa: E402
    scale_stability_verdict,
    summarize_classifier_control_status,
    validate_stage_b3_summary,
)


DATASET_OUTPUT_NAMES = {
    "repope": "repope_family_split_manifest.json",
    "pope": "pope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}
READOUTS = ("Diag-kNN-selected", "Diag-Classifier", "Diag-vMF-probe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageB3"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["repope", "pope", "dash-b"])
    parser.add_argument("--ratio", type=float, default=REQUIRED_STAGE_B3_RATIO)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REQUIRED_STAGE_B3_SEEDS))
    parser.add_argument("--objective", default=STAGE_B3_OBJECTIVE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-family", type=int, default=None)
    return parser


def _stage_b3_output_paths(output_root: Path) -> dict[str, Path]:
    preflight_dir = output_root / "preflight"
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    return {
        "preflight_dir": preflight_dir,
        "manifest_dir": manifest_dir,
        "report_dir": report_dir,
        "preflight_json": preflight_dir / "stageB3_preflight.json",
        "preflight_md": preflight_dir / "stageB3_preflight.md",
        "metrics_long": report_dir / "stageB3_metrics_long.csv",
        "knn_scale_grid": report_dir / "knn_scale_grid.csv",
        "knn_selected_k": report_dir / "knn_selected_k.csv",
        "knn_stability_band": report_dir / "knn_stability_band.csv",
        "per_model_scale_stability": report_dir / "per_model_scale_stability.csv",
        "classifier_control": report_dir / "classifier_control.csv",
        "vmf_probe_summary": report_dir / "vmf_probe_summary.csv",
        "summary_json": report_dir / "STAGE_B3_SUMMARY.json",
        "summary_md": report_dir / "STAGE_B3_SUMMARY.md",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root: Path = args.output_root
    paths = _stage_b3_output_paths(output_root)
    for directory_key in ("preflight_dir", "manifest_dir", "report_dir"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    datasets = _validate_datasets(args.datasets)
    plan = validate_stage_b3_plan(
        ratio=float(args.ratio),
        seeds=args.seeds,
        objective=str(args.objective),
        encoder_family=STAGE_B_ENCODER_FAMILY,
    )
    manifest = load_stage_b3_panel(args.full_cache_root)
    panel_models = [str(row["model_alias"]) for row in manifest.models]
    model_rows = _select_model_rows(manifest.models, requested=args.models)

    split_maps = _build_splits(
        manifest.models[0],
        full_cache_root=args.full_cache_root,
        output_root=output_root,
        datasets=datasets,
        seed=REQUIRED_STAGE_B3_SEEDS[0],
    )
    excluded_models: dict[str, str] = {}
    if GLM_MODEL_ALIAS in panel_models:
        excluded_models[GLM_MODEL_ALIAS] = STAGE_B3_GLM_EXCLUSION_REASON

    preflight = build_stage_b3_preflight(
        manifest,
        excluded_models=excluded_models,
        split_ready=True,
        primary_dataset_available="repope" in datasets,
    )
    preflight["plan"] = plan
    preflight_path = paths["preflight_json"]
    preflight_md_path = paths["preflight_md"]
    _write_json(preflight_path, preflight)
    preflight_md_path.write_text(_render_preflight_markdown(preflight), encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    scale_grid_rows: list[dict[str, object]] = []
    selected_k_rows: list[dict[str, object]] = []
    vmf_summary_rows: list[dict[str, object]] = []
    model_failures: dict[str, str] = {}

    included_model_rows = [
        row for row in model_rows if str(row.get("model_alias", "")) not in excluded_models
    ]
    device = _training_device(str(args.device))
    for index, model_row in enumerate(included_model_rows, start=1):
        model_alias = str(model_row["model_alias"])
        print(f"[{index}/{len(included_model_rows)}] Stage B3 model={model_alias}", flush=True)
        try:
            result = _run_model_stage_b3(
                model_row,
                full_cache_root=args.full_cache_root,
                datasets=datasets,
                split_maps=split_maps,
                ratio=float(plan["negative_budget_ratio"]),
                seeds=tuple(int(seed) for seed in plan["seeds"]),
                bootstrap=int(args.bootstrap),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                device=device,
                limit_per_family=args.limit_per_family,
            )
        except Exception as exc:  # noqa: BLE001 - failures must be recorded.
            reason = f"{type(exc).__name__}: {exc}"
            model_failures[model_alias] = reason
            metric_rows.extend(_failed_metric_rows(model_alias, datasets, tuple(plan["seeds"]), reason))
            print(f"  failed: {reason}", flush=True)
            continue
        metric_rows.extend(result["metric_rows"])
        scale_grid_rows.extend(result["scale_grid_rows"])
        selected_k_rows.extend(result["selected_k_rows"])
        vmf_summary_rows.extend(result["vmf_summary_rows"])

    stability_rows, per_model_stability_rows = build_stage_b3_stability_band(
        scale_grid_rows,
        selected_k_rows=selected_k_rows,
    )
    status_exclusions = dict(excluded_models)
    status_exclusions.update(model_failures)
    classifier_rows = summarize_classifier_control_status(
        metric_rows,
        panel_models=[str(row["model_alias"]) for row in model_rows],
        excluded_models=status_exclusions,
    )

    metrics_path = paths["metrics_long"]
    scale_grid_path = paths["knn_scale_grid"]
    selected_k_path = paths["knn_selected_k"]
    stability_path = paths["knn_stability_band"]
    per_model_path = paths["per_model_scale_stability"]
    classifier_path = paths["classifier_control"]
    vmf_summary_path = paths["vmf_probe_summary"]
    write_csv_rows(metrics_path, metric_rows)
    write_csv_rows(scale_grid_path, scale_grid_rows)
    write_csv_rows(selected_k_path, selected_k_rows)
    write_csv_rows(stability_path, stability_rows)
    write_csv_rows(per_model_path, _with_excluded_stability_rows(per_model_stability_rows, status_exclusions))
    write_csv_rows(classifier_path, classifier_rows)
    write_csv_rows(vmf_summary_path, vmf_summary_rows)

    evaluated_models = sorted(
        {
            str(row["model_alias"])
            for row in metric_rows
            if row.get("metric_status") in {"passed", "undefined"}
        }
    )
    summary = validate_stage_b3_summary(
        {
            "stage": "stage_b3",
            "stage_c_started": False,
            "detector_selected": False,
            "full_cache_manifest": str(manifest.path),
            "panel_models": [str(row["model_alias"]) for row in model_rows],
            "evaluated_models": evaluated_models,
            "excluded_models": status_exclusions,
            "objective": STAGE_B3_OBJECTIVE,
            "encoder_family": STAGE_B_ENCODER_FAMILY,
            "negative_budget_ratio": REQUIRED_STAGE_B3_RATIO,
            "negative_budget_seeds": list(REQUIRED_STAGE_B3_SEEDS),
            "primary_decision_metric": "RePOPE test kNN scale-grid PR-AUC stability",
            "verdict": scale_stability_verdict(per_model_stability_rows),
            "preflight_path": str(preflight_path),
            "split_manifest_paths": {
                dataset: str(paths["manifest_dir"] / DATASET_OUTPUT_NAMES[dataset]) for dataset in datasets
            },
            "metrics_long_path": str(metrics_path),
            "knn_scale_grid_path": str(scale_grid_path),
            "knn_selected_k_path": str(selected_k_path),
            "knn_stability_band_path": str(stability_path),
            "per_model_scale_stability_path": str(per_model_path),
            "classifier_control_path": str(classifier_path),
            "vmf_probe_summary_path": str(vmf_summary_path),
            "stage_b3_scope": "Proxy Anchor kNN scale stability at fixed 0.5 negative-budget ratio.",
        }
    )
    summary_path = paths["summary_json"]
    summary_md_path = paths["summary_md"]
    _write_json(summary_path, summary)
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    print(f"Stage B3 summary={summary_path} verdict={summary['verdict']['verdict']}", flush=True)
    return 0


def _run_model_stage_b3(
    model_row: Mapping[str, object],
    *,
    full_cache_root: Path,
    datasets: Sequence[str],
    split_maps: Mapping[str, Mapping[str, str]],
    ratio: float,
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
    train_correct_available = int(np.sum(train_labels_all == 0))
    train_hard_available = int(np.sum(train_labels_all == 1))

    metric_rows: list[dict[str, object]] = []
    scale_grid_rows: list[dict[str, object]] = []
    selected_k_rows: list[dict[str, object]] = []
    vmf_summary_rows: list[dict[str, object]] = []

    for seed in seeds:
        selected_train_indices = subsample_stage_b2_training_indices(
            train_labels_all,
            ratio=float(ratio),
            seed=int(seed),
        )
        train_labels = train_labels_all[selected_train_indices]
        train_trajectories = trajectories_train_all[selected_train_indices]
        used_hard = int(np.sum(train_labels == 1))
        trained = train_stage_b_lstm(
            train_trajectories,
            train_labels,
            objective=STAGE_B3_OBJECTIVE,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=int(seed),
            patience=5,
        )

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

        repope_cal_grid = build_stage_b3_knn_scale_grid(
            model_alias=model_alias,
            dataset_family="repope",
            labels=np.asarray(repope["labels"], dtype=np.int64),
            splits=np.asarray(repope["splits"]),
            embeddings=family_embeddings["repope"],
            seed=int(seed),
            ratio=float(ratio),
            metric_split="cal",
        )
        selected = select_stage_b3_knn_k(repope_cal_grid)
        selected_row = {
            "row_type": "selected",
            "model_alias": model_alias,
            "objective": STAGE_B3_OBJECTIVE,
            "selected_k": int(selected["k"]),
            "selected_on": "repope/cal",
            "frozen_for_test": True,
            "selection_metric": "pr_auc",
            "selection_pr_auc": float(selected["pr_auc"]),
            "selection_roc_auc": float(selected["roc_auc"]),
            "num_bank_correct": int(selected["num_bank_correct"]),
            "negative_budget_ratio": float(ratio),
            "negative_budget_seed": int(seed),
            "metric_status": "passed",
        }
        selected_k_rows.append(selected_row)
        scale_grid_rows.extend(repope_cal_grid)

        for dataset, data in family_cache.items():
            test_grid = build_stage_b3_knn_scale_grid(
                model_alias=model_alias,
                dataset_family=dataset,
                labels=np.asarray(data["labels"], dtype=np.int64),
                splits=np.asarray(data["splits"]),
                embeddings=family_embeddings[dataset],
                seed=int(seed),
                ratio=float(ratio),
                candidates=[int(row["k"]) for row in repope_cal_grid],
                metric_split="test",
            )
            scale_grid_rows.extend(test_grid)
            rows = _evaluate_family(
                model_alias=model_alias,
                dataset_family=dataset,
                data=data,
                embeddings=family_embeddings[dataset],
                classifier_scores=np.asarray(classifier_scores[dataset], dtype=np.float32),
                selected_k=int(selected_row["selected_k"]),
                bootstrap=bootstrap,
                seed=int(seed),
                ratio=float(ratio),
                used_hard=used_hard,
                train_correct_available=train_correct_available,
                train_hard_available=train_hard_available,
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
        "scale_grid_rows": scale_grid_rows,
        "selected_k_rows": selected_k_rows,
        "vmf_summary_rows": vmf_summary_rows,
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
        manifest["stage"] = "stage_b3"
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
            raise ValueError(f"missing Stage B3 split for image_id={image_id} family={dataset_family}")
        row = dict(entry)
        row["stage_b3_split"] = split
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
        stage_row["stage_b_split"] = stage_row["stage_b3_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.CORRECT:
            primary_entries.append(dict(entry))
            labels.append(0)
            splits.append(str(entry["stage_b3_split"]))
        elif population == PopulationClass.HARD_HALLUCINATION:
            primary_entries.append(dict(entry))
            labels.append(1)
            splits.append(str(entry["stage_b3_split"]))
    if not primary_entries:
        raise ValueError("no Stage B3 primary population rows")
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


def _evaluate_family(
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
    used_hard: int,
    train_correct_available: int,
    train_hard_available: int,
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
            used_hard=used_hard,
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
        )
    ]
    knn_scores = _knn_scores_for_selected(embeddings=embeddings, labels=labels, splits=splits, selected_k=selected_k)
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            readout="Diag-kNN-selected",
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
            used_hard=used_hard,
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
        )
    )
    prototype = fit_single_vmf_prototype(embeddings[bank_mask])
    vmf_scores = score_single_vmf_prototype(prototype, embeddings)
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            readout="Diag-vMF-probe",
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
            used_hard=used_hard,
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
        )
    )
    return rows


def _knn_scores_for_selected(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    selected_k: int,
) -> np.ndarray:
    from mind.trajectory.stage_b3_knn import compute_stage_b_knn_scores

    bank_mask = (splits == "bank") & (labels == 0)
    return compute_stage_b_knn_scores(
        bank_embeddings=np.asarray(embeddings, dtype=np.float32)[bank_mask],
        query_embeddings=np.asarray(embeddings, dtype=np.float32),
        k=int(selected_k),
    )


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
    train_correct_available: int,
    train_hard_available: int,
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
        "objective": STAGE_B3_OBJECTIVE,
        "readout": readout,
        "eval_split": "test",
        "metric_split": "test",
        "eval_scope": "pooled",
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
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
        "training_dataset_family": "repope",
        "num_encoder_train_correct": int(train_correct_available),
        "num_encoder_train_hard_hallucination_available": int(train_hard_available),
        "num_encoder_train_hard_hallucination_used": int(used_hard),
        "num_encoder_train": int(train_correct_available + used_hard),
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
        "objective": STAGE_B3_OBJECTIVE,
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
        "num_bank_correct": int(bank_mask.sum()),
        "embedding_dim": prototype["embedding_dim"],
        "mean_resultant_length": prototype["mean_resultant_length"],
        "concentration_proxy": prototype["concentration_proxy"],
    }


def _failed_metric_rows(
    model_alias: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    reason: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for dataset in datasets:
            for readout in READOUTS:
                rows.append(
                    {
                        "model_alias": model_alias,
                        "model_name": model_alias,
                        "dataset_family": dataset,
                        "encoder_family": STAGE_B_ENCODER_FAMILY,
                        "objective": STAGE_B3_OBJECTIVE,
                        "readout": readout,
                        "eval_split": "test",
                        "metric_split": "test",
                        "eval_scope": "pooled",
                        "negative_budget_ratio": REQUIRED_STAGE_B3_RATIO,
                        "negative_budget_seed": int(seed),
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
                        "training_dataset_family": "repope",
                    }
                )
    return rows


def _with_excluded_stability_rows(
    rows: Sequence[Mapping[str, object]],
    excluded_models: Mapping[str, str],
) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    existing = {str(row.get("model_alias", "")) for row in output}
    for model, reason in sorted(excluded_models.items()):
        if model not in existing:
            output.append(
                {
                    "model_alias": model,
                    "verdict": "insufficient_coverage",
                    "status": "excluded",
                    "median_band_size": "",
                    "min_band_size": "",
                    "max_band_size": "",
                    "num_valid_runs": 0,
                    "required_seed_count": len(REQUIRED_STAGE_B3_SEEDS),
                    "seed_band_sizes": "",
                    "selected_k_values": "",
                    "stability_drop_tolerance": 0.02,
                    "reason": reason,
                }
            )
    return output


def _excluded_counts(entries: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    counts = {"false_negative": 0, "parsed_none": 0}
    for row in entries:
        if str(row.get("stage_b3_split", "")) != split:
            continue
        stage_row = dict(row)
        stage_row["stage_b_split"] = stage_row["stage_b3_split"]
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
        raise SystemExit("unsupported Stage B3 dataset(s): " + ", ".join(invalid))
    if "repope" not in datasets:
        raise SystemExit("Stage B3 requires repope for scale-stability evaluation")
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
        raise SystemExit("requested Stage B3 models not found in unified manifest: " + ", ".join(missing))
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
        "# Stage B3 Preflight",
        "",
        "Stage B3 evaluates Proxy Anchor kNN scale stability at a fixed 0.5 ratio.",
        "",
        f"- total_panel_models: {preflight.get('total_panel_models')}",
        f"- evaluable_models: {preflight.get('evaluable_models')}",
        f"- cache_root_readiness: {preflight.get('cache_root_readiness')}",
        f"- split_readiness: {preflight.get('split_readiness')}",
        f"- fixed_objective: {preflight.get('fixed_objective')}",
        f"- fixed_negative_budget_ratio: {preflight.get('fixed_negative_budget_ratio')}",
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
        "# Stage B3 Summary",
        "",
        "Stage B3 evaluates Proxy Anchor kNN scale stability at a fixed 0.5 ratio.",
        "Stage C has not started.",
        "",
        f"- stage_c_started: {str(summary.get('stage_c_started', False)).lower()}",
        f"- detector_selected: {str(summary.get('detector_selected', False)).lower()}",
        f"- objective: {summary.get('objective', STAGE_B3_OBJECTIVE)}",
        f"- negative_budget_ratio: {summary.get('negative_budget_ratio', REQUIRED_STAGE_B3_RATIO)}",
        f"- verdict: {verdict.get('verdict', 'scale_stability_inconclusive')}",
        "",
        "## Outputs",
        "",
        f"- metrics_long: {summary.get('metrics_long_path', '')}",
        f"- knn_scale_grid: {summary.get('knn_scale_grid_path', '')}",
        f"- knn_stability_band: {summary.get('knn_stability_band_path', '')}",
        f"- classifier_control: {summary.get('classifier_control_path', '')}",
        f"- vmf_probe_summary: {summary.get('vmf_probe_summary_path', '')}",
        "",
        "## Excluded Models",
        "",
    ]
    excluded = summary.get("excluded_models", {})
    if isinstance(excluded, Mapping) and excluded:
        for model, reason in excluded.items():
            lines.append(f"- {model}: {reason}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
