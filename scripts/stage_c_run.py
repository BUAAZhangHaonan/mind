#!/usr/bin/env python3
"""Run Stage C detector-family comparison on frozen Proxy Anchor embeddings."""

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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mind.trajectory.stage_a_closeout import (  # noqa: E402
    FAMILY_SUBSETS,
    build_closeout_family_split,
    write_csv_rows,
    write_split_manifest,
)
from mind.trajectory.stage_a_metrics import binary_diagnostic_metrics, bootstrap_binary_metrics  # noqa: E402
from mind.trajectory.stage_a_population import PopulationClass, classify_entry  # noqa: E402
from mind.trajectory.stage_a_representations import build_lstm_trajectory  # noqa: E402
from mind.trajectory.stage_b2_budget import subsample_stage_b2_training_indices  # noqa: E402
from mind.trajectory.stage_b_glm_qc import GLM_MODEL_ALIAS  # noqa: E402
from mind.trajectory.stage_b_manifest import stream_stage_b_full_cache_entries  # noqa: E402
from mind.trajectory.stage_b_objectives import STAGE_B_ENCODER_FAMILY  # noqa: E402
from mind.trajectory.stage_b_training import score_stage_b_lstm, train_stage_b_lstm  # noqa: E402
from mind.trajectory.stage_b4_vmf import (  # noqa: E402
    fit_mixture_vmf_support,
    fit_single_vmf_support,
    score_mixture_vmf_support,
    score_single_vmf_support,
)
from mind.trajectory.stage_c_manifest import (  # noqa: E402
    REQUIRED_STAGE_C_RATIO,
    REQUIRED_STAGE_C_SEEDS,
    STAGE_C_GLM_EXCLUSION_REASON,
    STAGE_C_OBJECTIVE,
    build_stage_c_preflight,
    load_stage_c_panel,
    validate_stage_c_plan,
)
from mind.trajectory.stage_c_status import (  # noqa: E402
    build_stage_c_per_model_summary,
    summarize_stage_c_detector_panel,
    validate_stage_c_summary,
)
from mind.trajectory.stage_c_support import (  # noqa: E402
    STAGE_C_LOGISTIC_C_VALUES,
    STAGE_C_MIXTURE_K_VALUES,
    build_stage_c_knn_grid,
    build_stage_c_radius_grid,
    build_stage_c_vmf_grid,
    score_knn_density_support,
    score_radius_ball_support,
    select_stage_c_candidate,
)


DATASET_OUTPUT_NAMES = {
    "repope": "repope_family_split_manifest.json",
    "pope": "pope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}
STAGE_C_METHODS = ("single_vmf", "mixture_vmf", "radius_ball", "knn", "logistic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageC"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["repope", "pope", "dash-b"])
    parser.add_argument("--ratio", type=float, default=REQUIRED_STAGE_C_RATIO)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REQUIRED_STAGE_C_SEEDS))
    parser.add_argument("--objective", default=STAGE_C_OBJECTIVE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-family", type=int, default=None)
    return parser


def _stage_c_output_paths(output_root: Path) -> dict[str, Path]:
    preflight_dir = output_root / "preflight"
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    return {
        "preflight_dir": preflight_dir,
        "manifest_dir": manifest_dir,
        "report_dir": report_dir,
        "preflight_json": preflight_dir / "stageC_preflight.json",
        "preflight_md": preflight_dir / "stageC_preflight.md",
        "repope_split_manifest": manifest_dir / DATASET_OUTPUT_NAMES["repope"],
        "pope_split_manifest": manifest_dir / DATASET_OUTPUT_NAMES["pope"],
        "dash_b_split_manifest": manifest_dir / DATASET_OUTPUT_NAMES["dash-b"],
        "metrics_long": report_dir / "stageC_metrics_long.csv",
        "repope_main_table": report_dir / "repope_main_table.csv",
        "pope_secondary_table": report_dir / "pope_secondary_table.csv",
        "dash_b_secondary_table": report_dir / "dash_b_secondary_table.csv",
        "knn_selected_k": report_dir / "knn_selected_k.csv",
        "radius_ball_selected_rho": report_dir / "radius_ball_selected_rho.csv",
        "vmf_selected_k": report_dir / "vmf_selected_k.csv",
        "logistic_selected_c": report_dir / "logistic_selected_c.csv",
        "per_model_detector_summary": report_dir / "per_model_detector_summary.csv",
        "summary_json": report_dir / "STAGE_C_SUMMARY.json",
        "summary_md": report_dir / "STAGE_C_SUMMARY.md",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _stage_c_output_paths(args.output_root)
    for directory_key in ("preflight_dir", "manifest_dir", "report_dir"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    datasets = _validate_datasets(args.datasets)
    plan = validate_stage_c_plan(
        ratio=float(args.ratio),
        seeds=args.seeds,
        objective=str(args.objective),
        encoder_family=STAGE_B_ENCODER_FAMILY,
    )
    manifest = load_stage_c_panel(args.full_cache_root)
    panel_models = [str(row["model_alias"]) for row in manifest.models]
    model_rows = _select_model_rows(manifest.models, requested=args.models)
    split_maps = _build_splits(
        manifest.models[0],
        full_cache_root=args.full_cache_root,
        output_root=args.output_root,
        datasets=datasets,
        seed=REQUIRED_STAGE_C_SEEDS[0],
    )
    excluded_models: dict[str, str] = {}
    if GLM_MODEL_ALIAS in panel_models:
        excluded_models[GLM_MODEL_ALIAS] = STAGE_C_GLM_EXCLUSION_REASON

    preflight = build_stage_c_preflight(
        manifest,
        excluded_models=excluded_models,
        split_ready=True,
        primary_dataset_available="repope" in datasets,
    )
    preflight["plan"] = plan
    _write_json(paths["preflight_json"], preflight)
    paths["preflight_md"].write_text(_render_preflight_markdown(preflight), encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    knn_selected_rows: list[dict[str, object]] = []
    radius_selected_rows: list[dict[str, object]] = []
    vmf_selected_rows: list[dict[str, object]] = []
    logistic_selected_rows: list[dict[str, object]] = []
    model_failures: dict[str, str] = {}
    included_rows = [row for row in model_rows if str(row.get("model_alias", "")) not in excluded_models]
    device = _training_device(str(args.device))
    for index, model_row in enumerate(included_rows, start=1):
        model_alias = str(model_row["model_alias"])
        print(f"[{index}/{len(included_rows)}] Stage C model={model_alias}", flush=True)
        try:
            result = _run_model_stage_c(
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
        knn_selected_rows.extend(result["knn_selected_rows"])
        radius_selected_rows.extend(result["radius_selected_rows"])
        vmf_selected_rows.extend(result["vmf_selected_rows"])
        logistic_selected_rows.extend(result["logistic_selected_rows"])

    status_exclusions = dict(excluded_models)
    status_exclusions.update(model_failures)
    panel_for_summary = [str(row["model_alias"]) for row in model_rows]
    per_model_rows = build_stage_c_per_model_summary(
        metric_rows,
        panel_models=panel_for_summary,
        excluded_models=status_exclusions,
    )
    detector_panel = summarize_stage_c_detector_panel(metric_rows)
    evaluated_models = sorted(
        {
            str(row["model_alias"])
            for row in metric_rows
            if str(row.get("metric_status", "")) in {"passed", "undefined"}
        }
    )

    write_csv_rows(paths["metrics_long"], metric_rows)
    write_csv_rows(paths["repope_main_table"], _filter_rows(metric_rows, dataset="repope"))
    write_csv_rows(paths["pope_secondary_table"], _filter_rows(metric_rows, dataset="pope"))
    write_csv_rows(paths["dash_b_secondary_table"], _filter_rows(metric_rows, dataset="dash-b"))
    write_csv_rows(paths["knn_selected_k"], knn_selected_rows)
    write_csv_rows(paths["radius_ball_selected_rho"], radius_selected_rows)
    write_csv_rows(paths["vmf_selected_k"], vmf_selected_rows)
    write_csv_rows(paths["logistic_selected_c"], logistic_selected_rows)
    write_csv_rows(paths["per_model_detector_summary"], per_model_rows)

    summary = validate_stage_c_summary(
        {
            "stage": "stage_c",
            "stage_d_started": False,
            "full_cache_manifest": str(manifest.path),
            "panel_models": panel_for_summary,
            "evaluated_models": evaluated_models,
            "excluded_models": status_exclusions,
            "objective": STAGE_C_OBJECTIVE,
            "encoder_family": STAGE_B_ENCODER_FAMILY,
            "negative_budget_ratio": REQUIRED_STAGE_C_RATIO,
            "negative_budget_seeds": list(REQUIRED_STAGE_C_SEEDS),
            "support_winner": detector_panel["support_winner"],
            "comparator_status": detector_panel["comparator_status"],
            "panel_verdict": detector_panel["panel_verdict"],
            "support_winner_mean_pr_auc": detector_panel["support_winner_mean_pr_auc"],
            "logistic_mean_pr_auc": detector_panel["logistic_mean_pr_auc"],
            "support_minus_logistic_pr_auc": detector_panel["support_minus_logistic_pr_auc"],
            "method_stats": detector_panel["method_stats"],
            "preflight_path": str(paths["preflight_json"]),
            "split_manifest_paths": {
                dataset: str(paths["manifest_dir"] / DATASET_OUTPUT_NAMES[dataset]) for dataset in datasets
            },
            "metrics_long_path": str(paths["metrics_long"]),
            "repope_main_table_path": str(paths["repope_main_table"]),
            "pope_secondary_table_path": str(paths["pope_secondary_table"]),
            "dash_b_secondary_table_path": str(paths["dash_b_secondary_table"]),
            "knn_selected_k_path": str(paths["knn_selected_k"]),
            "radius_ball_selected_rho_path": str(paths["radius_ball_selected_rho"]),
            "vmf_selected_k_path": str(paths["vmf_selected_k"]),
            "logistic_selected_c_path": str(paths["logistic_selected_c"]),
            "per_model_detector_summary_path": str(paths["per_model_detector_summary"]),
            "stage_c_scope": (
                "Detector-family selection on frozen Sphere-Traj-LSTM Proxy Anchor embeddings. "
                "Stage D has not started."
            ),
        }
    )
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(_render_summary_markdown(summary), encoding="utf-8")
    print(
        "Stage C summary="
        f"{paths['summary_json']} support_winner={summary['support_winner']} "
        f"panel_verdict={summary['panel_verdict']}",
        flush=True,
    )
    return 0


def _run_model_stage_c(
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
    train_trajectories_all = np.asarray(repope["trajectories"], dtype=np.float32)[train_mask]
    num_layers = int(train_trajectories_all.shape[1])
    hidden_dim = int(train_trajectories_all.shape[2])
    train_correct_available = int(np.sum(train_labels_all == 0))
    train_hard_available = int(np.sum(train_labels_all == 1))

    metric_rows: list[dict[str, object]] = []
    knn_selected_rows: list[dict[str, object]] = []
    radius_selected_rows: list[dict[str, object]] = []
    vmf_selected_rows: list[dict[str, object]] = []
    logistic_selected_rows: list[dict[str, object]] = []

    for seed in seeds:
        selected_train_indices = subsample_stage_b2_training_indices(
            train_labels_all,
            ratio=float(ratio),
            seed=int(seed),
        )
        train_labels = train_labels_all[selected_train_indices]
        train_trajectories = train_trajectories_all[selected_train_indices]
        used_hard = int(np.sum(train_labels == 1))
        trained = train_stage_b_lstm(
            train_trajectories,
            train_labels,
            objective=STAGE_C_OBJECTIVE,
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

        repope_labels = np.asarray(repope["labels"], dtype=np.int64)
        repope_splits = np.asarray(repope["splits"])
        repope_embeddings = family_embeddings["repope"]
        selected = _select_stage_c_hyperparameters(
            model_alias=model_alias,
            labels=repope_labels,
            splits=repope_splits,
            embeddings=repope_embeddings,
            train_mask=train_mask,
            selected_train_indices=selected_train_indices,
            seed=int(seed),
            ratio=float(ratio),
        )
        knn_selected_rows.append(selected["knn"])
        radius_selected_rows.append(selected["radius_ball"])
        vmf_selected_rows.append(selected["mixture_vmf"])
        logistic_selected_rows.append(selected["logistic"])
        logistic_model = _fit_logistic_model(
            repope_embeddings[train_mask][selected_train_indices],
            train_labels_all[selected_train_indices],
            C=float(selected["logistic"]["selected_C"]),
            seed=int(seed),
        )
        for dataset, data in family_cache.items():
            labels = np.asarray(data["labels"], dtype=np.int64)
            splits = np.asarray(data["splits"])
            embeddings = family_embeddings[dataset]
            logistic_scores = logistic_model.predict_proba(embeddings)[:, 1].astype(np.float32)
            metric_rows.extend(
                _evaluate_selected_methods(
                    model_alias=model_alias,
                    dataset_family=dataset,
                    data=data,
                    embeddings=embeddings,
                    labels=labels,
                    splits=splits,
                    selected=selected,
                    logistic_scores=logistic_scores,
                    bootstrap=bootstrap,
                    seed=int(seed),
                    ratio=float(ratio),
                    used_hard=used_hard,
                    train_correct_available=train_correct_available,
                    train_hard_available=train_hard_available,
                )
            )
    return {
        "metric_rows": metric_rows,
        "knn_selected_rows": knn_selected_rows,
        "radius_selected_rows": radius_selected_rows,
        "vmf_selected_rows": vmf_selected_rows,
        "logistic_selected_rows": logistic_selected_rows,
    }


def _select_stage_c_hyperparameters(
    *,
    model_alias: str,
    labels: np.ndarray,
    splits: np.ndarray,
    embeddings: np.ndarray,
    train_mask: np.ndarray,
    selected_train_indices: np.ndarray,
    seed: int,
    ratio: float,
) -> dict[str, dict[str, object]]:
    knn_grid = build_stage_c_knn_grid(
        model_alias=model_alias,
        dataset_family="repope",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=seed,
        metric_split="cal",
        ratio=ratio,
    )
    knn_selected = select_stage_c_candidate(
        knn_grid,
        method="knn",
        parameter_name="k",
        allowed_values=[int(row["k"]) for row in knn_grid],
    )
    mixture_grid = build_stage_c_vmf_grid(
        model_alias=model_alias,
        dataset_family="repope",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=seed,
        method="mixture_vmf",
        metric_split="cal",
        ratio=ratio,
    )
    vmf_selected = select_stage_c_candidate(
        mixture_grid,
        method="mixture_vmf",
        parameter_name="K",
        allowed_values=STAGE_C_MIXTURE_K_VALUES,
    )
    radius_grid = build_stage_c_radius_grid(
        model_alias=model_alias,
        dataset_family="repope",
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        seed=seed,
        metric_split="cal",
        ratio=ratio,
    )
    radius_selected = select_stage_c_candidate(
        radius_grid,
        method="radius_ball",
        parameter_name="rho",
        allowed_values=[float(row["rho"]) for row in radius_grid],
    )
    logistic_grid = _build_logistic_c_grid(
        model_alias=model_alias,
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        train_mask=train_mask,
        selected_train_indices=selected_train_indices,
        seed=seed,
        ratio=ratio,
    )
    logistic_selected = select_stage_c_candidate(
        logistic_grid,
        method="logistic",
        parameter_name="C",
        allowed_values=STAGE_C_LOGISTIC_C_VALUES,
    )
    return {
        "knn": knn_selected,
        "mixture_vmf": vmf_selected,
        "radius_ball": radius_selected,
        "logistic": logistic_selected,
    }


def _build_logistic_c_grid(
    *,
    model_alias: str,
    labels: np.ndarray,
    splits: np.ndarray,
    embeddings: np.ndarray,
    train_mask: np.ndarray,
    selected_train_indices: np.ndarray,
    seed: int,
    ratio: float,
) -> list[dict[str, object]]:
    train_embeddings = np.asarray(embeddings, dtype=np.float32)[train_mask][selected_train_indices]
    train_labels = np.asarray(labels, dtype=np.int64)[train_mask][selected_train_indices]
    cal_mask = np.asarray(splits) == "cal"
    y = np.asarray(labels, dtype=np.int64)[cal_mask]
    rows: list[dict[str, object]] = []
    for C in STAGE_C_LOGISTIC_C_VALUES:
        model = _fit_logistic_model(train_embeddings, train_labels, C=float(C), seed=int(seed))
        scores = model.predict_proba(np.asarray(embeddings, dtype=np.float32)[cal_mask])[:, 1].astype(np.float32)
        rows.append(
            {
                "row_type": "support_candidate",
                "model_alias": model_alias,
                "model_name": model_alias,
                "objective": STAGE_C_OBJECTIVE,
                "dataset_family": "repope",
                "method": "logistic",
                "support_family": "supervised_comparator",
                "readout": "StageC-logistic-C-grid",
                "split": "cal",
                "eval_split": "cal",
                "metric_split": "cal",
                "eval_scope": "pooled",
                "negative_budget_ratio": float(ratio),
                "negative_budget_seed": int(seed),
                "C": float(C),
                "parameter_value": float(C),
                "num_bank_correct": int(np.sum((np.asarray(splits) == "bank") & (np.asarray(labels) == 0))),
                "num_eval": int(y.size),
                **_candidate_metrics(y, scores),
            }
        )
    return rows


def _evaluate_selected_methods(
    *,
    model_alias: str,
    dataset_family: str,
    data: Mapping[str, object],
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    selected: Mapping[str, Mapping[str, object]],
    logistic_scores: np.ndarray,
    bootstrap: int,
    seed: int,
    ratio: float,
    used_hard: int,
    train_correct_available: int,
    train_hard_available: int,
) -> list[dict[str, object]]:
    bank_mask = (splits == "bank") & (labels == 0)
    bank_embeddings = np.asarray(embeddings, dtype=np.float32)[bank_mask]
    rows: list[dict[str, object]] = []
    single_model = fit_single_vmf_support(bank_embeddings)
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="single_vmf",
            labels=labels,
            splits=splits,
            scores=score_single_vmf_support(single_model, embeddings),
            entries=data["entries"],
            all_entries=data["all_entries"],
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_value="",
            num_bank_correct=int(bank_mask.sum()),
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
            used_hard=used_hard,
        )
    )
    mixture_model = fit_mixture_vmf_support(
        bank_embeddings,
        k=int(selected["mixture_vmf"]["selected_K"]),
        seed=int(seed),
    )
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="mixture_vmf",
            labels=labels,
            splits=splits,
            scores=score_mixture_vmf_support(mixture_model, embeddings),
            entries=data["entries"],
            all_entries=data["all_entries"],
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_value=int(selected["mixture_vmf"]["selected_K"]),
            num_bank_correct=int(bank_mask.sum()),
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
            used_hard=used_hard,
        )
    )
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="knn",
            labels=labels,
            splits=splits,
            scores=score_knn_density_support(
                bank_embeddings=bank_embeddings,
                query_embeddings=embeddings,
                k=int(selected["knn"]["selected_k"]),
            ),
            entries=data["entries"],
            all_entries=data["all_entries"],
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_value=int(selected["knn"]["selected_k"]),
            num_bank_correct=int(bank_mask.sum()),
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
            used_hard=used_hard,
        )
    )
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="radius_ball",
            labels=labels,
            splits=splits,
            scores=score_radius_ball_support(
                bank_embeddings=bank_embeddings,
                query_embeddings=embeddings,
                rho=float(selected["radius_ball"]["selected_rho"]),
            ),
            entries=data["entries"],
            all_entries=data["all_entries"],
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_value=float(selected["radius_ball"]["selected_rho"]),
            num_bank_correct=int(bank_mask.sum()),
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
            used_hard=used_hard,
        )
    )
    rows.append(
        _metric_row(
            model_alias=model_alias,
            dataset_family=dataset_family,
            method="logistic",
            labels=labels,
            splits=splits,
            scores=logistic_scores,
            entries=data["entries"],
            all_entries=data["all_entries"],
            bootstrap=bootstrap,
            seed=seed,
            ratio=ratio,
            selected_value=float(selected["logistic"]["selected_C"]),
            num_bank_correct=int(bank_mask.sum()),
            train_correct_available=train_correct_available,
            train_hard_available=train_hard_available,
            used_hard=used_hard,
        )
    )
    return rows


def _fit_logistic_model(train_x: np.ndarray, train_y: np.ndarray, *, C: float, seed: int) -> Pipeline:
    if np.unique(train_y).size < 2:
        raise ValueError("logistic comparator training labels must contain both classes")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(np.asarray(train_x, dtype=np.float32), np.asarray(train_y, dtype=np.int64))
    return model


def _candidate_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    undefined_reason = ""
    if y.size == 0:
        undefined_reason = "no samples in cal split"
    elif np.unique(y).size < 2:
        undefined_reason = "one class present in cal split"
    elif not np.isfinite(scores).all():
        undefined_reason = "non-finite scores"
    metrics = _undefined_metrics() if undefined_reason else binary_diagnostic_metrics(y, scores)
    return {"metric_status": "undefined" if undefined_reason else "passed", "failure_reason": undefined_reason, **metrics}


def _metric_row(
    *,
    model_alias: str,
    dataset_family: str,
    method: str,
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
    entries: Sequence[Mapping[str, object]],
    all_entries: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
    ratio: float,
    selected_value: int | float | str,
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
    selected_k = selected_value if method == "knn" else ""
    selected_K = selected_value if method == "mixture_vmf" else ""
    selected_rho = selected_value if method == "radius_ball" else ""
    selected_C = selected_value if method == "logistic" else ""
    return {
        "model_alias": model_alias,
        "model_name": model_alias,
        "dataset_family": dataset_family,
        "encoder_family": STAGE_B_ENCODER_FAMILY,
        "objective": STAGE_C_OBJECTIVE,
        "method": method,
        "readout": f"StageC-{method}",
        "support_family": "supervised_comparator" if method == "logistic" else method,
        "method_role": "supervised_comparator" if method == "logistic" else "support_estimator",
        "eval_split": "test",
        "metric_split": "test",
        "eval_scope": "pooled",
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
        "selected_k": selected_k,
        "selected_K": selected_K,
        "selected_rho": selected_rho,
        "selected_C": selected_C,
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
        "num_encoder_train_hard_hallucination": int(used_hard),
        "num_excluded_false_negative": excluded["false_negative"],
        "num_excluded_parsed_none": excluded["parsed_none"],
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
        manifest["stage"] = "stage_c"
        manifest["split_source_model"] = split_source_model["model_alias"]
        manifest["split_application"] = "image_id assignments are applied to every panel model"
        path = output_root / "manifests" / DATASET_OUTPUT_NAMES[dataset]
        write_split_manifest(manifest, path)
        split_maps[dataset] = {str(row["image_id"]): str(row["split"]) for row in manifest["assignments"]}
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
            raise ValueError(f"missing Stage C split for image_id={image_id} family={dataset_family}")
        row = dict(entry)
        row["stage_c_split"] = split
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
        stage_row["stage_b_split"] = stage_row["stage_c_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.CORRECT:
            primary_entries.append(dict(entry))
            labels.append(0)
            splits.append(str(entry["stage_c_split"]))
        elif population == PopulationClass.HARD_HALLUCINATION:
            primary_entries.append(dict(entry))
            labels.append(1)
            splits.append(str(entry["stage_c_split"]))
    if not primary_entries:
        raise ValueError("no Stage C primary population rows")
    trajectories = np.stack([build_lstm_trajectory(row) for row in primary_entries], axis=0).astype(np.float32, copy=False)
    return {"entries": primary_entries, "labels": np.asarray(labels, dtype=np.int64), "splits": np.asarray(splits), "trajectories": trajectories}


def _excluded_counts(entries: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    counts = {"false_negative": 0, "parsed_none": 0}
    for row in entries:
        if str(row.get("stage_c_split", "")) != split:
            continue
        stage_row = dict(row)
        stage_row["stage_b_split"] = stage_row["stage_c_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.FALSE_NEGATIVE_ERROR:
            counts["false_negative"] += 1
        elif population == PopulationClass.PARSED_NONE:
            counts["parsed_none"] += 1
    return counts


def _failed_metric_rows(model_alias: str, datasets: Sequence[str], seeds: Sequence[int], reason: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for dataset in datasets:
            for method in STAGE_C_METHODS:
                rows.append(
                    {
                        "model_alias": model_alias,
                        "model_name": model_alias,
                        "dataset_family": dataset,
                        "encoder_family": STAGE_B_ENCODER_FAMILY,
                        "objective": STAGE_C_OBJECTIVE,
                        "method": method,
                        "readout": f"StageC-{method}",
                        "support_family": "supervised_comparator" if method == "logistic" else method,
                        "method_role": "supervised_comparator" if method == "logistic" else "support_estimator",
                        "eval_split": "test",
                        "metric_split": "test",
                        "eval_scope": "pooled",
                        "negative_budget_ratio": REQUIRED_STAGE_C_RATIO,
                        "negative_budget_seed": int(seed),
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


def _validate_datasets(values: Sequence[str]) -> list[str]:
    datasets = [str(value) for value in values]
    invalid = [dataset for dataset in datasets if dataset not in FAMILY_SUBSETS]
    if invalid:
        raise SystemExit("unsupported Stage C dataset(s): " + ", ".join(invalid))
    if "repope" not in datasets:
        raise SystemExit("Stage C requires repope for detector-family selection")
    return datasets


def _select_model_rows(model_rows: Sequence[Mapping[str, object]], *, requested: Sequence[str] | None) -> list[dict[str, object]]:
    if not requested:
        return [dict(row) for row in model_rows]
    requested_set = {str(model) for model in requested}
    selected = [dict(row) for row in model_rows if str(row.get("model_alias", "")) in requested_set]
    found = {str(row["model_alias"]) for row in selected}
    missing = sorted(requested_set - found)
    if missing:
        raise SystemExit("requested Stage C models not found in unified manifest: " + ", ".join(missing))
    return selected


def _filter_rows(rows: Sequence[Mapping[str, object]], *, dataset: str) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("dataset_family", "")).lower() == dataset]


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
        "# Stage C Preflight",
        "",
        "Stage C compares support estimators on frozen Proxy Anchor embeddings.",
        "",
        f"- total_panel_models: {preflight.get('total_panel_models')}",
        f"- evaluable_models: {preflight.get('evaluable_models')}",
        f"- cache_root_readiness: {preflight.get('cache_root_readiness')}",
        f"- split_readiness: {preflight.get('split_readiness')}",
        f"- fixed_objective: {preflight.get('fixed_objective')}",
        f"- fixed_encoder_family: {preflight.get('fixed_encoder_family')}",
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
    lines = [
        "# Stage C Summary",
        "",
        "Stage C compares detector families on frozen Proxy Anchor embeddings.",
        "Stage D has not started.",
        "",
        f"- stage_d_started: {str(summary.get('stage_d_started', False)).lower()}",
        f"- objective: {summary.get('objective', STAGE_C_OBJECTIVE)}",
        f"- negative_budget_ratio: {summary.get('negative_budget_ratio', REQUIRED_STAGE_C_RATIO)}",
        f"- support_winner: {summary.get('support_winner', '')}",
        f"- comparator_status: {summary.get('comparator_status', '')}",
        f"- panel_verdict: {summary.get('panel_verdict', '')}",
        "",
        "## Outputs",
        "",
        f"- metrics_long: {summary.get('metrics_long_path', '')}",
        f"- repope_main_table: {summary.get('repope_main_table_path', '')}",
        f"- knn_selected_k: {summary.get('knn_selected_k_path', '')}",
        f"- radius_ball_selected_rho: {summary.get('radius_ball_selected_rho_path', '')}",
        f"- vmf_selected_k: {summary.get('vmf_selected_k_path', '')}",
        f"- logistic_selected_c: {summary.get('logistic_selected_c_path', '')}",
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
