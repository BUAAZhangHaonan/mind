#!/usr/bin/env python3
"""Run Stage D final frozen-method evaluation."""

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
from mind.trajectory.stage_a_representations import (  # noqa: E402
    build_lstm_trajectory,
    build_representation,
)
from mind.trajectory.stage_b2_budget import subsample_stage_b2_training_indices  # noqa: E402
from mind.trajectory.stage_b_glm_qc import GLM_MODEL_ALIAS  # noqa: E402
from mind.trajectory.stage_b_manifest import stream_stage_b_full_cache_entries  # noqa: E402
from mind.trajectory.stage_b_objectives import STAGE_B_ENCODER_FAMILY  # noqa: E402
from mind.trajectory.stage_b_training import score_stage_b_lstm, train_stage_b_lstm  # noqa: E402
from mind.trajectory.stage_b4_vmf import fit_single_vmf_support, score_single_vmf_support  # noqa: E402
from mind.trajectory.stage_c_support import (  # noqa: E402
    build_stage_c_radius_candidates,
    score_radius_ball_support,
)
from mind.trajectory.stage_d_manifest import (  # noqa: E402
    REQUIRED_STAGE_D_RATIO,
    REQUIRED_STAGE_D_SEEDS,
    STAGE_D_GLM_EXCLUSION_REASON,
    STAGE_D_MAIN_DETECTOR,
    STAGE_D_OBJECTIVE,
    STAGE_D_PARAM_DETECTOR,
    build_stage_d_preflight,
    load_stage_d_panel,
    validate_stage_d_plan,
)
from mind.trajectory.stage_d_protocols import (  # noqa: E402
    PRIMARY_STAGE_D_PROTOCOLS,
    STAGE_D_TIER_A_METHODS,
    build_stage_d_calibration_scopes,
    protocol_source_target,
    related_method_feasibility_payload,
    stage_d_baseline_tiers,
    validate_stage_d_protocols,
)
from mind.trajectory.stage_d_status import (  # noqa: E402
    build_stage_d_family_summary,
    domain_expansion_verdict,
    validate_stage_d_summary,
)


DATASET_OUTPUT_NAMES = {
    "repope": "repope_family_split_manifest.json",
    "pope": "pope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}
STAGE_D_METHODS = tuple(STAGE_D_TIER_A_METHODS)
LINEAR_C = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageD"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--protocols", nargs="+", default=list(PRIMARY_STAGE_D_PROTOCOLS))
    parser.add_argument("--ratio", type=float, default=REQUIRED_STAGE_D_RATIO)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REQUIRED_STAGE_D_SEEDS))
    parser.add_argument("--objective", default=STAGE_D_OBJECTIVE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-family", type=int, default=None)
    return parser


def _stage_d_output_paths(output_root: Path) -> dict[str, Path]:
    preflight_dir = output_root / "preflight"
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    return {
        "preflight_dir": preflight_dir,
        "manifest_dir": manifest_dir,
        "report_dir": report_dir,
        "preflight_json": preflight_dir / "stageD_preflight.json",
        "preflight_md": preflight_dir / "stageD_preflight.md",
        "repope_split_manifest": manifest_dir / DATASET_OUTPUT_NAMES["repope"],
        "pope_split_manifest": manifest_dir / DATASET_OUTPUT_NAMES["pope"],
        "dash_b_split_manifest": manifest_dir / DATASET_OUTPUT_NAMES["dash-b"],
        "cross_domain_metrics_long": report_dir / "cross_domain_metrics_long.csv",
        "cross_domain_primary_table": report_dir / "cross_domain_primary_table.csv",
        "cross_domain_oracle_recalibration_table": report_dir / "cross_domain_oracle_recalibration_table.csv",
        "domain_expansion_tierA": report_dir / "domain_expansion_tierA.csv",
        "domain_expansion_tierB": report_dir / "domain_expansion_tierB.csv",
        "related_method_feasibility_md": report_dir / "related_method_feasibility.md",
        "related_method_feasibility_json": report_dir / "related_method_feasibility.json",
        "model_family_summary": report_dir / "model_family_summary.csv",
        "model_family_notes": report_dir / "model_family_notes.md",
        "per_model_stageD_summary": report_dir / "per_model_stageD_summary.csv",
        "summary_json": report_dir / "STAGE_D_SUMMARY.json",
        "summary_md": report_dir / "STAGE_D_SUMMARY.md",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _stage_d_output_paths(args.output_root)
    for directory_key in ("preflight_dir", "manifest_dir", "report_dir"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    protocols = validate_stage_d_protocols(args.protocols)
    datasets = _datasets_for_protocols(protocols)
    plan = validate_stage_d_plan(
        ratio=float(args.ratio),
        seeds=args.seeds,
        objective=str(args.objective),
        encoder_family=STAGE_B_ENCODER_FAMILY,
        main_detector=STAGE_D_MAIN_DETECTOR,
        param_detector=STAGE_D_PARAM_DETECTOR,
    )
    manifest = load_stage_d_panel(args.full_cache_root)
    panel_models = [str(row["model_alias"]) for row in manifest.models]
    model_rows = _select_model_rows(manifest.models, requested=args.models)
    summary_panel = _build_stage_d_summary_panel(panel_models=panel_models, requested_models=args.models)
    split_maps = _build_splits(
        manifest.models[0],
        full_cache_root=args.full_cache_root,
        output_root=args.output_root,
        datasets=datasets,
        seed=REQUIRED_STAGE_D_SEEDS[0],
    )
    excluded_models: dict[str, str] = {}
    if GLM_MODEL_ALIAS in panel_models:
        excluded_models[GLM_MODEL_ALIAS] = STAGE_D_GLM_EXCLUSION_REASON
    preflight = build_stage_d_preflight(manifest, excluded_models=excluded_models, split_ready=True)
    preflight["plan"] = plan
    preflight["protocols"] = protocols
    preflight["baseline_tiers"] = stage_d_baseline_tiers()
    _write_json(paths["preflight_json"], preflight)
    paths["preflight_md"].write_text(_render_preflight_markdown(preflight), encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    model_failures: dict[str, str] = {}
    included_rows = [row for row in model_rows if str(row.get("model_alias", "")) not in excluded_models]
    device = _training_device(str(args.device))
    for index, model_row in enumerate(included_rows, start=1):
        model_alias = str(model_row["model_alias"])
        print(f"[{index}/{len(included_rows)}] Stage D model={model_alias}", flush=True)
        try:
            result = _run_model_stage_d(
                model_row,
                full_cache_root=args.full_cache_root,
                datasets=datasets,
                protocols=protocols,
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
            metric_rows.extend(_failed_metric_rows(model_alias, protocols, tuple(plan["seeds"]), reason))
            print(f"  failed: {reason}", flush=True)
            continue
        metric_rows.extend(result)

    skipped_models = {
        str(model): str(reason)
        for model, reason in dict(summary_panel["skipped_models"]).items()
    }
    panel_for_summary = [str(model) for model in summary_panel["panel_models"]]
    evaluated_models = sorted(
        {
            str(row["model_alias"])
            for row in metric_rows
            if str(row.get("metric_status", "")) in {"passed", "undefined"}
        }
    )
    separate_env_models = _separate_env_models(manifest.models)
    family_rows = build_stage_d_family_summary(
        panel_models=panel_for_summary,
        metric_rows=metric_rows,
        excluded_models=excluded_models,
        separate_env_models=separate_env_models,
    )
    per_model_rows = _per_model_summary(
        panel_models=panel_for_summary,
        metric_rows=metric_rows,
        excluded_models=excluded_models,
        failed_models=model_failures,
        skipped_models=skipped_models,
    )
    feasibility = related_method_feasibility_payload()
    domain_verdict = domain_expansion_verdict(metric_rows)

    write_csv_rows(paths["cross_domain_metrics_long"], metric_rows)
    write_csv_rows(paths["cross_domain_primary_table"], _filter_rows(metric_rows, calibration_scope="source_calibration"))
    write_csv_rows(
        paths["cross_domain_oracle_recalibration_table"],
        _filter_rows(metric_rows, calibration_scope="oracle_target_calibration"),
    )
    write_csv_rows(paths["domain_expansion_tierA"], _tier_a_rows(metric_rows))
    write_csv_rows(paths["domain_expansion_tierB"], _tier_b_rows())
    _write_json(paths["related_method_feasibility_json"], feasibility)
    paths["related_method_feasibility_md"].write_text(_render_feasibility_markdown(feasibility), encoding="utf-8")
    write_csv_rows(paths["model_family_summary"], family_rows)
    paths["model_family_notes"].write_text(_render_family_notes(family_rows), encoding="utf-8")
    write_csv_rows(paths["per_model_stageD_summary"], per_model_rows)

    summary = validate_stage_d_summary(
        {
            "stage": "stage_d",
            "stage_e_started": False,
            "method_redesigned": False,
            "full_cache_manifest": str(manifest.path),
            "panel_models": panel_for_summary,
            "evaluated_models": evaluated_models,
            "excluded_models": excluded_models,
            "failed_models": model_failures,
            "skipped_models": skipped_models,
            "objective": STAGE_D_OBJECTIVE,
            "encoder_family": STAGE_B_ENCODER_FAMILY,
            "negative_budget_ratio": REQUIRED_STAGE_D_RATIO,
            "negative_budget_seeds": list(REQUIRED_STAGE_D_SEEDS),
            "main_method": "MIND-main",
            "main_detector": STAGE_D_MAIN_DETECTOR,
            "param_method": "MIND-param",
            "param_detector": STAGE_D_PARAM_DETECTOR,
            "protocols": protocols,
            "domain_expansion_verdict": domain_verdict,
            "key_cross_domain_findings": _key_cross_domain_findings(metric_rows),
            "key_family_findings": _key_family_findings(family_rows),
            "preflight_path": str(paths["preflight_json"]),
            "cross_domain_metrics_long_path": str(paths["cross_domain_metrics_long"]),
            "cross_domain_primary_table_path": str(paths["cross_domain_primary_table"]),
            "cross_domain_oracle_recalibration_table_path": str(paths["cross_domain_oracle_recalibration_table"]),
            "domain_expansion_tierA_path": str(paths["domain_expansion_tierA"]),
            "domain_expansion_tierB_path": str(paths["domain_expansion_tierB"]),
            "related_method_feasibility_md_path": str(paths["related_method_feasibility_md"]),
            "related_method_feasibility_json_path": str(paths["related_method_feasibility_json"]),
            "model_family_summary_path": str(paths["model_family_summary"]),
            "model_family_notes_path": str(paths["model_family_notes"]),
            "per_model_stageD_summary_path": str(paths["per_model_stageD_summary"]),
        }
    )
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(_render_summary_markdown(summary), encoding="utf-8")
    print(
        f"Stage D summary={paths['summary_json']} "
        f"domain_expansion_verdict={summary['domain_expansion_verdict']}",
        flush=True,
    )
    return 0


def _run_model_stage_d(
    model_row: Mapping[str, object],
    *,
    full_cache_root: Path,
    datasets: Sequence[str],
    protocols: Sequence[str],
    split_maps: Mapping[str, Mapping[str, str]],
    ratio: float,
    seeds: Sequence[int],
    bootstrap: int,
    epochs: int,
    batch_size: int,
    device: str,
    limit_per_family: int | None,
) -> list[dict[str, object]]:
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
            "final_hidden": primary["final_hidden"],
            "halp_lite": primary["halp_lite"],
            "output_confidence": primary["output_confidence"],
        }
    source_datasets = sorted({protocol_source_target(protocol)[0] for protocol in protocols})
    metric_rows: list[dict[str, object]] = []
    for source_dataset in source_datasets:
        source = family_cache[source_dataset]
        train_mask = np.asarray(source["splits"]) == "encoder_train"
        train_labels_all = np.asarray(source["labels"], dtype=np.int64)[train_mask]
        train_trajectories_all = np.asarray(source["trajectories"], dtype=np.float32)[train_mask]
        num_layers = int(train_trajectories_all.shape[1])
        hidden_dim = int(train_trajectories_all.shape[2])
        train_correct_available = int(np.sum(train_labels_all == 0))
        train_hard_available = int(np.sum(train_labels_all == 1))
        for seed in seeds:
            selected_train_indices = subsample_stage_b2_training_indices(
                train_labels_all,
                ratio=float(ratio),
                seed=int(seed),
            )
            train_labels = train_labels_all[selected_train_indices]
            trained = train_stage_b_lstm(
                train_trajectories_all[selected_train_indices],
                train_labels,
                objective=STAGE_D_OBJECTIVE,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                seed=int(seed),
                patience=5,
            )
            embeddings_by_dataset: dict[str, np.ndarray] = {}
            for dataset, data in family_cache.items():
                embeddings, _ = score_stage_b_lstm(
                    trained.model,
                    np.asarray(data["trajectories"], dtype=np.float32),
                    batch_size=batch_size,
                )
                embeddings_by_dataset[dataset] = embeddings
            source_context = _build_source_context(
                family_cache=family_cache,
                embeddings_by_dataset=embeddings_by_dataset,
                source_dataset=source_dataset,
                selected_train_indices=selected_train_indices,
                seed=int(seed),
            )
            for protocol in protocols:
                protocol_source, target_dataset = protocol_source_target(protocol)
                if protocol_source != source_dataset:
                    continue
                for scope_row in build_stage_d_calibration_scopes(protocol):
                    metric_rows.extend(
                        _evaluate_protocol_scope(
                            model_alias=model_alias,
                            protocol=protocol,
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            calibration_scope=str(scope_row["calibration_scope"]),
                            diagnostic_only=bool(scope_row["diagnostic_only"]),
                            family_cache=family_cache,
                            embeddings_by_dataset=embeddings_by_dataset,
                            source_context=source_context,
                            bootstrap=bootstrap,
                            seed=int(seed),
                            ratio=float(ratio),
                            train_correct_available=train_correct_available,
                            train_hard_available=train_hard_available,
                            used_hard=int(np.sum(train_labels == 1)),
                        )
                    )
    return metric_rows


def _build_source_context(
    *,
    family_cache: Mapping[str, Mapping[str, object]],
    embeddings_by_dataset: Mapping[str, np.ndarray],
    source_dataset: str,
    selected_train_indices: np.ndarray,
    seed: int,
) -> dict[str, object]:
    source = family_cache[source_dataset]
    labels = np.asarray(source["labels"], dtype=np.int64)
    splits = np.asarray(source["splits"])
    embeddings = np.asarray(embeddings_by_dataset[source_dataset], dtype=np.float32)
    train_mask = splits == "encoder_train"
    train_labels = labels[train_mask][selected_train_indices]
    source_bank_mask = (splits == "bank") & (labels == 0)
    source_bank = embeddings[source_bank_mask]
    single_vmf = fit_single_vmf_support(source_bank)
    return {
        "source_labels": labels,
        "source_splits": splits,
        "source_embeddings": embeddings,
        "source_bank": source_bank,
        "single_vmf": single_vmf,
        "selected_train_indices": selected_train_indices,
        "train_labels": train_labels,
        "logistic_z_models": _fit_linear_models(
            embeddings[train_mask][selected_train_indices],
            train_labels,
            seed=seed,
        ),
        "final_hidden_models": _fit_linear_models(
            np.asarray(source["final_hidden"], dtype=np.float32)[train_mask][selected_train_indices],
            train_labels,
            seed=seed,
        ),
        "halp_lite_models": _fit_linear_models(
            np.asarray(source["halp_lite"], dtype=np.float32)[train_mask][selected_train_indices],
            train_labels,
            seed=seed,
        ),
    }


def _evaluate_protocol_scope(
    *,
    model_alias: str,
    protocol: str,
    source_dataset: str,
    target_dataset: str,
    calibration_scope: str,
    diagnostic_only: bool,
    family_cache: Mapping[str, Mapping[str, object]],
    embeddings_by_dataset: Mapping[str, np.ndarray],
    source_context: Mapping[str, object],
    bootstrap: int,
    seed: int,
    ratio: float,
    train_correct_available: int,
    train_hard_available: int,
    used_hard: int,
) -> list[dict[str, object]]:
    target = family_cache[target_dataset]
    target_embeddings = np.asarray(embeddings_by_dataset[target_dataset], dtype=np.float32)
    source_bank = np.asarray(source_context["source_bank"], dtype=np.float32)
    target_labels = np.asarray(target["labels"], dtype=np.int64)
    target_splits = np.asarray(target["splits"])
    calibration_dataset = source_dataset if calibration_scope == "source_calibration" else target_dataset
    calibration = family_cache[calibration_dataset]
    calibration_embeddings = np.asarray(embeddings_by_dataset[calibration_dataset], dtype=np.float32)
    selected_rho = _select_radius_from_calibration(
        bank_embeddings=source_bank,
        cal_embeddings=calibration_embeddings,
        cal_labels=np.asarray(calibration["labels"], dtype=np.int64),
        cal_splits=np.asarray(calibration["splits"]),
    )
    selected_c = _select_linear_c(
        models=source_context["logistic_z_models"],
        cal_features=calibration_embeddings,
        cal_labels=np.asarray(calibration["labels"], dtype=np.int64),
        cal_splits=np.asarray(calibration["splits"]),
    )
    selected_final_c = _select_linear_c(
        models=source_context["final_hidden_models"],
        cal_features=np.asarray(calibration["final_hidden"], dtype=np.float32),
        cal_labels=np.asarray(calibration["labels"], dtype=np.int64),
        cal_splits=np.asarray(calibration["splits"]),
    )
    selected_halp_c = _select_linear_c(
        models=source_context["halp_lite_models"],
        cal_features=np.asarray(calibration["halp_lite"], dtype=np.float32),
        cal_labels=np.asarray(calibration["labels"], dtype=np.int64),
        cal_splits=np.asarray(calibration["splits"]),
    )
    method_scores = {
        "MIND-main": score_radius_ball_support(
            bank_embeddings=source_bank,
            query_embeddings=target_embeddings,
            rho=selected_rho,
        ),
        "MIND-param": score_single_vmf_support(source_context["single_vmf"], target_embeddings),
        "logistic(z)": source_context["logistic_z_models"][selected_c].predict_proba(target_embeddings)[:, 1].astype(np.float32),
        "final-hidden linear probe": source_context["final_hidden_models"][selected_final_c]
        .predict_proba(np.asarray(target["final_hidden"], dtype=np.float32))[:, 1]
        .astype(np.float32),
        "output-confidence": np.asarray(target["output_confidence"], dtype=np.float32),
        "HALP-lite": source_context["halp_lite_models"][selected_halp_c]
        .predict_proba(np.asarray(target["halp_lite"], dtype=np.float32))[:, 1]
        .astype(np.float32),
    }
    selected_parameters = {
        "MIND-main": f"rho={selected_rho}",
        "MIND-param": "single_vmf",
        "logistic(z)": f"C={selected_c}",
        "final-hidden linear probe": f"C={selected_final_c}",
        "output-confidence": "negative_top2_margin",
        "HALP-lite": f"C={selected_halp_c}",
    }
    rows: list[dict[str, object]] = []
    for method, scores in method_scores.items():
        rows.append(
            _metric_row(
                model_alias=model_alias,
                protocol=protocol,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                method=method,
                calibration_scope=calibration_scope,
                diagnostic_only=diagnostic_only,
                labels=target_labels,
                splits=target_splits,
                scores=scores,
                all_entries=target["all_entries"],
                bootstrap=bootstrap,
                seed=seed,
                ratio=ratio,
                selected_parameter=selected_parameters[method],
                train_correct_available=train_correct_available,
                train_hard_available=train_hard_available,
                used_hard=used_hard,
            )
        )
    return rows


def _select_radius_from_calibration(
    *,
    bank_embeddings: np.ndarray,
    cal_embeddings: np.ndarray,
    cal_labels: np.ndarray,
    cal_splits: np.ndarray,
) -> float:
    cal_mask = cal_splits == "cal"
    cal_correct_mask = cal_mask & (cal_labels == 0)
    candidates = build_stage_c_radius_candidates(
        bank_embeddings=bank_embeddings,
        cal_embeddings=cal_embeddings[cal_correct_mask],
        cal_labels=np.zeros(int(cal_correct_mask.sum()), dtype=np.int64),
    )
    y = cal_labels[cal_mask]
    rows: list[tuple[float, float, float]] = []
    for candidate in candidates:
        rho = float(candidate["rho"])
        scores = score_radius_ball_support(
            bank_embeddings=bank_embeddings,
            query_embeddings=cal_embeddings[cal_mask],
            rho=rho,
        )
        metrics = binary_diagnostic_metrics(y, scores) if np.unique(y).size >= 2 else {"pr_auc": float("nan"), "roc_auc": float("nan")}
        rows.append((float(metrics["pr_auc"]), float(metrics["roc_auc"]), rho))
    rows = [row for row in rows if np.isfinite(row[0]) and np.isfinite(row[1])]
    if not rows:
        return float(candidates[0]["rho"])
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return float(rows[0][2])


def _fit_linear_models(features: np.ndarray, labels: np.ndarray, *, seed: int) -> dict[float, Pipeline]:
    if np.unique(labels).size < 2:
        raise ValueError("linear baseline labels must contain both classes")
    models: dict[float, Pipeline] = {}
    for c_value in (0.1, 1.0, 10.0):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(c_value),
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=int(seed),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64))
        models[float(c_value)] = model
    return models


def _select_linear_c(
    *,
    models: Mapping[float, Pipeline],
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    cal_splits: np.ndarray,
) -> float:
    mask = cal_splits == "cal"
    y = cal_labels[mask]
    rows: list[tuple[float, float, float]] = []
    for c_value, model in models.items():
        scores = model.predict_proba(np.asarray(cal_features, dtype=np.float32)[mask])[:, 1].astype(np.float32)
        metrics = binary_diagnostic_metrics(y, scores) if np.unique(y).size >= 2 else {"pr_auc": float("nan"), "roc_auc": float("nan")}
        rows.append((float(metrics["pr_auc"]), float(metrics["roc_auc"]), float(c_value)))
    rows = [row for row in rows if np.isfinite(row[0]) and np.isfinite(row[1])]
    if not rows:
        return LINEAR_C
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return float(rows[0][2])


def _metric_row(
    *,
    model_alias: str,
    protocol: str,
    source_dataset: str,
    target_dataset: str,
    method: str,
    calibration_scope: str,
    diagnostic_only: bool,
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
    all_entries: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
    ratio: float,
    selected_parameter: str,
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
        "protocol": protocol,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "method": method,
        "tier": "tierA",
        "training_budget": "50pct_hard_negatives" if method in {"MIND-main", "MIND-param", "logistic(z)"} else "same_constraint_linear_or_cached",
        "calibration_scope": calibration_scope,
        "diagnostic_only": bool(diagnostic_only),
        "selected_parameter": selected_parameter,
        "metric_status": "undefined" if undefined_reason else "passed",
        "failure_reason": undefined_reason,
        "negative_budget_ratio": float(ratio),
        "negative_budget_seed": int(seed),
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
        "num_encoder_train_correct": int(train_correct_available),
        "num_encoder_train_hard_hallucination": int(used_hard),
        "num_encoder_train_hard_hallucination_available": int(train_hard_available),
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
        manifest["stage"] = "stage_d"
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
            raise ValueError(f"missing Stage D split for image_id={image_id} family={dataset_family}")
        row = dict(entry)
        row["stage_d_split"] = split
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def _primary_family_data(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    primary_entries: list[dict[str, object]] = []
    labels: list[int] = []
    splits: list[str] = []
    output_scores: list[float] = []
    for entry in entries:
        stage_row = dict(entry)
        stage_row["stage_b_split"] = stage_row["stage_d_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.CORRECT:
            primary_entries.append(dict(entry))
            labels.append(0)
            splits.append(str(entry["stage_d_split"]))
            output_scores.append(_output_confidence_score(entry))
        elif population == PopulationClass.HARD_HALLUCINATION:
            primary_entries.append(dict(entry))
            labels.append(1)
            splits.append(str(entry["stage_d_split"]))
            output_scores.append(_output_confidence_score(entry))
    if not primary_entries:
        raise ValueError("no Stage D primary population rows")
    trajectories = np.stack([build_lstm_trajectory(row) for row in primary_entries], axis=0).astype(np.float32, copy=False)
    final_hidden = np.stack([build_representation(row, "Raw-Static") for row in primary_entries], axis=0).astype(np.float32, copy=False)
    halp_lite = np.stack(
        [
            np.concatenate(
                [
                    build_representation(row, "Sphere-Static"),
                    build_representation(row, "Sphere-Traj-MeanPool"),
                ],
                axis=0,
            )
            for row in primary_entries
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return {
        "entries": primary_entries,
        "labels": np.asarray(labels, dtype=np.int64),
        "splits": np.asarray(splits),
        "trajectories": trajectories,
        "final_hidden": final_hidden,
        "halp_lite": halp_lite,
        "output_confidence": np.asarray(output_scores, dtype=np.float32),
    }


def _output_confidence_score(entry: Mapping[str, object]) -> float:
    logits = entry.get("first_token_logits")
    if logits is None:
        return 0.0
    if isinstance(logits, torch.Tensor):
        values = logits.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1)
    else:
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        return 0.0
    top2 = np.partition(values, -2)[-2:]
    margin = float(np.max(top2) - np.min(top2))
    return -margin


def _excluded_counts(entries: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    counts = {"false_negative": 0, "parsed_none": 0}
    for row in entries:
        if str(row.get("stage_d_split", "")) != split:
            continue
        stage_row = dict(row)
        stage_row["stage_b_split"] = stage_row["stage_d_split"]
        population = classify_entry(stage_row)
        if population == PopulationClass.FALSE_NEGATIVE_ERROR:
            counts["false_negative"] += 1
        elif population == PopulationClass.PARSED_NONE:
            counts["parsed_none"] += 1
    return counts


def _datasets_for_protocols(protocols: Sequence[str]) -> list[str]:
    datasets = sorted({dataset for protocol in protocols for dataset in protocol_source_target(protocol)})
    return [dataset for dataset in ("repope", "pope", "dash-b") if dataset in datasets]


def _select_model_rows(model_rows: Sequence[Mapping[str, object]], *, requested: Sequence[str] | None) -> list[dict[str, object]]:
    if not requested:
        return [dict(row) for row in model_rows]
    requested_set = {str(model) for model in requested}
    selected = [dict(row) for row in model_rows if str(row.get("model_alias", "")) in requested_set]
    found = {str(row["model_alias"]) for row in selected}
    missing = sorted(requested_set - found)
    if missing:
        raise SystemExit("requested Stage D models not found in unified manifest: " + ", ".join(missing))
    return selected


def _build_stage_d_summary_panel(*, panel_models: Sequence[str], requested_models: Sequence[str] | None) -> dict[str, object]:
    panel = [str(model) for model in panel_models]
    if not requested_models:
        return {"panel_models": panel, "skipped_models": {}}
    requested = {str(model) for model in requested_models}
    skipped = {
        model: "not requested in this run"
        for model in panel
        if model not in requested and model != GLM_MODEL_ALIAS
    }
    return {"panel_models": panel, "skipped_models": skipped}


def _separate_env_models(model_rows: Sequence[Mapping[str, object]]) -> set[str]:
    output: set[str] = set()
    for row in model_rows:
        status = str(row.get("status", ""))
        origin = str(row.get("cache_origin", ""))
        env = str(row.get("extraction_env_name", ""))
        if "separate" in status or "separate" in origin or env in {"mind-gemma4-py311", "mind-molmo-py311"}:
            output.add(str(row.get("model_alias", "")))
    return output


def _failed_metric_rows(model_alias: str, protocols: Sequence[str], seeds: Sequence[int], reason: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for protocol in protocols:
            source, target = protocol_source_target(protocol)
            for method in STAGE_D_METHODS:
                rows.append(
                    {
                        "model_alias": model_alias,
                        "protocol": protocol,
                        "source_dataset": source,
                        "target_dataset": target,
                        "method": method,
                        "tier": "tierA",
                        "calibration_scope": "source_calibration",
                        "diagnostic_only": False,
                        "metric_status": "failed",
                        "failure_reason": reason,
                        "negative_budget_ratio": REQUIRED_STAGE_D_RATIO,
                        "negative_budget_seed": int(seed),
                        **_undefined_metrics(),
                    }
                )
    return rows


def _filter_rows(rows: Sequence[Mapping[str, object]], *, calibration_scope: str) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("calibration_scope", "")) == calibration_scope]


def _tier_a_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("calibration_scope", "")) == "source_calibration"
        and str(row.get("method", "")) in STAGE_D_TIER_A_METHODS
    ]


def _tier_b_rows() -> list[dict[str, object]]:
    return [
        {
            "method": "official HALP",
            "tier": "tierB",
            "status": "infeasible_without_external_reproduction",
            "label": "ceiling_broader_access",
            "reason": "official HALP is broader-access and not part of the primary fair-comparison table",
        }
    ]


def _per_model_summary(
    *,
    panel_models: Sequence[str],
    metric_rows: Sequence[Mapping[str, object]],
    excluded_models: Mapping[str, object],
    failed_models: Mapping[str, object],
    skipped_models: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in panel_models:
        if model in excluded_models:
            rows.append({"model_alias": model, "status": "excluded", "reason": excluded_models[model]})
            continue
        if model in failed_models:
            rows.append({"model_alias": model, "status": "failed", "reason": failed_models[model]})
            continue
        if model in skipped_models:
            rows.append({"model_alias": model, "status": "skipped", "reason": skipped_models[model]})
            continue
        model_rows = [
            row
            for row in metric_rows
            if str(row.get("model_alias")) == model
            and str(row.get("method")) == "MIND-main"
            and str(row.get("metric_status")) == "passed"
            and str(row.get("calibration_scope")) == "source_calibration"
        ]
        mean_pr = float(np.mean([float(row["pr_auc"]) for row in model_rows])) if model_rows else float("nan")
        rows.append({"model_alias": model, "status": "evaluated", "mind_main_mean_pr_auc": mean_pr, "reason": ""})
    return rows


def _key_cross_domain_findings(rows: Sequence[Mapping[str, object]]) -> str:
    source_rows = [
        row
        for row in rows
        if str(row.get("method")) == "MIND-main"
        and str(row.get("calibration_scope")) == "source_calibration"
        and str(row.get("metric_status")) == "passed"
    ]
    if not source_rows:
        return "no passed MIND-main rows"
    grouped: dict[str, list[float]] = {}
    for row in source_rows:
        grouped.setdefault(str(row["protocol"]), []).append(float(row["pr_auc"]))
    return "; ".join(f"{protocol}:{float(np.mean(values)):.6f}" for protocol, values in sorted(grouped.items()))


def _key_family_findings(rows: Sequence[Mapping[str, object]]) -> str:
    return "; ".join(
        f"{row['family']}:{row['num_evaluable_models']}/{row['num_panel_models']}"
        for row in rows
    )


def _render_preflight_markdown(preflight: Mapping[str, object]) -> str:
    lines = [
        "# Stage D Preflight",
        "",
        "Stage D evaluates the frozen MIND-main method.",
        "",
        f"- total_panel_models: {preflight.get('total_panel_models')}",
        f"- evaluable_models: {preflight.get('evaluable_models')}",
        f"- frozen_main_method: {preflight.get('frozen_main_method')}",
        f"- fixed_negative_budget_ratio: {preflight.get('fixed_negative_budget_ratio')}",
        f"- stage_e_started: {str(preflight.get('stage_e_started', False)).lower()}",
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


def _render_feasibility_markdown(payload: Mapping[str, object]) -> str:
    lines = ["# Related Method Feasibility", ""]
    for row in payload.get("methods", []):
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                f"## {row.get('method')}",
                "",
                f"- detection_granularity: {row.get('detection_granularity')}",
                f"- required_supervision: {row.get('required_supervision')}",
                f"- required_access_type: {row.get('required_access_type')}",
                f"- generation_timing: {row.get('generation_timing')}",
                f"- method_type: {row.get('method_type')}",
                f"- executable_with_current_cache: {str(row.get('executable_with_current_cache')).lower()}",
                f"- incompatibility_reason: {row.get('incompatibility_reason')}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_family_notes(rows: Sequence[Mapping[str, object]]) -> str:
    lines = ["# Stage D Model Family Notes", ""]
    for row in rows:
        lines.append(
            f"- {row['family']}: {row['num_evaluable_models']}/{row['num_panel_models']} evaluable. "
            f"{row.get('family_specific_notes', '')} {row.get('main_env_vs_separate_env_note', '')}".strip()
        )
    return "\n".join(lines) + "\n"


def _render_summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Stage D Summary",
            "",
            "Stage D evaluates the frozen method. It does not redesign the method.",
            "",
            f"- domain_expansion_verdict: {summary.get('domain_expansion_verdict')}",
            f"- stage_e_started: {str(summary.get('stage_e_started', False)).lower()}",
            f"- method_redesigned: {str(summary.get('method_redesigned', False)).lower()}",
            f"- key_cross_domain_findings: {summary.get('key_cross_domain_findings')}",
            f"- key_family_findings: {summary.get('key_family_findings')}",
            "",
            "## Excluded Models",
            "",
            *[
                f"- {model}: {reason}"
                for model, reason in dict(summary.get("excluded_models", {}) or {}).items()
            ],
        ]
    ) + "\n"


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


if __name__ == "__main__":
    raise SystemExit(main())
