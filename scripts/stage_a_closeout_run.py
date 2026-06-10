#!/usr/bin/env python3
"""Run Stage A closeout diagnostics on the Experiment 2 full-cache panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
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
    CLOSEOUT_READOUTS,
    CLOSEOUT_VARIANTS,
    FAMILY_SUBSETS,
    build_closeout_family_split,
    decide_sphere_closeout_verdict,
    load_closeout_panel_manifest,
    population_audit_row,
    render_closeout_summary_markdown,
    stream_full_cache_entries,
    summarize_closeout_status,
    write_csv_rows,
    write_split_manifest,
)
from mind.trajectory.stage_a_metrics import binary_diagnostic_metrics, bootstrap_binary_metrics  # noqa: E402
from mind.trajectory.stage_a_population import PopulationClass, classify_entry  # noqa: E402
from mind.trajectory.stage_a_readouts import (  # noqa: E402
    compute_knn_scores,
    train_logistic_diagnostic,
    train_lstm_diagnostic,
)
from mind.trajectory.stage_a_representations import (  # noqa: E402
    DEFAULT_STAGE_A_SEED,
    build_lstm_trajectory,
    build_raw_lstm_trajectory,
    build_representation_matrix,
    set_deterministic_seed,
)


SPLIT_OUTPUT_NAMES = {
    "pope": "pope_family_split_manifest.json",
    "repope": "repope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}
PRIMARY_VARIANTS = (
    "Raw-Static",
    "Sphere-Static",
    "Raw-Traj-MeanPool",
    "Sphere-Traj-MeanPool",
)
LSTM_VARIANTS = ("Raw-Traj-LSTM", "Sphere-Traj-LSTM")
SPLITS = ("encoder_train", "bank", "cal", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageA_closeout"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["repope", "pope", "dash-b"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_STAGE_A_SEED)
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--skip-lstm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_deterministic_seed(args.seed)
    output_root: Path = args.output_root
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    panel = load_closeout_panel_manifest(args.full_cache_root)
    panel_models = [str(row["model_alias"]) for row in panel.models]
    requested = set(args.models) if args.models else set(panel_models)
    model_rows = [row for row in panel.models if str(row["model_alias"]) in requested]
    if args.models and len(model_rows) != len(requested):
        found = {str(row["model_alias"]) for row in model_rows}
        missing = sorted(requested - found)
        raise SystemExit("requested models not found in unified manifest: " + ", ".join(missing))

    datasets = _validate_datasets(args.datasets)
    splits_by_family = _build_or_load_splits(
        model_rows,
        full_cache_root=args.full_cache_root,
        output_root=output_root,
        datasets=datasets,
        seed=args.seed,
        limit_per_family=args.limit_per_family,
    )

    metric_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for model_row in model_rows:
        model_name = str(model_row["model_alias"])
        try:
            for family in datasets:
                entries = _load_family_entries(
                    model_row,
                    args.full_cache_root,
                    family=family,
                    split_manifest=splits_by_family[family],
                    limit_per_family=args.limit_per_family,
                )
                balance_rows.extend(_balance_rows(model_name, family, entries))
                population_rows.extend(_population_rows(model_name, family, entries))
                primary = [
                    row
                    for row in entries
                    if classify_entry(row)
                    in {PopulationClass.CORRECT, PopulationClass.HARD_HALLUCINATION}
                ]
                if not primary:
                    raise ValueError(f"{model_name}/{family} has no primary closeout population")
                metric_rows.extend(
                    _run_family_metrics(
                        model_name=model_name,
                        dataset_family=family,
                        entries=primary,
                        all_entries=entries,
                        bootstrap=args.bootstrap,
                        seed=args.seed,
                        device=args.device,
                        lstm_epochs=args.lstm_epochs,
                        knn_k=args.knn_k,
                        skip_lstm=args.skip_lstm,
                    )
                )
        except Exception as error:  # noqa: BLE001 - must record failed panel models.
            failures[model_name] = str(error)
            print(f"model failed: {model_name}: {error}", file=sys.stderr)

    report_dir = output_root / "reports"
    audit_dir = output_root / "audit"
    metrics_path = report_dir / "closeout_metrics_long.csv"
    write_csv_rows(metrics_path, metric_rows)
    write_csv_rows(audit_dir / "cache_label_balance.csv", balance_rows)
    write_csv_rows(audit_dir / "closeout_population_audit.csv", population_rows)

    table_paths = _write_tables(report_dir, metric_rows)
    per_model_path = report_dir / "per_model_summary.csv"
    per_model_rows = _per_model_summary_rows(metric_rows, failures)
    write_csv_rows(per_model_path, per_model_rows)

    verdict = decide_sphere_closeout_verdict(metric_rows)
    status = summarize_closeout_status(
        panel_models=[str(row["model_alias"]) for row in model_rows],
        metric_rows=metric_rows,
        failures=failures,
    )
    summary = {
        "stage": "stage_a_closeout",
        "stage_a_closed": True,
        "stage_b_started": False,
        "full_cache_manifest": str(panel.path),
        "panel_models": [str(row["model_alias"]) for row in model_rows],
        "datasets": datasets,
        "variants": list(CLOSEOUT_VARIANTS),
        "readouts": list(CLOSEOUT_READOUTS),
        "sphere_closeout_verdict": verdict,
        "failed_models": status["failed_models"],
        "evaluated_models": status["evaluated_models"],
        "missing_models": status["missing_models"],
        "all_panel_models_present": status["all_panel_models_present"],
        "metrics_long_path": str(metrics_path),
        "repope_classifier_table_path": str(table_paths["repope_classifier"]),
        "repope_knn_table_path": str(table_paths["repope_knn"]),
        "pope_secondary_table_path": str(table_paths["pope_secondary"]),
        "dash_b_secondary_table_path": str(table_paths["dash_b_secondary"]),
        "per_model_summary_path": str(per_model_path),
        "notes": [
            "Stage A closeout does not validate the final MIND detector.",
            "Stage B was not started.",
        ],
    }
    summary_path = report_dir / "STAGE_A_CLOSEOUT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (report_dir / "STAGE_A_CLOSEOUT_SUMMARY.md").write_text(
        render_closeout_summary_markdown(summary),
        encoding="utf-8",
    )
    print(f"Stage A closeout complete summary={summary_path} verdict={verdict['verdict']}")
    return 0 if not failures else 2


def _validate_datasets(values: Sequence[str]) -> list[str]:
    datasets = [str(value) for value in values]
    unsupported = sorted(set(datasets) - set(FAMILY_SUBSETS))
    if unsupported:
        raise SystemExit("unsupported closeout datasets: " + ", ".join(unsupported))
    return datasets


def _build_or_load_splits(
    model_rows: Sequence[Mapping[str, object]],
    *,
    full_cache_root: Path,
    output_root: Path,
    datasets: Sequence[str],
    seed: int,
    limit_per_family: int | None,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    split_source_model = model_rows[0]
    for family in datasets:
        rows = list(
            stream_full_cache_entries(
                split_source_model,
                full_cache_root,
                dataset_family=family,
                include_tensors=False,
            )
        )
        if limit_per_family is not None:
            rows = rows[: int(limit_per_family)]
        manifest = build_closeout_family_split(rows, family=family, seed=seed)
        manifest["split_source_model"] = split_source_model["model_alias"]
        manifest["split_application"] = "image_id assignments are applied to every panel model"
        output_path = output_root / "manifests" / SPLIT_OUTPUT_NAMES[family]
        write_split_manifest(manifest, output_path)
        result[family] = manifest
    return result


def _load_family_entries(
    model_row: Mapping[str, object],
    full_cache_root: Path,
    *,
    family: str,
    split_manifest: Mapping[str, object],
    limit_per_family: int | None,
) -> list[dict[str, object]]:
    split_by_image = {
        str(row["image_id"]): str(row["split"])
        for row in split_manifest.get("assignments", [])
        if isinstance(row, Mapping)
    }
    rows = []
    for entry in stream_full_cache_entries(
        model_row,
        full_cache_root,
        dataset_family=family,
        include_tensors=True,
    ):
        if limit_per_family is not None and len(rows) >= int(limit_per_family):
            break
        image_id = str(entry.get("image_id", ""))
        split = split_by_image.get(image_id)
        if split is None:
            raise ValueError(f"missing closeout split for image_id={image_id}")
        entry["stage_a_split"] = split
        rows.append(entry)
    return rows


def _run_family_metrics(
    *,
    model_name: str,
    dataset_family: str,
    entries: Sequence[Mapping[str, object]],
    all_entries: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
    device: str,
    lstm_epochs: int,
    knn_k: int,
    skip_lstm: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = _labels(entries)
    splits = _splits(entries)
    train_mask = splits == "encoder_train"
    if np.unique(labels[train_mask]).size < 2:
        raise ValueError(f"{model_name}/{dataset_family} encoder_train is missing one class")
    for variant in PRIMARY_VARIANTS:
        features = build_representation_matrix(entries, variant).values
        classifier = train_logistic_diagnostic(
            features[train_mask],
            labels[train_mask],
            seed=seed,
            max_iter=1000,
        )
        classifier_scores = classifier.model.predict_proba(features)[:, 1].astype(np.float32)
        rows.append(
            _metric_row(
                model_name,
                dataset_family,
                entries,
                all_entries,
                labels,
                splits,
                classifier_scores,
                variant=variant,
                readout="Diag-Classifier",
                bootstrap=bootstrap,
                seed=seed,
                num_bank_correct=_bank_correct_count(entries),
            )
        )
        bank_mask = (splits == "bank") & _correct_mask(entries)
        metric = "angular" if variant.startswith("Sphere-") else "euclidean"
        knn_scores = _knn_scores(features[bank_mask], features, k=knn_k, metric=metric, device=device)
        rows.append(
            _metric_row(
                model_name,
                dataset_family,
                entries,
                all_entries,
                labels,
                splits,
                knn_scores,
                variant=variant,
                readout="Diag-KNN",
                bootstrap=bootstrap,
                seed=seed,
                num_bank_correct=int(bank_mask.sum()),
            )
        )
    if skip_lstm:
        return rows
    for variant in LSTM_VARIANTS:
        trajectories = _lstm_trajectories(entries, variant)
        num_layers = int(trajectories.shape[1])
        hidden_dim = int(trajectories.shape[2])
        result = train_lstm_diagnostic(
            trajectories[train_mask],
            labels[train_mask],
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            epochs=lstm_epochs,
            batch_size=128,
            device=_training_device(device),
            seed=seed,
            patience=3,
        )
        scores, embeddings = _score_lstm(result.model, trajectories)
        rows.append(
            _metric_row(
                model_name,
                dataset_family,
                entries,
                all_entries,
                labels,
                splits,
                scores,
                variant=variant,
                readout="Diag-Classifier",
                bootstrap=bootstrap,
                seed=seed,
                num_bank_correct=_bank_correct_count(entries),
            )
        )
        bank_mask = (splits == "bank") & _correct_mask(entries)
        embeddings = _l2_normalize(embeddings)
        knn_scores = _knn_scores(embeddings[bank_mask], embeddings, k=knn_k, metric="angular", device=device)
        rows.append(
            _metric_row(
                model_name,
                dataset_family,
                entries,
                all_entries,
                labels,
                splits,
                knn_scores,
                variant=variant,
                readout="Diag-KNN",
                bootstrap=bootstrap,
                seed=seed,
                num_bank_correct=int(bank_mask.sum()),
            )
        )
    return rows


def _metric_row(
    model_name: str,
    dataset_family: str,
    entries: Sequence[Mapping[str, object]],
    all_entries: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
    *,
    variant: str,
    readout: str,
    bootstrap: int,
    seed: int,
    num_bank_correct: int,
) -> dict[str, object]:
    mask = splits == "test"
    y = labels[mask]
    split_scores = scores[mask]
    undefined_reason = ""
    if y.size == 0:
        undefined_reason = "no samples in test split"
    elif np.unique(y).size < 2:
        undefined_reason = "one class present in test split"
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
        "model_name": model_name,
        "dataset_family": dataset_family,
        "variant": variant,
        "readout": readout,
        "eval_split": "test",
        "eval_scope": "pooled",
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


def _write_tables(report_dir: Path, metric_rows: Sequence[Mapping[str, object]]) -> dict[str, Path]:
    paths = {
        "repope_classifier": report_dir / "repope_main_table_classifier.csv",
        "repope_knn": report_dir / "repope_main_table_knn.csv",
        "pope_secondary": report_dir / "pope_secondary_table.csv",
        "dash_b_secondary": report_dir / "dash_b_secondary_table.csv",
    }
    write_csv_rows(
        paths["repope_classifier"],
        [
            row
            for row in metric_rows
            if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-Classifier"
        ],
    )
    write_csv_rows(
        paths["repope_knn"],
        [
            row
            for row in metric_rows
            if row.get("dataset_family") == "repope" and row.get("readout") == "Diag-KNN"
        ],
    )
    write_csv_rows(paths["pope_secondary"], [row for row in metric_rows if row.get("dataset_family") == "pope"])
    write_csv_rows(paths["dash_b_secondary"], [row for row in metric_rows if row.get("dataset_family") == "dash-b"])
    return paths


def _per_model_summary_rows(
    metric_rows: Sequence[Mapping[str, object]],
    failures: Mapping[str, str],
) -> list[dict[str, object]]:
    rows = []
    by_model: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_model[str(row["model_name"])].append(row)
    for model_name, values in sorted(by_model.items()):
        repope = [
            row
            for row in values
            if row["dataset_family"] == "repope"
            and row["readout"] == "Diag-Classifier"
            and row["variant"] in {"Raw-Traj-LSTM", "Sphere-Traj-LSTM"}
        ]
        raw = next((row for row in repope if row["variant"] == "Raw-Traj-LSTM"), None)
        sphere = next((row for row in repope if row["variant"] == "Sphere-Traj-LSTM"), None)
        rows.append(
            {
                "model_name": model_name,
                "status": "evaluated",
                "repope_raw_traj_lstm_pr_auc": "" if raw is None else raw.get("pr_auc", ""),
                "repope_sphere_traj_lstm_pr_auc": "" if sphere is None else sphere.get("pr_auc", ""),
                "repope_sphere_minus_raw_pr_auc": ""
                if raw is None or sphere is None
                else float(sphere["pr_auc"]) - float(raw["pr_auc"]),
                "failure_reason": "",
            }
        )
    for model_name, reason in sorted(failures.items()):
        rows.append(
            {
                "model_name": model_name,
                "status": "failed",
                "repope_raw_traj_lstm_pr_auc": "",
                "repope_sphere_traj_lstm_pr_auc": "",
                "repope_sphere_minus_raw_pr_auc": "",
                "failure_reason": reason,
            }
        )
    return rows


def _balance_rows(
    model_name: str,
    family: str,
    entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [population_audit_row(entries, model_name=model_name, dataset_family=family)]
    by_subset: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for entry in entries:
        by_subset[str(entry.get("subset", ""))].append(entry)
    for subset, values in sorted(by_subset.items()):
        rows.append(
            population_audit_row(
                values,
                model_name=model_name,
                dataset_family=family,
                subset=subset,
            )
        )
    return rows


def _population_rows(
    model_name: str,
    family: str,
    entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    by_split: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for entry in entries:
        by_split[str(entry.get("stage_a_split", ""))].append(entry)
    for split, values in sorted(by_split.items()):
        rows.append(
            population_audit_row(
                values,
                model_name=model_name,
                dataset_family=family,
                split=split,
            )
        )
    return rows


def _lstm_trajectories(entries: Sequence[Mapping[str, object]], variant: str) -> np.ndarray:
    builder = build_raw_lstm_trajectory if variant == "Raw-Traj-LSTM" else build_lstm_trajectory
    return np.stack([builder(row) for row in entries], axis=0).astype(np.float32, copy=False)


def _score_lstm(
    model: torch.nn.Module,
    trajectories: np.ndarray,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
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


def _knn_scores(
    bank: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
    metric: str,
    device: str,
) -> np.ndarray:
    if bank.shape[0] < k:
        raise ValueError(f"bank has fewer than {k} correct samples")
    backend = "torch" if _cuda_available(device) else "numpy"
    return compute_knn_scores(
        bank,
        query,
        k=k,
        metric=metric,
        backend=backend,
        device=device if backend == "torch" else None,
        chunk_size=4096,
    )


def _training_device(device: str) -> str:
    return device if _cuda_available(device) else "cpu"


def _cuda_available(device: str) -> bool:
    return (not device.startswith("cuda")) or torch.cuda.is_available()


def _labels(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [1 if classify_entry(row) == PopulationClass.HARD_HALLUCINATION else 0 for row in entries],
        dtype=np.int64,
    )


def _splits(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([str(row["stage_a_split"]) for row in entries])


def _correct_mask(entries: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([classify_entry(row) == PopulationClass.CORRECT for row in entries], dtype=bool)


def _bank_correct_count(entries: Sequence[Mapping[str, object]]) -> int:
    return int(sum(1 for row in entries if row["stage_a_split"] == "bank" and classify_entry(row) == PopulationClass.CORRECT))


def _excluded_counts(entries: Sequence[Mapping[str, object]], *, split: str) -> dict[str, int]:
    counts = {"false_negative": 0, "parsed_none": 0}
    for row in entries:
        if str(row.get("stage_a_split", "")) != split:
            continue
        cls = classify_entry(row)
        if cls == PopulationClass.FALSE_NEGATIVE_ERROR:
            counts["false_negative"] += 1
        elif cls == PopulationClass.PARSED_NONE:
            counts["parsed_none"] += 1
    return counts


def _undefined_metrics() -> dict[str, float]:
    return {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "tpr_at_1pct_fpr": float("nan"),
        "fpr_at_95pct_tpr": float("nan"),
    }


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


if __name__ == "__main__":
    raise SystemExit(main())
