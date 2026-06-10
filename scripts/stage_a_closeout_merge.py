#!/usr/bin/env python3
"""Merge parallel Stage A closeout shard outputs into one closeout root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) in sys.path:
    sys.path.remove(str(REPO_SRC))
sys.path.insert(0, str(REPO_SRC))

from mind.trajectory.stage_a_closeout import (  # noqa: E402
    CLOSEOUT_READOUTS,
    CLOSEOUT_VARIANTS,
    decide_sphere_closeout_verdict,
    render_closeout_summary_markdown,
    summarize_closeout_status,
    write_csv_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageA_closeout"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root: Path = args.output_root
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    metric_rows = _read_many(args.input_roots, "reports/closeout_metrics_long.csv")
    balance_rows = _read_many(args.input_roots, "audit/cache_label_balance.csv")
    population_rows = _read_many(args.input_roots, "audit/closeout_population_audit.csv")
    per_model_rows = _read_many(args.input_roots, "reports/per_model_summary.csv")
    summaries = [_read_json(root / "reports" / "STAGE_A_CLOSEOUT_SUMMARY.json") for root in args.input_roots]

    for name in (
        "pope_family_split_manifest.json",
        "repope_family_split_manifest.json",
        "dash_b_split_manifest.json",
    ):
        source = _first_existing(root / "manifests" / name for root in args.input_roots)
        if source is not None:
            shutil.copy2(source, output_root / "manifests" / name)

    metrics_path = output_root / "reports" / "closeout_metrics_long.csv"
    write_csv_rows(metrics_path, metric_rows)
    write_csv_rows(output_root / "audit" / "cache_label_balance.csv", balance_rows)
    write_csv_rows(output_root / "audit" / "closeout_population_audit.csv", population_rows)
    write_csv_rows(output_root / "reports" / "per_model_summary.csv", per_model_rows)
    table_paths = _write_tables(output_root / "reports", metric_rows)

    panel_models: list[str] = []
    failures: dict[str, str] = {}
    datasets: list[str] = []
    for summary in summaries:
        for model in summary.get("panel_models", []):
            text = str(model)
            if text not in panel_models:
                panel_models.append(text)
        for dataset in summary.get("datasets", []):
            text = str(dataset)
            if text not in datasets:
                datasets.append(text)
        failed = summary.get("failed_models", {})
        if isinstance(failed, Mapping):
            failures.update({str(model): str(reason) for model, reason in failed.items()})

    status = summarize_closeout_status(
        panel_models=panel_models,
        metric_rows=metric_rows,
        failures=failures,
    )
    verdict = decide_sphere_closeout_verdict(metric_rows)
    summary = {
        "stage": "stage_a_closeout",
        "stage_a_closed": True,
        "stage_b_started": False,
        "panel_models": panel_models,
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
        "per_model_summary_path": str(output_root / "reports" / "per_model_summary.csv"),
        "merged_input_roots": [str(root) for root in args.input_roots],
        "notes": [
            "Stage A closeout does not validate the final MIND detector.",
            "Stage B was not started.",
        ],
    }
    summary_path = output_root / "reports" / "STAGE_A_CLOSEOUT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "reports" / "STAGE_A_CLOSEOUT_SUMMARY.md").write_text(
        render_closeout_summary_markdown(summary),
        encoding="utf-8",
    )
    print(f"merged closeout roots into {output_root} verdict={verdict['verdict']}")
    return 0


def _read_many(roots: Sequence[Path], relative_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for root in roots:
        path = root / relative_path
        if path.exists():
            rows.extend(_read_csv(path))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


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


if __name__ == "__main__":
    raise SystemExit(main())
