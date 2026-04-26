#!/usr/bin/env python3
"""Prepare decisive-round Goal 1 full-curve report artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd


FULL_CURVE_VARIANT = "raw_plus_calibrated_full_curve"
FEATURE_ONLY_VARIANTS = (
    "full",
    "raw_curve_only",
    "raw_plus_calibrated_simple",
    FULL_CURVE_VARIANT,
)
KNOWN_MODEL_NAMES = (
    "llava-onevision-7b",
    "internvl3.5-8b",
    "qwen3-vl-8b",
    "molmo-7b-d-0924",
)
VARIANT_ORDER = (
    "full",
    "drift_only",
    "no_manifold",
    "linear_probe",
    "output_p_yes",
    "output_logit_margin",
    "output_chosen_answer_confidence",
    "raw_curve_only",
    "raw_plus_calibrated_simple",
    FULL_CURVE_VARIANT,
    "raw_plus_calibrated_haar",
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_report_dirs(source_root: Path, dest_root: Path) -> list[Path]:
    dest_root.mkdir(parents=True, exist_ok=True)
    copied_reports: list[Path] = []
    for source_report in sorted(path for path in source_root.iterdir() if path.is_dir()):
        dest_report = dest_root / source_report.name
        if not dest_report.exists():
            shutil.copytree(source_report, dest_report)
        copied_reports.append(dest_report)
    return copied_reports


def replace_full_rows_from_full_curve(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    frame = pd.read_csv(csv_path)
    if "variant" not in frame.columns:
        return False
    source_rows = frame.loc[frame["variant"] == FULL_CURVE_VARIANT].copy()
    if source_rows.empty:
        return False
    source_rows.loc[:, "variant"] = "full"
    key_columns = ["variant"]
    if "random_state" in frame.columns:
        key_columns.append("random_state")
    merged = pd.concat([frame, source_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_columns, keep="last")
    if "variant" in merged.columns:
        rank = {name: index for index, name in enumerate(VARIANT_ORDER)}
        merged["__variant_rank"] = merged["variant"].map(rank).fillna(len(rank)).astype(int)
        sort_columns = ["__variant_rank"]
        if "random_state" in merged.columns:
            sort_columns.append("random_state")
        merged = merged.sort_values(sort_columns).drop(columns="__variant_rank").reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    return True


def alias_full_curve_in_report(report_dir: Path) -> bool:
    baselines_path = report_dir / "baselines.json"
    if not baselines_path.exists():
        return False
    baselines = load_json(baselines_path)
    full_curve_payload = baselines.get(FULL_CURVE_VARIANT)
    if not isinstance(full_curve_payload, dict):
        return False

    variant_results = report_dir / "variant_results"
    full_curve_csv = variant_results / f"{FULL_CURVE_VARIANT}.csv"
    full_csv = variant_results / "full.csv"
    aliased_payload = copy.deepcopy(full_curve_payload)
    if full_curve_csv.exists():
        full_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_curve_csv, full_csv)
        aliased_payload["result_path"] = str(full_csv)

    baselines["full"] = aliased_payload
    baselines["full_variant"] = FULL_CURVE_VARIANT
    write_json(baselines_path, baselines)
    replace_full_rows_from_full_curve(report_dir / "ablations.csv")
    replace_full_rows_from_full_curve(report_dir / "split_sensitivity.csv")
    return True


def parse_model_name(report_name: str) -> str | None:
    for model_name in KNOWN_MODEL_NAMES:
        if report_name.startswith(f"round2-{model_name}-"):
            return model_name
    return None


def infer_feature_spec(report_name: str, features_root: Path) -> tuple[Path, str, int, Path | None] | None:
    model_name = parse_model_name(report_name)
    if model_name is None:
        return None
    suffix = report_name.removeprefix(f"round2-{model_name}-")
    if suffix.endswith("-halp-row"):
        return None
    if suffix.startswith("popular-object-heldout"):
        candidates = (
            features_root / f"round2-{model_name}-popular-object-heldout-refresh" / "popular.parquet",
            features_root / f"round2-{model_name}-popular" / "popular.parquet",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate, "object_heldout", 2, None
        return candidates[0], "object_heldout", 2, None
    if suffix.startswith("popular"):
        return features_root / f"round2-{model_name}-popular" / "popular.parquet", "image_grouped", 5, None
    if suffix.startswith("adversarial"):
        return features_root / f"round2-{model_name}-adversarial" / "adversarial.parquet", "image_grouped", 5, None
    if suffix.startswith("dash-b"):
        return features_root / f"round2-{model_name}-dash-b" / "main.parquet", "image_grouped", 5, None
    if suffix.startswith("repope"):
        return (
            features_root / f"round2-{model_name}-popular" / "popular.parquet",
            "image_grouped",
            5,
            features_root.parent / "normalized" / "repope" / "popular.jsonl",
        )
    return None


def infer_cache_path(report_name: str, source_root: Path) -> Path:
    model_name = parse_model_name(report_name) or "unknown-model"
    suffix = report_name.removeprefix(f"round2-{model_name}-")
    if suffix.startswith("dash-b"):
        return source_root.parent / "cache" / model_name / "dash-b" / "main"
    if suffix.startswith("repope"):
        return source_root.parent / "cache" / model_name / "repope" / "popular"
    if suffix.startswith("adversarial"):
        return source_root.parent / "cache" / model_name / "pope" / "adversarial"
    return source_root.parent / "cache" / model_name / "pope" / "popular"


def infer_reference_root(report_name: str, source_root: Path) -> Path:
    if "-dash-b" in report_name:
        dash_b_root = source_root.parent / "reference_banks_dash_b"
        if dash_b_root.exists():
            return dash_b_root
    return source_root.parent / "reference_banks"


def build_feature_baseline_command(
    *,
    report_name: str,
    source_root: Path,
    dest_root: Path,
    features_root: Path,
) -> list[str] | None:
    feature_spec = infer_feature_spec(report_name, features_root)
    model_name = parse_model_name(report_name)
    if feature_spec is None or model_name is None:
        return None
    features_path, split_strategy, num_folds, label_overrides = feature_spec
    if not features_path.exists():
        return None
    compute_script = Path(__file__).resolve().parent / "compute_baselines.py"
    command = [
        sys.executable,
        str(compute_script),
        "--features-path",
        str(features_path),
        "--cache-path",
        str(infer_cache_path(report_name, source_root)),
        "--reference-root",
        str(infer_reference_root(report_name, source_root)),
        "--model-name",
        model_name,
        "--output-root",
        str(dest_root),
        "--experiment-name",
        report_name,
        "--split-strategy",
        split_strategy,
        "--num-folds",
        str(num_folds),
        "--bank-scope",
        str(load_json(dest_root / report_name / "baselines.json").get("bank_scope", "object")),
        "--full-variant",
        FULL_CURVE_VARIANT,
        "--variants",
        ",".join(FEATURE_ONLY_VARIANTS),
    ]
    if label_overrides is not None:
        if not label_overrides.exists():
            return None
        command.extend(["--label-overrides", str(label_overrides)])
    return command


def maybe_run_feature_baselines(
    *,
    source_root: Path,
    dest_root: Path,
    features_root: Path,
    runner: Callable[[list[str]], object] | None,
) -> int:
    run = runner
    if run is None:
        run = lambda command: subprocess.run(command, check=True)
    executed = 0
    for report_dir in sorted(path for path in dest_root.iterdir() if path.is_dir()):
        if not (report_dir / "baselines.json").exists():
            continue
        command = build_feature_baseline_command(
            report_name=report_dir.name,
            source_root=source_root,
            dest_root=dest_root,
            features_root=features_root,
        )
        if command is None:
            continue
        run(command)
        executed += 1
        alias_full_curve_in_report(report_dir)
    return executed


def metric(payload: object, name: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(name)
    if value is None:
        return None
    return float(value)


def choose_old_full_payload(baselines: dict[str, object]) -> object:
    simple_payload = baselines.get("raw_plus_calibrated_simple")
    if isinstance(simple_payload, dict):
        return simple_payload
    return baselines.get("full")


def format_float(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return ""
    if signed:
        return f"{value:+.4f}"
    return f"{value:.4f}"


def gap_values(
    *,
    old_full: object,
    new_full: object,
    comparator: object,
    metric_name: str,
) -> tuple[float | None, float | None, float | None]:
    comparator_value = metric(comparator, metric_name)
    old_value = metric(old_full, metric_name)
    new_value = metric(new_full, metric_name)
    if comparator_value is None or old_value is None or new_value is None:
        return None, None, None
    old_gap = comparator_value - old_value
    new_gap = comparator_value - new_value
    return old_gap, new_gap, old_gap - new_gap


def build_comparison_rows(source_root: Path, dest_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dest_report in sorted(path for path in dest_root.iterdir() if path.is_dir()):
        source_baselines_path = source_root / dest_report.name / "baselines.json"
        dest_baselines_path = dest_report / "baselines.json"
        if not source_baselines_path.exists() or not dest_baselines_path.exists():
            continue
        source_baselines = load_json(source_baselines_path)
        dest_baselines = load_json(dest_baselines_path)
        old_full = choose_old_full_payload(source_baselines)
        new_full = dest_baselines.get("full")
        old_pr = metric(old_full, "pr_auc")
        new_pr = metric(new_full, "pr_auc")
        old_roc = metric(old_full, "roc_auc")
        new_roc = metric(new_full, "roc_auc")
        linear_pr = gap_values(
            old_full=old_full,
            new_full=new_full,
            comparator=dest_baselines.get("linear_probe"),
            metric_name="pr_auc",
        )
        no_manifold_pr = gap_values(
            old_full=old_full,
            new_full=new_full,
            comparator=dest_baselines.get("no_manifold"),
            metric_name="pr_auc",
        )
        linear_roc = gap_values(
            old_full=old_full,
            new_full=new_full,
            comparator=dest_baselines.get("linear_probe"),
            metric_name="roc_auc",
        )
        no_manifold_roc = gap_values(
            old_full=old_full,
            new_full=new_full,
            comparator=dest_baselines.get("no_manifold"),
            metric_name="roc_auc",
        )
        rows.append(
            {
                "report": dest_report.name,
                "old_full_pr_auc": format_float(old_pr),
                "new_full_pr_auc": format_float(new_pr),
                "full_pr_auc_delta": format_float(None if old_pr is None or new_pr is None else new_pr - old_pr, signed=True),
                "linear_probe_pr_gap_old": format_float(linear_pr[0]),
                "linear_probe_pr_gap_new": format_float(linear_pr[1]),
                "linear_probe_pr_gap_shrink": format_float(linear_pr[2], signed=True),
                "no_manifold_pr_gap_old": format_float(no_manifold_pr[0]),
                "no_manifold_pr_gap_new": format_float(no_manifold_pr[1]),
                "no_manifold_pr_gap_shrink": format_float(no_manifold_pr[2], signed=True),
                "old_full_roc_auc": format_float(old_roc),
                "new_full_roc_auc": format_float(new_roc),
                "full_roc_auc_delta": format_float(
                    None if old_roc is None or new_roc is None else new_roc - old_roc,
                    signed=True,
                ),
                "linear_probe_roc_gap_old": format_float(linear_roc[0]),
                "linear_probe_roc_gap_new": format_float(linear_roc[1]),
                "linear_probe_roc_gap_shrink": format_float(linear_roc[2], signed=True),
                "no_manifold_roc_gap_old": format_float(no_manifold_roc[0]),
                "no_manifold_roc_gap_new": format_float(no_manifold_roc[1]),
                "no_manifold_roc_gap_shrink": format_float(no_manifold_roc[2], signed=True),
            }
        )
    return rows


def write_comparison_markdown(source_root: Path, dest_root: Path, comparison_path: Path) -> None:
    columns = [
        "report",
        "old_full_pr_auc",
        "new_full_pr_auc",
        "full_pr_auc_delta",
        "linear_probe_pr_gap_old",
        "linear_probe_pr_gap_new",
        "linear_probe_pr_gap_shrink",
        "no_manifold_pr_gap_old",
        "no_manifold_pr_gap_new",
        "no_manifold_pr_gap_shrink",
        "old_full_roc_auc",
        "new_full_roc_auc",
        "full_roc_auc_delta",
        "linear_probe_roc_gap_old",
        "linear_probe_roc_gap_new",
        "linear_probe_roc_gap_shrink",
        "no_manifold_roc_gap_old",
        "no_manifold_roc_gap_new",
        "no_manifold_roc_gap_shrink",
    ]
    rows = build_comparison_rows(source_root, dest_root)
    lines = [
        "# Decisive Full-Curve Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_goal1_reports(
    *,
    source_root: Path,
    dest_root: Path,
    comparison_path: Path,
    features_root: Path | None = None,
    run_feature_baselines: bool = False,
    runner: Callable[[list[str]], object] | None = None,
) -> None:
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    comparison_path = Path(comparison_path)
    if features_root is None:
        features_root = source_root.parent / "features"
    else:
        features_root = Path(features_root)

    copied_reports = copy_report_dirs(source_root, dest_root)
    for report_dir in copied_reports:
        alias_full_curve_in_report(report_dir)
    if run_feature_baselines:
        maybe_run_feature_baselines(
            source_root=source_root,
            dest_root=dest_root,
            features_root=features_root,
            runner=runner,
        )
    write_comparison_markdown(source_root, dest_root, comparison_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/round2_2026_04/reports"))
    parser.add_argument("--dest-root", type=Path, default=Path("outputs/decisive_round_2026_04/reports"))
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=Path("docs/tables/decisive_full_curve_comparison.md"),
    )
    parser.add_argument("--run-feature-baselines", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_goal1_reports(
        source_root=args.source_root,
        dest_root=args.dest_root,
        features_root=args.features_root,
        comparison_path=args.comparison_path,
        run_feature_baselines=args.run_feature_baselines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
