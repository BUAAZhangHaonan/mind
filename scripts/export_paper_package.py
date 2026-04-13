#!/usr/bin/env python3
"""Export the round-two MIND paper package from saved experiment artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ROUND_TWO_MODEL_LABELS = {
    "qwen3-vl-8b": "Qwen3-VL-8B",
    "internvl3.5-8b": "InternVL3.5-8B",
    "llava-onevision-7b": "LLaVA-OneVision-7B",
    "molmo-7b-d-0924": "Molmo-7B-D-0924",
}

ROUND_TWO_MODEL_ORDER = list(ROUND_TWO_MODEL_LABELS)

ROUND_TWO_BENCHMARK_LABELS = {
    "popular": "POPE popular",
    "dash-b": "DASH-B",
    "adversarial": "POPE adversarial",
    "repope": "RePOPE",
}

ROUND_TWO_BENCHMARK_ORDER = ["popular", "dash-b", "adversarial", "repope"]

MAIN_METHOD_ORDER = [
    ("output_p_yes", "p_yes"),
    ("output_logit_margin", "logit_margin"),
    ("output_chosen_answer_confidence", "chosen_confidence"),
    ("drift_only", "drift_only"),
    ("no_manifold", "no_manifold"),
    ("full", "full MIND"),
    ("linear_probe", "linear_probe"),
]

WIDE_MAIN_METHOD_ORDER = [
    ("output_p_yes", "p_yes"),
    ("output_logit_margin", "logit_margin"),
    ("output_chosen_answer_confidence", "chosen_confidence"),
    ("drift_only", "drift_only"),
    ("no_manifold", "no_manifold"),
    ("full", "full_MIND"),
    ("linear_probe", "linear_probe"),
]

FEATURE_VARIANT_ORDER = [
    ("raw_curve_only", "raw_only"),
    ("raw_plus_calibrated_simple", "raw_plus_simple_stats"),
    ("raw_plus_calibrated_full_curve", "raw_plus_full_curve"),
    ("raw_plus_calibrated_haar", "raw_plus_Haar"),
]

TRANSFER_METHOD_ORDER = [
    ("object", "object"),
    ("shared", "shared"),
    ("shuffled_object", "shuffled_object"),
    ("linear_probe", "linear_probe"),
]

METRIC_COLUMNS = ["roc_auc", "roc_auc_ci", "pr_auc", "pr_auc_ci"]

KNOWN_MODEL_KEYS = sorted(ROUND_TWO_MODEL_LABELS, key=len, reverse=True)
KNOWN_BENCHMARK_KEYS = sorted(ROUND_TWO_BENCHMARK_LABELS, key=len, reverse=True)


@dataclass
class RoundTwoReport:
    path: Path
    model_key: str
    benchmark_key: str
    protocol: str
    baseline: dict[str, object] | None = None
    ablations: pd.DataFrame | None = None
    split_sensitivity: pd.DataFrame | None = None
    halp: dict[str, object] | None = None
    glsim: dict[str, object] | None = None
    variant_results: dict[str, pd.DataFrame] | None = None


def build_output_paths(output_root: Path) -> dict[str, Path]:
    tables_root = output_root / "tables"
    figures_root = output_root / "figures"
    return {
        "table1_csv": tables_root / "table1_main.csv",
        "table1_md": tables_root / "table1_main.md",
        "table1_pope_popular_csv": tables_root / "table1_pope_popular.csv",
        "table1_pope_popular_md": tables_root / "table1_pope_popular.md",
        "table1_dash_b_csv": tables_root / "table1_dash_b.csv",
        "table1_dash_b_md": tables_root / "table1_dash_b.md",
        "table2_csv": tables_root / "table2_feature_ablation.csv",
        "table2_md": tables_root / "table2_feature_ablation.md",
        "table3_csv": tables_root / "table3_transfer_controls.csv",
        "table3_md": tables_root / "table3_transfer_controls.md",
        "supp_pope_adversarial_csv": tables_root / "supp_pope_adversarial.csv",
        "supp_pope_adversarial_md": tables_root / "supp_pope_adversarial.md",
        "supp_repope_csv": tables_root / "supp_repope.csv",
        "supp_repope_md": tables_root / "supp_repope.md",
        "supp_dash_b_transfer_csv": tables_root / "supp_dash_b_transfer.csv",
        "supp_dash_b_transfer_md": tables_root / "supp_dash_b_transfer.md",
        "supp_split_sensitivity_csv": tables_root / "supp_split_sensitivity.csv",
        "supp_split_sensitivity_md": tables_root / "supp_split_sensitivity.md",
        "figure1": figures_root / "figure1_method_diagram.png",
        "figure2": figures_root / "figure2_popular_curves.png",
        "figure3": figures_root / "figure3_transfer_comparison.png",
        "figure_manifest": output_root / "figure_manifest.json",
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_ci(payload: dict[str, object], metric: str) -> str:
    confidence_intervals = payload.get("confidence_intervals")
    if not isinstance(confidence_intervals, dict):
        return ""
    interval = confidence_intervals.get(metric)
    if not isinstance(interval, dict):
        return ""
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None:
        return ""
    return f"[{float(lower):.4f}, {float(upper):.4f}]"


def _metric_row(
    *,
    model_key: str,
    benchmark_key: str,
    protocol: str,
    method: str,
    payload: dict[str, object],
    report_path: Path,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "model": ROUND_TWO_MODEL_LABELS[model_key],
        "benchmark": ROUND_TWO_BENCHMARK_LABELS[benchmark_key],
        "protocol": protocol,
        "method": method,
        "roc_auc": float(payload["roc_auc"]),
        "roc_auc_ci": _format_ci(payload, "roc_auc"),
        "pr_auc": float(payload["pr_auc"]),
        "pr_auc_ci": _format_ci(payload, "pr_auc"),
        "report_path": str(report_path),
    }
    if extra:
        row.update(extra)
    return row


def _format_metric_cell(payload: dict[str, object] | None) -> str:
    if not payload:
        return ""
    return (
        f"ROC {float(payload['roc_auc']):.4f} {_format_ci(payload, 'roc_auc')}; "
        f"PR {float(payload['pr_auc']):.4f} {_format_ci(payload, 'pr_auc')}"
    )


def _metric_frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def _table_to_markdown(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    if not columns:
        return "| |\n| --- |\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    if frame.empty:
        return "\n".join([header, divider]) + "\n"
    body_lines = []
    for row in frame.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body_lines]) + "\n"


def _write_table_bundle(
    frame: pd.DataFrame,
    *,
    export_csv: Path,
    export_md: Path,
    docs_csv: Path,
    docs_md: Path,
) -> None:
    for path in (export_csv, docs_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    for path in (export_md, docs_md):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_table_to_markdown(frame), encoding="utf-8")


def _match_model_key(report_name: str) -> str | None:
    for model_key in KNOWN_MODEL_KEYS:
        if report_name == model_key or report_name.startswith(f"{model_key}-"):
            return model_key
    return None


def _match_benchmark_key(remainder: str) -> str | None:
    for benchmark_key in KNOWN_BENCHMARK_KEYS:
        if remainder == benchmark_key or remainder.startswith(f"{benchmark_key}-"):
            return benchmark_key
    return None


def parse_round_two_report_name(name: str) -> tuple[str, str, str]:
    base = name[7:] if name.startswith("round2-") else name
    model_key = _match_model_key(base)
    if model_key is None:
        raise ValueError(f"Could not parse model key from round-two report name: {name}")
    remainder = base[len(model_key) :].lstrip("-")
    benchmark_key = _match_benchmark_key(remainder)
    if benchmark_key is None:
        raise ValueError(f"Could not parse benchmark key from round-two report name: {name}")
    suffix = remainder[len(benchmark_key) :].lstrip("-")
    if suffix.endswith("object-heldout") or suffix.endswith("object_heldout"):
        protocol = "object_heldout"
    elif suffix.endswith("row"):
        protocol = "row"
    else:
        protocol = "image_grouped"
    return model_key, benchmark_key, protocol


def discover_round_two_reports(reports_root: Path) -> list[RoundTwoReport]:
    discovered: dict[Path, RoundTwoReport] = {}
    for json_name in ("baselines.json", "halp.json"):
        for payload_path in reports_root.rglob(json_name):
            report_dir = payload_path.parent
            model_key, benchmark_key, protocol = parse_round_two_report_name(report_dir.name)
            report = discovered.get(report_dir)
            if report is None:
                report = RoundTwoReport(
                    path=report_dir,
                    model_key=model_key,
                    benchmark_key=benchmark_key,
                    protocol=protocol,
                    variant_results={},
                )
                discovered[report_dir] = report
            if json_name == "baselines.json":
                report.baseline = _load_json(payload_path)
                report.ablations = _read_csv_if_exists(report_dir / "ablations.csv")
                report.split_sensitivity = _read_csv_if_exists(report_dir / "split_sensitivity.csv")
                variant_results: dict[str, pd.DataFrame] = {}
                variant_root = report_dir / "variant_results"
                if variant_root.exists():
                    for variant_path in variant_root.glob("*.csv"):
                        variant_results[variant_path.stem] = pd.read_csv(variant_path)
                report.variant_results = variant_results
            elif json_name == "halp.json":
                report.halp = _load_json(payload_path)
    return sorted(discovered.values(), key=lambda item: (item.model_key, item.benchmark_key, item.protocol, item.path.name))


def _report_completeness_key(report: RoundTwoReport, *, kind: str) -> tuple[int, int, int, int, int, str]:
    variant_count = len(report.variant_results or {})
    if kind == "baseline":
        return (
            1 if report.baseline is not None else 0,
            variant_count,
            1 if report.ablations is not None else 0,
            1 if report.split_sensitivity is not None else 0,
            -len(report.path.name),
            report.path.name,
        )
    if kind == "halp":
        return (
            1 if report.halp is not None else 0,
            1 if (report.path / "halp_results.csv").exists() else 0,
            1 if (report.path / "halp_selection.csv").exists() else 0,
            0,
            -len(report.path.name),
            report.path.name,
        )
    return (0, 0, 0, 0, -len(report.path.name), report.path.name)


def _find_report(
    reports: list[RoundTwoReport],
    *,
    model_key: str,
    benchmark_key: str,
    protocol: str,
    kind: str,
    bank_scope: str | None = None,
) -> RoundTwoReport | None:
    candidates: list[RoundTwoReport] = []
    for report in reports:
        if report.model_key != model_key or report.benchmark_key != benchmark_key or report.protocol != protocol:
            continue
        if kind == "baseline":
            if report.baseline is None:
                continue
            if bank_scope is not None and report.baseline.get("bank_scope") != bank_scope:
                continue
            candidates.append(report)
            continue
        if kind == "halp" and report.halp is not None:
            candidates.append(report)
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda report: _report_completeness_key(report, kind=kind))


def _baseline_payload(report: RoundTwoReport, variant: str) -> dict[str, object]:
    if report.baseline is None:
        raise ValueError(f"Report {report.path} does not contain baselines.json")
    payload = report.baseline.get(variant)
    if not isinstance(payload, dict):
        raise ValueError(f"Missing baseline variant {variant!r} in {report.path}")
    return payload


def _baseline_payload_optional(report: RoundTwoReport, variant: str) -> dict[str, object] | None:
    if report.baseline is not None:
        payload = report.baseline.get(variant)
        if isinstance(payload, dict):
            return payload
    if report.ablations is not None and not report.ablations.empty and "variant" in report.ablations.columns:
        matches = report.ablations[report.ablations["variant"] == variant]
        if not matches.empty:
            row = matches.iloc[0].to_dict()
            if row.get("roc_auc") is not None and row.get("pr_auc") is not None:
                return {
                    "roc_auc": float(row["roc_auc"]),
                    "pr_auc": float(row["pr_auc"]),
                }
    return None


def _result_frame(report: RoundTwoReport, variant: str) -> pd.DataFrame:
    if report.variant_results is None or variant not in report.variant_results:
        raise ValueError(f"Missing variant result CSV {variant!r} in {report.path}")
    return report.variant_results[variant]


def build_main_table(reports: list[RoundTwoReport]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in ROUND_TWO_MODEL_ORDER:
        for benchmark_key in ("popular", "dash-b"):
            report = _find_report(
                reports,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="image_grouped",
                kind="baseline",
                bank_scope="object",
            )
            if report is None:
                continue
            for variant, method_label in MAIN_METHOD_ORDER:
                rows.append(
                    _metric_row(
                        model_key=model_key,
                        benchmark_key=benchmark_key,
                        protocol="image_grouped",
                        method=method_label,
                        payload=_baseline_payload(report, variant),
                        report_path=report.path,
                    )
                )
    columns = [
        "model",
        "benchmark",
        "protocol",
        "method",
        *METRIC_COLUMNS,
        "report_path",
    ]
    return _metric_frame(rows, columns)


def build_feature_table(reports: list[RoundTwoReport]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in ROUND_TWO_MODEL_ORDER:
        for benchmark_key in ("popular", "dash-b"):
            report = _find_report(
                reports,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="image_grouped",
                kind="baseline",
                bank_scope="object",
            )
            if report is None:
                continue
            for variant_key, variant_label in FEATURE_VARIANT_ORDER:
                payload = _baseline_payload(report, variant_key)
                rows.append(
                    {
                        "model": ROUND_TWO_MODEL_LABELS[model_key],
                        "benchmark": ROUND_TWO_BENCHMARK_LABELS[benchmark_key],
                        "protocol": "image_grouped",
                        "feature_variant": variant_label,
                        "roc_auc": float(payload["roc_auc"]),
                        "roc_auc_ci": _format_ci(payload, "roc_auc"),
                        "pr_auc": float(payload["pr_auc"]),
                        "pr_auc_ci": _format_ci(payload, "pr_auc"),
                        "report_path": str(report.path),
                    }
                )
    columns = [
        "model",
        "benchmark",
        "protocol",
        "feature_variant",
        *METRIC_COLUMNS,
        "report_path",
    ]
    return _metric_frame(rows, columns)


def build_wide_feature_table(reports: list[RoundTwoReport]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = ["model", "benchmark", *[variant_label for _, variant_label in FEATURE_VARIANT_ORDER]]
    for model_key in ROUND_TWO_MODEL_ORDER:
        for benchmark_key in ("popular", "dash-b"):
            report = _find_report(
                reports,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="image_grouped",
                kind="baseline",
                bank_scope="object",
            )
            if report is None:
                continue
            row: dict[str, object] = {
                "model": ROUND_TWO_MODEL_LABELS[model_key],
                "benchmark": ROUND_TWO_BENCHMARK_LABELS[benchmark_key],
            }
            for variant_key, variant_label in FEATURE_VARIANT_ORDER:
                row[variant_label] = _format_metric_cell(_baseline_payload_optional(report, variant_key))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_transfer_table(reports: list[RoundTwoReport], *, benchmark_key: str = "popular") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in ROUND_TWO_MODEL_ORDER:
        for protocol in ("image_grouped", "object_heldout"):
            for bank_scope, method_label in TRANSFER_METHOD_ORDER:
                report = _find_report(
                    reports,
                    model_key=model_key,
                    benchmark_key=benchmark_key,
                    protocol=protocol,
                    kind="baseline",
                    bank_scope=bank_scope if bank_scope in {"object", "shared", "shuffled_object"} else "object",
                )
                if report is None:
                    continue
                payload = _baseline_payload(report, "linear_probe" if bank_scope == "linear_probe" else "full")

                rows.append(
                    {
                        "model": ROUND_TWO_MODEL_LABELS[model_key],
                        "benchmark": ROUND_TWO_BENCHMARK_LABELS[benchmark_key],
                        "protocol": protocol,
                        "bank_scope": bank_scope,
                        "method": method_label,
                        "roc_auc": float(payload["roc_auc"]),
                        "roc_auc_ci": _format_ci(payload, "roc_auc"),
                        "pr_auc": float(payload["pr_auc"]),
                        "pr_auc_ci": _format_ci(payload, "pr_auc"),
                        "report_path": str(report.path),
                    }
                )
    columns = [
        "model",
        "benchmark",
        "protocol",
        "bank_scope",
        "method",
        *METRIC_COLUMNS,
        "report_path",
    ]
    return _metric_frame(rows, columns)


def build_wide_transfer_table(reports: list[RoundTwoReport], *, benchmark_key: str = "popular") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = ["model", "benchmark", "method", "image_grouped", "object_heldout"]
    for model_key in ROUND_TWO_MODEL_ORDER:
        for bank_scope, method_label in TRANSFER_METHOD_ORDER:
            row: dict[str, object] = {
                "model": ROUND_TWO_MODEL_LABELS[model_key],
                "benchmark": ROUND_TWO_BENCHMARK_LABELS[benchmark_key],
                "method": method_label,
                "image_grouped": "",
                "object_heldout": "",
            }
            for protocol in ("image_grouped", "object_heldout"):
                payload: dict[str, object] | None = None
                report = _find_report(
                    reports,
                    model_key=model_key,
                    benchmark_key=benchmark_key,
                    protocol=protocol,
                    kind="baseline",
                    bank_scope=bank_scope if bank_scope in {"object", "shared", "shuffled_object"} else "object",
                )
                if report is not None:
                    payload = _baseline_payload(report, "linear_probe" if bank_scope == "linear_probe" else "full")
                row[protocol] = _format_metric_cell(payload)
            if row["image_grouped"] or row["object_heldout"]:
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_benchmark_table(
    reports: list[RoundTwoReport],
    *,
    benchmark_key: str,
    table_protocol: str = "image_grouped",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in ROUND_TWO_MODEL_ORDER:
        report = _find_report(
            reports,
            model_key=model_key,
            benchmark_key=benchmark_key,
            protocol=table_protocol,
            kind="baseline",
            bank_scope="object",
        )
        if report is None:
            continue
        for variant, method_label in MAIN_METHOD_ORDER:
            rows.append(
                _metric_row(
                    model_key=model_key,
                    benchmark_key=benchmark_key,
                    protocol=table_protocol,
                    method=method_label,
                    payload=_baseline_payload(report, variant),
                    report_path=report.path,
                )
            )
    columns = [
        "model",
        "benchmark",
        "protocol",
        "method",
        *METRIC_COLUMNS,
        "report_path",
    ]
    return _metric_frame(rows, columns)


def build_wide_benchmark_table(
    reports: list[RoundTwoReport],
    *,
    benchmark_key: str,
    table_protocol: str = "image_grouped",
) -> pd.DataFrame:
    columns = ["model", "benchmark", *[column_label for _, column_label in WIDE_MAIN_METHOD_ORDER]]
    rows: list[dict[str, object]] = []
    for model_key in ROUND_TWO_MODEL_ORDER:
        row: dict[str, object] = {
            "model": ROUND_TWO_MODEL_LABELS[model_key],
            "benchmark": ROUND_TWO_BENCHMARK_LABELS[benchmark_key],
        }
        report = _find_report(
            reports,
            model_key=model_key,
            benchmark_key=benchmark_key,
            protocol=table_protocol,
            kind="baseline",
            bank_scope="object",
        )
        if report is None:
            continue
        for variant_key, column_label in WIDE_MAIN_METHOD_ORDER:
            row[column_label] = _format_metric_cell(_baseline_payload(report, variant_key))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_supp_split_sensitivity_table(reports: list[RoundTwoReport]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for report in reports:
        if report.baseline is None or report.split_sensitivity is None:
            continue
        split_frame = report.split_sensitivity.copy()
        split_frame["model"] = ROUND_TWO_MODEL_LABELS[report.model_key]
        split_frame["benchmark"] = ROUND_TWO_BENCHMARK_LABELS[report.benchmark_key]
        split_frame["protocol"] = report.protocol
        split_frame["bank_scope"] = str(report.baseline.get("bank_scope", "object"))
        split_frame["report_path"] = str(report.path)
        rows.extend(split_frame.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["model", "benchmark", "protocol", "bank_scope", "variant", "report_path"])
    frame = pd.DataFrame(rows)
    preferred_columns = [column for column in ["model", "benchmark", "protocol", "bank_scope", "variant", "report_path"] if column in frame.columns]
    return frame.loc[:, preferred_columns + [column for column in frame.columns if column not in preferred_columns]]


def plot_method_diagram(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 3.8))
    axis.axis("off")

    steps = [
        "Question + image",
        "Pre-answer hidden states",
        "Round-two reference banks",
        "Layerwise drift curve",
        "Simple-stat feature set",
        "Compact early-warning score",
    ]
    x_positions = np.linspace(0.08, 0.92, len(steps))
    box_kwargs = {
        "boxstyle": "round,pad=0.35",
        "facecolor": "#F4F1E8",
        "edgecolor": "#2F4858",
        "linewidth": 1.5,
    }
    for index, (x_pos, label) in enumerate(zip(x_positions, steps)):
        axis.text(
            x_pos,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=box_kwargs,
            transform=axis.transAxes,
        )
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.06, 0.55),
                xytext=(x_pos + 0.06, 0.55),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#2F4858"},
            )
    axis.text(
        0.5,
        0.14,
        "Object hallucination detection from pre-answer geometry, frozen on simple calibrated statistics.",
        ha="center",
        va="center",
        fontsize=11,
        color="#2F4858",
        transform=axis.transAxes,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_popular_curves(reports: list[RoundTwoReport], output_path: Path) -> None:
    selected_models = ["qwen3-vl-8b", "internvl3.5-8b", "llava-onevision-7b", "molmo-7b-d-0924"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = np.asarray(axes)
    for axis, model_key in zip(axes.flat, selected_models):
        report = _find_report(
            reports,
            model_key=model_key,
            benchmark_key="popular",
            protocol="image_grouped",
            kind="baseline",
            bank_scope="object",
        )
        if report is None:
            axis.axis("off")
            continue
        frame = _result_frame(report, "full")
        fpr, tpr, _ = roc_curve(frame["label"], frame["score"])
        precision, recall, _ = precision_recall_curve(frame["label"], frame["score"])
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall[::-1], precision[::-1])
        axis.plot(fpr, tpr, linewidth=2, color="#2F6690")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
        axis.set_title(f"{ROUND_TWO_MODEL_LABELS[model_key]} ROC")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend([f"AUC={roc_auc:.3f}, PR={pr_auc:.3f}"], loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_transfer_comparison(table3: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.asarray(axes)
    for axis, model_label in zip(axes.flat, ROUND_TWO_MODEL_LABELS.values()):
        subset = table3[(table3["model"] == model_label) & (table3["method"].isin(["object", "shared", "shuffled_object"]))]
        if subset.empty:
            axis.axis("off")
            continue
        subset = subset.sort_values(["method", "protocol"])
        methods = subset["method"].tolist()
        x_positions = np.arange(len(methods))
        axis.bar(x_positions, subset["roc_auc"].astype(float).to_list(), color="#2F6690")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(methods, rotation=20)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(model_label)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_optional_table(
    frame: pd.DataFrame,
    *,
    export_csv: Path,
    export_md: Path,
    docs_csv: Path,
    docs_md: Path,
) -> None:
    _write_table_bundle(frame, export_csv=export_csv, export_md=export_md, docs_csv=docs_csv, docs_md=docs_md)


def export_paper_package(
    *,
    reports_root: Path,
    output_root: Path,
    tables_root: Path = Path("docs/tables/round2"),
) -> dict[str, Path]:
    paths = build_output_paths(output_root)
    reports = discover_round_two_reports(reports_root)

    main_table = build_main_table(reports)
    popular_main_table = build_wide_benchmark_table(reports, benchmark_key="popular")
    dash_b_main_table = build_wide_benchmark_table(reports, benchmark_key="dash-b")
    feature_table = build_wide_feature_table(reports)
    transfer_long_table = build_transfer_table(reports)
    transfer_table = build_wide_transfer_table(reports)
    pop_adversarial_table = build_wide_benchmark_table(reports, benchmark_key="adversarial")
    repope_table = build_wide_benchmark_table(reports, benchmark_key="repope")
    dash_b_transfer_table = build_wide_benchmark_table(
        [report for report in reports if report.benchmark_key == "dash-b"],
        benchmark_key="dash-b",
        table_protocol="object_heldout",
    )
    split_sensitivity_table = build_supp_split_sensitivity_table(reports)

    _write_table_bundle(
        main_table,
        export_csv=paths["table1_csv"],
        export_md=paths["table1_md"],
        docs_csv=tables_root / "table1_main.csv",
        docs_md=tables_root / "table1_main.md",
    )
    _write_table_bundle(
        popular_main_table,
        export_csv=paths["table1_pope_popular_csv"],
        export_md=paths["table1_pope_popular_md"],
        docs_csv=tables_root / "table1_pope_popular.csv",
        docs_md=tables_root / "table1_pope_popular.md",
    )
    _write_table_bundle(
        dash_b_main_table,
        export_csv=paths["table1_dash_b_csv"],
        export_md=paths["table1_dash_b_md"],
        docs_csv=tables_root / "table1_dash_b.csv",
        docs_md=tables_root / "table1_dash_b.md",
    )
    _write_table_bundle(
        feature_table,
        export_csv=paths["table2_csv"],
        export_md=paths["table2_md"],
        docs_csv=tables_root / "table2_feature_ablation.csv",
        docs_md=tables_root / "table2_feature_ablation.md",
    )
    _write_table_bundle(
        transfer_table,
        export_csv=paths["table3_csv"],
        export_md=paths["table3_md"],
        docs_csv=tables_root / "table3_transfer_controls.csv",
        docs_md=tables_root / "table3_transfer_controls.md",
    )
    _write_optional_table(
        pop_adversarial_table,
        export_csv=paths["supp_pope_adversarial_csv"],
        export_md=paths["supp_pope_adversarial_md"],
        docs_csv=tables_root / "supp_pope_adversarial.csv",
        docs_md=tables_root / "supp_pope_adversarial.md",
    )
    _write_optional_table(
        repope_table,
        export_csv=paths["supp_repope_csv"],
        export_md=paths["supp_repope_md"],
        docs_csv=tables_root / "supp_repope.csv",
        docs_md=tables_root / "supp_repope.md",
    )
    if dash_b_transfer_table.empty:
        for path in (
            paths["supp_dash_b_transfer_csv"],
            paths["supp_dash_b_transfer_md"],
            tables_root / "supp_dash_b_transfer.csv",
            tables_root / "supp_dash_b_transfer.md",
        ):
            path.unlink(missing_ok=True)
    else:
        _write_optional_table(
            dash_b_transfer_table,
            export_csv=paths["supp_dash_b_transfer_csv"],
            export_md=paths["supp_dash_b_transfer_md"],
            docs_csv=tables_root / "supp_dash_b_transfer.csv",
            docs_md=tables_root / "supp_dash_b_transfer.md",
        )
    _write_optional_table(
        split_sensitivity_table,
        export_csv=paths["supp_split_sensitivity_csv"],
        export_md=paths["supp_split_sensitivity_md"],
        docs_csv=tables_root / "supp_split_sensitivity.csv",
        docs_md=tables_root / "supp_split_sensitivity.md",
    )

    plot_method_diagram(paths["figure1"])
    _plot_popular_curves(reports, paths["figure2"])
    _plot_transfer_comparison(transfer_long_table, paths["figure3"])

    manifest = {
        "figure1": {"title": "Round-two method diagram", "path": str(paths["figure1"])},
        "figure2": {"title": "Popular ROC curves", "path": str(paths["figure2"])},
        "figure3": {"title": "Transfer comparison", "path": str(paths["figure3"])},
    }
    paths["figure_manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/round2_2026_04/reports"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/paper_closeout"))
    parser.add_argument("--tables-root", type=Path, default=Path("docs/tables/round2"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = export_paper_package(reports_root=args.reports_root, output_root=args.output_root, tables_root=args.tables_root)
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
