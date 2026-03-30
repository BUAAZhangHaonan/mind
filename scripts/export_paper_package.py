#!/usr/bin/env python3
"""Export the MIND paper closeout package from saved experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


METRIC_ORDER = [
    "roc_auc",
    "pr_auc",
    "tpr_at_fpr_0.01",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "false_positive_rate",
]

MODEL_LABELS = {
    "qwen": "Qwen3-VL-8B",
    "internvl": "InternVL3.5-8B",
}

EXPERIMENTS = {
    "qwen_popular": "correction-qwen3-vl-8b-popular",
    "qwen_popular_shared": "correction-qwen3-vl-8b-popular-shared",
    "qwen_popular_row": "correction-qwen3-vl-8b-popular-row",
    "qwen_popular_object": "correction-qwen3-vl-8b-popular-object-heldout",
    "qwen_popular_shared_object": "correction-qwen3-vl-8b-popular-shared-object-heldout",
    "qwen_popular_repope": "correction-qwen3-vl-8b-popular-repope",
    "qwen_adversarial": "correction-qwen3-vl-8b-adversarial",
    "internvl_popular": "correction-internvl3.5-8b-popular",
    "internvl_popular_shared": "correction-internvl3.5-8b-popular-shared",
    "internvl_popular_row": "correction-internvl3.5-8b-popular-row",
    "internvl_popular_object": "correction-internvl3.5-8b-popular-object-heldout",
    "internvl_popular_shared_object": "correction-internvl3.5-8b-popular-shared-object-heldout",
    "internvl_popular_repope": "correction-internvl3.5-8b-popular-repope",
    "internvl_adversarial": "correction-internvl3.5-8b-adversarial",
}


def build_output_paths(output_root: Path) -> dict[str, Path]:
    tables_root = output_root / "tables"
    figures_root = output_root / "figures"
    return {
        "table1_csv": tables_root / "table1_main_grouped_results.csv",
        "table1_md": tables_root / "table1_main_grouped_results.md",
        "table2_csv": tables_root / "table2_structure_comparison.csv",
        "table2_md": tables_root / "table2_structure_comparison.md",
        "table3_csv": tables_root / "table3_object_transfer_boundary.csv",
        "table3_md": tables_root / "table3_object_transfer_boundary.md",
        "figure1": figures_root / "figure1_method_diagram.png",
        "figure2": figures_root / "figure2_popular_grouped_curves.png",
        "figure3": figures_root / "figure3_protocol_comparison.png",
        "figure_manifest": output_root / "figure_manifest.json",
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_overall_metrics(reports_root: Path, experiment_name: str) -> dict[str, float]:
    payload = _load_json(reports_root / experiment_name / "metrics.json")
    return {metric: float(payload["overall"][metric]) for metric in METRIC_ORDER}


def load_baseline_metrics(
    reports_root: Path,
    experiment_name: str,
    variant: str,
) -> dict[str, float]:
    payload = _load_json(reports_root / experiment_name / "baselines.json")
    return {metric: float(payload[variant][metric]) for metric in METRIC_ORDER}


def load_results_frame(reports_root: Path, experiment_name: str) -> pd.DataFrame:
    return pd.read_csv(reports_root / experiment_name / "results.csv")


def build_table1(reports_root: Path) -> pd.DataFrame:
    rows = [
        {"setting": "Qwen popular", **load_overall_metrics(reports_root, EXPERIMENTS["qwen_popular"])},
        {
            "setting": "Qwen popular + RePOPE",
            **load_overall_metrics(reports_root, EXPERIMENTS["qwen_popular_repope"]),
        },
        {"setting": "Qwen adversarial", **load_overall_metrics(reports_root, EXPERIMENTS["qwen_adversarial"])},
        {"setting": "InternVL popular", **load_overall_metrics(reports_root, EXPERIMENTS["internvl_popular"])},
        {
            "setting": "InternVL popular + RePOPE",
            **load_overall_metrics(reports_root, EXPERIMENTS["internvl_popular_repope"]),
        },
        {
            "setting": "InternVL adversarial",
            **load_overall_metrics(reports_root, EXPERIMENTS["internvl_adversarial"]),
        },
    ]
    return pd.DataFrame(rows)[["setting", *METRIC_ORDER]]


def build_table2(reports_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in ["qwen", "internvl"]:
        popular_key = f"{model_key}_popular"
        shared_key = f"{model_key}_popular_shared"
        rows.extend(
            [
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "full MIND (object bank)",
                    **load_overall_metrics(reports_root, EXPERIMENTS[popular_key]),
                },
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "full MIND (shared bank)",
                    **load_overall_metrics(reports_root, EXPERIMENTS[shared_key]),
                },
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "drift-only",
                    **load_baseline_metrics(reports_root, EXPERIMENTS[popular_key], "drift_only"),
                },
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "no-manifold",
                    **load_baseline_metrics(reports_root, EXPERIMENTS[popular_key], "no_manifold"),
                },
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "linear probe",
                    **load_baseline_metrics(reports_root, EXPERIMENTS[popular_key], "linear_probe"),
                },
            ]
        )
    return pd.DataFrame(rows)[["model", "variant", *METRIC_ORDER]]


def build_table3(reports_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key in ["qwen", "internvl"]:
        object_key = f"{model_key}_popular_object"
        shared_key = f"{model_key}_popular_shared_object"
        rows.extend(
            [
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "full MIND (object bank)",
                    **load_overall_metrics(reports_root, EXPERIMENTS[object_key]),
                },
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "full MIND (shared bank)",
                    **load_overall_metrics(reports_root, EXPERIMENTS[shared_key]),
                },
                {
                    "model": MODEL_LABELS[model_key],
                    "variant": "linear probe",
                    **load_baseline_metrics(reports_root, EXPERIMENTS[object_key], "linear_probe"),
                },
            ]
        )
    return pd.DataFrame(rows)[["model", "variant", *METRIC_ORDER]]


def _table_to_markdown(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
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


def write_table_bundle(frame: pd.DataFrame, *, csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(_table_to_markdown(frame), encoding="utf-8")


def _prepare_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def plot_method_diagram(output_path: Path) -> None:
    _prepare_figure(output_path)
    figure, axis = plt.subplots(figsize=(12, 3.6))
    axis.axis("off")

    steps = [
        "Image + object\nquestion",
        "Selected pre-answer\nhidden states",
        "Grounded reference bank\n(object or shared)",
        "Layerwise manifold\ndrift curve",
        "Calibrated wavelets +\nraw magnitude",
        "Lightweight early-warning\nscore",
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
        "Ranking-oriented early warning: compressible geometric drift before answer generation",
        ha="center",
        va="center",
        fontsize=11,
        color="#2F4858",
        transform=axis.transAxes,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_popular_curve_panel(reports_root: Path, output_path: Path) -> None:
    _prepare_figure(output_path)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    curve_specs = [
        (MODEL_LABELS["qwen"], EXPERIMENTS["qwen_popular"]),
        (MODEL_LABELS["internvl"], EXPERIMENTS["internvl_popular"]),
    ]
    for row_index, (model_label, experiment_name) in enumerate(curve_specs):
        frame = load_results_frame(reports_root, experiment_name)
        fpr, tpr, _ = roc_curve(frame["label"], frame["score"])
        precision, recall, _ = precision_recall_curve(frame["label"], frame["score"])
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall[::-1], precision[::-1])

        roc_axis = axes[row_index, 0]
        pr_axis = axes[row_index, 1]

        roc_axis.plot(fpr, tpr, linewidth=2, color="#2F6690")
        roc_axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
        roc_axis.set_title(f"{model_label} ROC")
        roc_axis.set_xlabel("False Positive Rate")
        roc_axis.set_ylabel("True Positive Rate")
        roc_axis.legend([f"AUC={roc_auc:.3f}"], loc="lower right")

        pr_axis.plot(recall, precision, linewidth=2, color="#C06C84")
        pr_axis.set_title(f"{model_label} PR")
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        pr_axis.legend([f"AUC={pr_auc:.3f}"], loc="lower left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_protocol_comparison(reports_root: Path, output_path: Path) -> None:
    _prepare_figure(output_path)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    protocols = [
        ("row", "Legacy row"),
        ("image_grouped", "image_grouped"),
        ("object_heldout", "object_heldout"),
    ]
    metrics = ["roc_auc", "pr_auc", "tpr_at_fpr_0.01"]
    colors = {"Qwen3-VL-8B": "#2F6690", "InternVL3.5-8B": "#C06C84"}

    model_protocol_metrics = {
        MODEL_LABELS["qwen"]: {
            "row": load_overall_metrics(reports_root, EXPERIMENTS["qwen_popular_row"]),
            "image_grouped": load_overall_metrics(reports_root, EXPERIMENTS["qwen_popular"]),
            "object_heldout": load_overall_metrics(reports_root, EXPERIMENTS["qwen_popular_object"]),
        },
        MODEL_LABELS["internvl"]: {
            "row": load_overall_metrics(reports_root, EXPERIMENTS["internvl_popular_row"]),
            "image_grouped": load_overall_metrics(reports_root, EXPERIMENTS["internvl_popular"]),
            "object_heldout": load_overall_metrics(reports_root, EXPERIMENTS["internvl_popular_object"]),
        },
    }

    x_positions = np.arange(len(protocols))
    width = 0.34
    for axis, metric in zip(axes, metrics):
        for offset, model_label in enumerate(MODEL_LABELS.values()):
            values = [model_protocol_metrics[model_label][protocol][metric] for protocol, _ in protocols]
            axis.bar(
                x_positions + (offset - 0.5) * width,
                values,
                width=width,
                label=model_label,
                color=colors[model_label],
            )
        axis.set_xticks(x_positions)
        axis.set_xticklabels([label for _, label in protocols], rotation=10)
        axis.set_title(metric)
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def export_paper_package(*, reports_root: Path, output_root: Path) -> dict[str, Path]:
    paths = build_output_paths(output_root)
    table1 = build_table1(reports_root)
    table2 = build_table2(reports_root)
    table3 = build_table3(reports_root)

    write_table_bundle(table1, csv_path=paths["table1_csv"], markdown_path=paths["table1_md"])
    write_table_bundle(table2, csv_path=paths["table2_csv"], markdown_path=paths["table2_md"])
    write_table_bundle(table3, csv_path=paths["table3_csv"], markdown_path=paths["table3_md"])

    plot_method_diagram(paths["figure1"])
    plot_popular_curve_panel(reports_root, paths["figure2"])
    plot_protocol_comparison(reports_root, paths["figure3"])

    manifest = {
        "figure1": {"title": "Method diagram", "path": str(paths["figure1"])},
        "figure2": {"title": "Popular ROC and PR curves", "path": str(paths["figure2"])},
        "figure3": {"title": "Protocol comparison", "path": str(paths["figure3"])},
    }
    paths["figure_manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/correction_phase/reports"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/paper_closeout"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = export_paper_package(reports_root=args.reports_root, output_root=args.output_root)
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
