"""Plot helpers for MIND experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


def _prepare_output(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_ablation_summary(frame: pd.DataFrame, *, metric: str, base_variant: str) -> pd.DataFrame:
    summary = frame.set_index("variant")[[metric]].copy()
    base_value = float(summary.loc[base_variant, metric])
    summary["delta"] = (summary[metric] - base_value).round(6)
    return summary


def plot_drift_curves(
    frame: pd.DataFrame | None = None,
    *,
    grounded_curve: np.ndarray | None = None,
    hallucinated_curve: np.ndarray | None = None,
    output_path: str | Path,
) -> None:
    if frame is not None:
        drift_columns = [column for column in frame.columns if column.startswith("raw_drift_")]
        if not drift_columns:
            drift_columns = [column for column in frame.columns if column.startswith("drift_")]
        grounded_curve = frame.loc[frame["label"] == 0, drift_columns].mean(axis=0).to_numpy()
        hallucinated_curve = frame.loc[frame["label"] == 1, drift_columns].mean(axis=0).to_numpy()
    if grounded_curve is None or hallucinated_curve is None:
        raise ValueError("plot_drift_curves requires either a frame or both drift curves.")

    path = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(grounded_curve, label="grounded", linewidth=2)
    axis.plot(hallucinated_curve, label="hallucinated", linewidth=2)
    axis.set_xlabel("Layer index")
    axis.set_ylabel("Drift score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_drift_curve_comparison(
    *,
    grounded_curve: np.ndarray,
    hallucinated_curve: np.ndarray,
    output_path: str | Path,
) -> None:
    plot_drift_curves(
        grounded_curve=grounded_curve,
        hallucinated_curve=hallucinated_curve,
        output_path=output_path,
    )


def plot_wavelet_energy_heatmap(
    *,
    matrix: np.ndarray | None = None,
    energy: np.ndarray | None = None,
    row_labels: list[str],
    col_labels: list[str] | None = None,
    column_labels: list[str] | None = None,
    output_path: str | Path,
) -> None:
    if matrix is None:
        matrix = energy
    labels = col_labels or column_labels
    if matrix is None or labels is None:
        raise ValueError("Wavelet heatmap requires matrix and column labels.")

    path = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=row_labels, cmap="viridis", ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_wavelet_heatmap(frame: pd.DataFrame, *, output_path: str | Path) -> None:
    energy_columns = [
        column
        for column in frame.columns
        if column == "cal_approx_energy" or column.startswith("cal_detail_energy_")
    ]
    if not energy_columns:
        energy_columns = [
            column
            for column in frame.columns
            if column == "approx_energy" or column.startswith("detail_energy_")
        ]
    matrix = frame[energy_columns].to_numpy()
    row_labels = [f"sample_{index}" for index in range(len(frame))]
    plot_wavelet_energy_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=energy_columns,
        output_path=output_path,
    )


def plot_roc_curve(
    *,
    false_positive_rate: np.ndarray | None = None,
    true_positive_rate: np.ndarray | None = None,
    roc_auc: float | None = None,
    fpr: np.ndarray | None = None,
    tpr: np.ndarray | None = None,
    y_true=None,
    y_score=None,
    output_path: str | Path,
) -> None:
    if false_positive_rate is None:
        false_positive_rate = fpr
    if true_positive_rate is None:
        true_positive_rate = tpr
    if y_true is not None and y_score is not None:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        roc_auc = float(roc_auc_score(y_true, y_score))
    if false_positive_rate is None or true_positive_rate is None or roc_auc is None:
        raise ValueError("ROC inputs are incomplete.")

    path = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC={roc_auc:.3f}", linewidth=2)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_ablation_bars(
    frame: pd.DataFrame,
    *,
    metric: str | None = None,
    output_path: str | Path,
) -> None:
    if metric is None:
        metric = "metric" if "metric" in frame.columns else "accuracy"

    path = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(data=frame, x="variant", y=metric, ax=axis, color="#4C78A8")
    axis.set_ylabel(metric)
    axis.set_xlabel("variant")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
