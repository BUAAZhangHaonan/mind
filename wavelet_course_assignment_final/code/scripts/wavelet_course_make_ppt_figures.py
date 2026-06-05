#!/usr/bin/env python3
"""Make PPT-friendly figures from wavelet-course v2 report CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REPORTS_DIR = Path("outputs/wavelet_course_v2/reports")
DEFAULT_OUTPUT_DIR = Path("outputs/wavelet_course_v2/figures/ppt")
VALID_FORMATS = {"png", "pdf"}
HALP_SCORE_TABLE = "halp_score_results"
REPORT_FILENAMES = (
    "metrics_wide_paired.csv",
    "pairwise_winrate.csv",
    "metrics_long.csv",
    "domain_baselines.csv",
)
HALP_RESULT_FILENAMES = (
    "halp_results.csv",
    "halp_official_row_results.csv",
)
STRUCTURAL_REQUIRED_COLUMNS = {
    "metrics_wide_paired.csv": (
        "block",
        "pair_id",
        "paired_status",
        "teacher_status",
        "ours_status",
        "teacher_total_seconds",
        "ours_total_seconds",
    ),
    "pairwise_winrate.csv": (
        "block",
        "metric",
        "comparable_pairs",
        "ours_wins",
        "teacher_wins",
        "ties",
        "ours_winrate",
        "teacher_winrate",
        "tie_rate",
    ),
    "metrics_long.csv": (
        "block",
        "pair_id",
        "source",
        "status",
        "config_name",
        "transform",
        "feature_protocol",
        "window_strategy",
        "window_mode",
        "threshold",
        "classifier",
        "total_seconds",
    ),
    "domain_baselines.csv": (
        "baseline_name",
        "method_family",
        "source",
        "status",
        "test_pr_auc",
        "total_seconds",
    ),
}
WIDE_SUCCESS_METRIC_COLUMNS = ("teacher_pr_auc", "ours_pr_auc", "delta_pr_auc")
LONG_SUCCESS_METRIC_COLUMNS = ("test_pr_auc",)
DOMAIN_SUCCESS_METRIC_COLUMNS = ("test_pr_auc",)
HALP_REQUIRED_COLUMNS = ("score", "label")
FACTOR_COLUMNS = (
    "transform",
    "feature_protocol",
    "window_strategy",
    "window_mode",
    "threshold",
    "classifier",
)
FACTOR_LABELS = {
    "transform": "Transform",
    "feature_protocol": "Feature protocol",
    "window_strategy": "Window strategy",
    "window_mode": "Window mode",
    "threshold": "Threshold",
    "classifier": "Classifier",
}
BLOCK_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "overall": 99}
COLORS = {
    "teacher": "#4C78A8",
    "ours": "#F58518",
    "halp": "#54A24B",
    "linear": "#B279A2",
    "neutral": "#5F6B6D",
    "zero": "#3A3A3A",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--format",
        default="png,pdf",
        help="Comma-separated output formats. Supported: png,pdf.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formats = parse_formats(args.format)
    tables = load_reports(args.reports_dir)
    generated = make_figures(tables, output_dir=args.output_dir, formats=formats)
    for path in generated:
        print(path)
    return 0


def parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not formats:
        raise ValueError("--format must include at least one format")
    unknown = sorted(set(formats) - VALID_FORMATS)
    if unknown:
        raise ValueError(f"unsupported format(s): {', '.join(unknown)}")
    return formats


def load_reports(reports_dir: Path) -> dict[str, pd.DataFrame]:
    reports_dir = Path(reports_dir)
    tables: dict[str, pd.DataFrame] = {}
    for filename in REPORT_FILENAMES:
        path = reports_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"required report CSV is missing: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{filename}: report must not be empty")
        validate_required_columns(filename, df, STRUCTURAL_REQUIRED_COLUMNS[filename])
        validate_no_nan(filename, df, STRUCTURAL_REQUIRED_COLUMNS[filename])
        tables[filename] = df

    wide_success = comparable_wide_rows(tables["metrics_wide_paired.csv"])
    validate_required_columns("metrics_wide_paired.csv", wide_success, WIDE_SUCCESS_METRIC_COLUMNS)
    validate_no_nan("metrics_wide_paired.csv", wide_success, WIDE_SUCCESS_METRIC_COLUMNS)
    coerce_numeric("metrics_wide_paired.csv", wide_success, (*WIDE_SUCCESS_METRIC_COLUMNS, "teacher_total_seconds", "ours_total_seconds"))

    long_success = success_rows(tables["metrics_long.csv"])
    validate_required_columns("metrics_long.csv", long_success, LONG_SUCCESS_METRIC_COLUMNS)
    validate_no_nan("metrics_long.csv", long_success, LONG_SUCCESS_METRIC_COLUMNS)
    coerce_numeric("metrics_long.csv", long_success, (*LONG_SUCCESS_METRIC_COLUMNS, "total_seconds"))

    domain_success = success_rows(tables["domain_baselines.csv"])
    validate_required_columns("domain_baselines.csv", domain_success, DOMAIN_SUCCESS_METRIC_COLUMNS)
    validate_no_nan("domain_baselines.csv", domain_success, DOMAIN_SUCCESS_METRIC_COLUMNS)
    coerce_numeric("domain_baselines.csv", domain_success, (*DOMAIN_SUCCESS_METRIC_COLUMNS, "total_seconds"))

    pr_winrate = tables["pairwise_winrate.csv"][
        tables["pairwise_winrate.csv"]["metric"].astype(str).str.lower().eq("pr_auc")
    ]
    if pr_winrate.empty:
        raise ValueError("pairwise_winrate.csv: no pr_auc rows found")
    validate_no_nan("pairwise_winrate.csv", pr_winrate, STRUCTURAL_REQUIRED_COLUMNS["pairwise_winrate.csv"])
    coerce_numeric(
        "pairwise_winrate.csv",
        pr_winrate,
        ("comparable_pairs", "ours_wins", "teacher_wins", "ties", "ours_winrate", "teacher_winrate", "tie_rate"),
    )
    tables[HALP_SCORE_TABLE] = load_halp_score_results(reports_dir)
    return tables


def load_halp_score_results(reports_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_paths = [reports_dir / filename for filename in HALP_RESULT_FILENAMES]
    for filename, path in zip(HALP_RESULT_FILENAMES, expected_paths, strict=True):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{filename}: report must not be empty")
        validate_required_columns(filename, df, HALP_REQUIRED_COLUMNS)

        scores = pd.to_numeric(df["score"], errors="coerce")
        score_values = scores.to_numpy(dtype=float)
        if scores.isna().any() or not np.isfinite(score_values).all():
            raise ValueError(f"{filename}: non-finite numeric values in required column: score")

        labels = pd.to_numeric(df["label"], errors="coerce")
        label_values = labels.to_numpy(dtype=float)
        invalid_labels = (
            labels.isna().any()
            or not np.isfinite(label_values).all()
            or not np.isin(label_values, [0.0, 1.0]).all()
        )
        if invalid_labels:
            raise ValueError(f"{filename}: label values must be 0 or 1")

        frame = pd.DataFrame(
            {
                "score": score_values,
                "label": label_values.astype(int),
                "result_file": filename,
            }
        )
        frames.append(frame)

    if not frames:
        expected = ", ".join(str(path) for path in expected_paths)
        raise FileNotFoundError(f"required HALP result CSV is missing: expected one of {expected}")

    results = pd.concat(frames, ignore_index=True)
    labels = set(results["label"].astype(int).unique())
    if labels != {0, 1}:
        raise ValueError("HALP score distribution requires rows for label 0 and label 1")
    return results


def validate_required_columns(filename: str, df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing required columns: {', '.join(missing)}")


def validate_no_nan(filename: str, df: pd.DataFrame, columns: Iterable[str]) -> None:
    column_list = list(columns)
    if df[column_list].isna().any().any():
        bad = [column for column in column_list if df[column].isna().any()]
        raise ValueError(f"{filename}: NaN values in required columns: {', '.join(bad)}")


def coerce_numeric(filename: str, df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{filename}: non-finite numeric values in required column: {column}")


def success_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["status"].astype(str).str.lower().eq("success")].copy()
    if rows.empty:
        raise ValueError("report contains no success rows")
    return rows


def comparable_wide_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[
        df["paired_status"].astype(str).str.lower().eq("success")
        & df["teacher_status"].astype(str).str.lower().eq("success")
        & df["ours_status"].astype(str).str.lower().eq("success")
    ].copy()
    if rows.empty:
        raise ValueError("metrics_wide_paired.csv: no comparable Teacher/Ours success rows")
    for column in (*WIDE_SUCCESS_METRIC_COLUMNS, "teacher_total_seconds", "ours_total_seconds"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def make_figures(tables: dict[str, pd.DataFrame], *, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    specs = (
        ("teacher_ours_paired_delta_pr_auc_by_block", plot_paired_delta_by_block),
        ("block_winrate_bar", plot_block_winrate),
        ("best_method_pr_auc_comparison", plot_best_method_comparison),
        ("runtime_teacher_vs_ours", plot_runtime_comparison),
        ("feature_group_attribution_proxy", plot_feature_group_attribution_proxy),
        ("halp_score_distribution_bell", plot_halp_score_distribution_bell),
    )
    for stem, plotter in specs:
        fig = plotter(tables)
        generated.extend(save_figure(fig, output_dir=output_dir, stem=stem, formats=formats))
        plt.close(fig)
    return generated


def save_figure(fig: plt.Figure, *, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = 220
        fig.savefig(path, format=fmt, **kwargs)
        paths.append(path)
    return paths


def plot_paired_delta_by_block(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    wide = comparable_wide_rows(tables["metrics_wide_paired.csv"])
    blocks = sorted(wide["block"].astype(str).unique(), key=block_sort_key)
    data = [wide.loc[wide["block"].astype(str).eq(block), "delta_pr_auc"].to_numpy(dtype=float) for block in blocks]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    positions = np.arange(len(blocks), dtype=float)
    ax.boxplot(
        data,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.6},
        boxprops={"facecolor": "#D9E8F5", "edgecolor": COLORS["teacher"], "linewidth": 1.2},
        whiskerprops={"color": COLORS["teacher"], "linewidth": 1.1},
        capprops={"color": COLORS["teacher"], "linewidth": 1.1},
    )
    for index, values in enumerate(data):
        jitter = deterministic_jitter(len(values), width=0.16)
        ax.scatter(
            np.full(len(values), positions[index]) + jitter,
            values,
            s=42,
            color=COLORS["ours"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.scatter(
            [positions[index]],
            [float(np.mean(values))],
            marker="D",
            s=62,
            color=COLORS["neutral"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    ax.axhline(0.0, color=COLORS["zero"], linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_title("Teacher/Ours Paired Delta PR-AUC by Block", fontsize=16, pad=12)
    ax.set_ylabel("Ours minus Teacher PR-AUC", fontsize=12)
    ax.set_xlabel("Block", fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(blocks)
    style_axes(ax)
    return fig


def plot_block_winrate(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    winrate = tables["pairwise_winrate.csv"]
    rows = winrate[
        winrate["metric"].astype(str).str.lower().eq("pr_auc")
        & ~winrate["block"].astype(str).str.lower().eq("overall")
    ].copy()
    if rows.empty:
        raise ValueError("pairwise_winrate.csv: no block-level pr_auc rows found")
    rows = rows.sort_values("block", key=lambda series: series.map(block_sort_key))
    for column in ("ours_winrate", "teacher_winrate", "tie_rate"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    validate_no_nan("pairwise_winrate.csv", rows, ("ours_winrate", "teacher_winrate", "tie_rate"))

    blocks = rows["block"].astype(str).tolist()
    x = np.arange(len(blocks), dtype=float)
    ours = rows["ours_winrate"].to_numpy(dtype=float)
    teacher = rows["teacher_winrate"].to_numpy(dtype=float)
    ties = rows["tie_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.bar(x, ours, width=0.64, label="Ours wins", color=COLORS["ours"])
    ax.bar(x, teacher, width=0.64, bottom=ours, label="Teacher wins", color=COLORS["teacher"])
    ax.bar(x, ties, width=0.64, bottom=ours + teacher, label="Ties", color="#B8B8B8")
    for index, value in enumerate(ours):
        ax.text(index, min(value + 0.035, 0.98), f"{value:.0%}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Block Win Rate on PR-AUC", fontsize=16, pad=12)
    ax.set_ylabel("Share of comparable pairs", fontsize=12)
    ax.set_xlabel("Block", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    style_axes(ax)
    return fig


def plot_best_method_comparison(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    labels, values, _details = best_method_rows(tables)
    x = np.arange(len(labels), dtype=float)
    colors = [COLORS["teacher"], COLORS["ours"], COLORS["halp"], COLORS["halp"], COLORS["linear"]]

    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    bars = ax.bar(x, values, width=0.64, color=colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Best Method Comparison on PR-AUC", fontsize=16, pad=12)
    ax.set_ylabel("Best test PR-AUC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(1.0, float(max(values)) + 0.10))
    style_axes(ax)
    return fig


def best_method_rows(tables: dict[str, pd.DataFrame]) -> tuple[list[str], np.ndarray, list[str]]:
    long_success = success_rows(tables["metrics_long.csv"]).copy()
    domain_success = success_rows(tables["domain_baselines.csv"]).copy()
    long_success["test_pr_auc"] = pd.to_numeric(long_success["test_pr_auc"], errors="coerce")
    domain_success["test_pr_auc"] = pd.to_numeric(domain_success["test_pr_auc"], errors="coerce")

    teacher = best_long_row(long_success, "Teacher")
    ours = best_long_row(long_success, "Ours")
    halp_row = best_domain_row(
        domain_success,
        "HALP official-row",
        lambda df: df["baseline_name"].astype(str).eq("halp_official_row_protocol")
        | df["source"].astype(str).eq("halp_official_legacy_row_protocol"),
    )
    halp_course = best_domain_row(
        domain_success,
        "HALP course-grouped",
        lambda df: df["baseline_name"].astype(str).eq("halp_official_mlp")
        | (
            df["method_family"].astype(str).eq("halp_official")
            & ~df["baseline_name"].astype(str).eq("halp_official_row_protocol")
        ),
    )
    linear = best_domain_row(
        domain_success,
        "Linear probe",
        lambda df: df["method_family"].astype(str).eq("linear_probe"),
    )

    rows = (teacher, ours, halp_row, halp_course, linear)
    labels = ["Teacher", "Ours", "HALP official-row", "HALP course-grouped", "Linear"]
    values = np.asarray([float(row["test_pr_auc"]) for row in rows], dtype=float)
    details = [
        str(teacher["config_name"]),
        str(ours["config_name"]),
        str(halp_row["baseline_name"]),
        str(halp_course["baseline_name"]),
        str(linear["baseline_name"]),
    ]
    if not np.isfinite(values).all():
        raise ValueError("best method comparison contains non-finite PR-AUC")
    return labels, values, details


def best_long_row(df: pd.DataFrame, source: str) -> pd.Series:
    rows = df[df["source"].astype(str).eq(source)].copy()
    if rows.empty:
        raise ValueError(f"metrics_long.csv: no success rows for source {source}")
    index = rows["test_pr_auc"].idxmax()
    return rows.loc[index]


def best_domain_row(df: pd.DataFrame, label: str, selector) -> pd.Series:
    rows = df[selector(df)].copy()
    if rows.empty:
        raise ValueError(f"domain_baselines.csv: no success rows for {label}")
    index = rows["test_pr_auc"].idxmax()
    return rows.loc[index]


def plot_runtime_comparison(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    wide = comparable_wide_rows(tables["metrics_wide_paired.csv"])
    blocks = sorted(wide["block"].astype(str).unique(), key=block_sort_key)
    teacher = np.asarray(
        [wide.loc[wide["block"].astype(str).eq(block), "teacher_total_seconds"].median() for block in blocks],
        dtype=float,
    )
    ours = np.asarray(
        [wide.loc[wide["block"].astype(str).eq(block), "ours_total_seconds"].median() for block in blocks],
        dtype=float,
    )
    if (teacher <= 0).any() or (ours <= 0).any():
        raise ValueError("metrics_wide_paired.csv: runtime values must be positive")

    x = np.arange(len(blocks), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.bar(x - width / 2, teacher, width=width, label="Teacher", color=COLORS["teacher"])
    ax.bar(x + width / 2, ours, width=width, label="Ours", color=COLORS["ours"])
    for index, (teacher_seconds, ours_seconds) in enumerate(zip(teacher, ours, strict=True)):
        ratio = teacher_seconds / ours_seconds
        ax.text(index, max(teacher_seconds, ours_seconds) * 1.08, f"{ratio:.1f}x", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Runtime Comparison: Teacher vs Ours", fontsize=16, pad=12)
    ax.set_ylabel("Median total seconds per pair (log scale)", fontsize=12)
    ax.set_xlabel("Block", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    style_axes(ax)
    return fig


def plot_feature_group_attribution_proxy(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    samples = feature_group_attribution_samples(tables)
    labels = [FACTOR_LABELS[factor] for factor in FACTOR_COLUMNS]
    data = [samples[factor] for factor in FACTOR_COLUMNS]
    positions = np.arange(1, len(labels) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    violin = ax.violinplot(
        data,
        positions=positions,
        orientation="horizontal",
        widths=0.72,
        showmeans=True,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#D7E8D2")
        body.set_edgecolor(COLORS["halp"])
        body.set_alpha(0.82)
    violin["cmeans"].set_color(COLORS["zero"])
    violin["cmeans"].set_linewidth(1.4)

    for index, values in enumerate(data, start=1):
        y = np.full(len(values), float(index)) + deterministic_jitter(len(values), width=0.18)
        ax.scatter(values, y, s=35, color=COLORS["neutral"], alpha=0.78, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0.0, color=COLORS["zero"], linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_title("Feature-Group Attribution Proxy (Not SHAP)", fontsize=16, pad=12)
    ax.set_xlabel("Block-centered mean delta PR-AUC by factor value", fontsize=12)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    style_axes(ax)
    return fig


def plot_halp_score_distribution_bell(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    rows = tables[HALP_SCORE_TABLE]
    scores = rows["score"].to_numpy(dtype=float)
    grid = score_density_grid(scores)

    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    for label, color in ((0, COLORS["teacher"]), (1, COLORS["ours"])):
        values = rows.loc[rows["label"].astype(int).eq(label), "score"].to_numpy(dtype=float)
        density = gaussian_kernel_density(values, grid, filename=HALP_SCORE_TABLE, label=label)
        ax.plot(grid, density, color=color, linewidth=2.1, label=f"label={label} (n={len(values)})")
        ax.fill_between(grid, density, color=color, alpha=0.18)

    ax.set_title("HALP Score Distribution (Not SHAP)", fontsize=16, pad=12)
    ax.set_xlabel("HALP score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    style_axes(ax)
    return fig


def score_density_grid(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("HALP score distribution contains non-finite scores")
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    if score_min == score_max:
        raise ValueError("HALP score distribution requires more than one score value")
    if 0.0 <= score_min and score_max <= 1.0:
        return np.linspace(0.0, 1.0, num=400, dtype=float)
    span = score_max - score_min
    padding = span * 0.08
    return np.linspace(score_min - padding, score_max + padding, num=400, dtype=float)


def gaussian_kernel_density(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    filename: str,
    label: int,
) -> np.ndarray:
    if values.size < 2:
        raise ValueError(f"{filename}: label {label} requires at least two scores for density")
    if not np.isfinite(values).all():
        raise ValueError(f"{filename}: label {label} contains non-finite scores")

    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError(f"{filename}: label {label} scores need non-zero spread for density")
    bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"{filename}: label {label} has invalid density bandwidth")

    scaled = (grid[:, np.newaxis] - values[np.newaxis, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2.0 * np.pi)
    if not np.isfinite(density).all():
        raise ValueError(f"{filename}: label {label} produced non-finite density")
    return density


def feature_group_attribution_samples(tables: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    wide = comparable_wide_rows(tables["metrics_wide_paired.csv"])
    long = success_rows(tables["metrics_long.csv"])
    factor_rows = long[long["source"].astype(str).eq("Teacher")].copy()
    if factor_rows.empty:
        raise ValueError("metrics_long.csv: no Teacher success rows for attribution proxy")

    factor_rows = factor_rows[["block", "pair_id", *FACTOR_COLUMNS]].drop_duplicates()
    duplicates = factor_rows.duplicated(["block", "pair_id"], keep=False)
    if duplicates.any():
        bad = factor_rows.loc[duplicates, ["block", "pair_id"]].head(5).to_dict("records")
        raise ValueError(f"metrics_long.csv: duplicate factor rows for attribution proxy: {bad}")

    merged = wide[["block", "pair_id", "delta_pr_auc"]].merge(
        factor_rows,
        on=["block", "pair_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(wide):
        raise ValueError("metrics_long.csv: missing factor rows for comparable paired rows")
    validate_no_nan("metrics_long.csv", merged, FACTOR_COLUMNS)

    block_mean = merged.groupby("block")["delta_pr_auc"].mean()
    samples: dict[str, np.ndarray] = {}
    for factor in FACTOR_COLUMNS:
        values: list[float] = []
        grouped = merged.groupby(["block", factor], dropna=False)["delta_pr_auc"].mean()
        for (block, _factor_value), mean_delta in grouped.items():
            effect = float(mean_delta) - float(block_mean.loc[block])
            if not np.isfinite(effect):
                raise ValueError("feature-group attribution proxy contains non-finite effect")
            values.append(effect)
        if not values:
            raise ValueError(f"feature-group attribution proxy has no samples for factor {factor}")
        samples[factor] = np.asarray(values, dtype=float)
    return samples


def deterministic_jitter(size: int, *, width: float) -> np.ndarray:
    if size <= 1:
        return np.zeros(size, dtype=float)
    return np.linspace(-width, width, num=size, dtype=float)


def block_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    return (BLOCK_ORDER.get(text, 50), text)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D7DCE0", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB6BC")
    ax.spines["bottom"].set_color("#AEB6BC")
    ax.tick_params(axis="both", labelsize=10)


if __name__ == "__main__":
    raise SystemExit(main())
