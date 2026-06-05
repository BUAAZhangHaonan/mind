from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def _load_script():
    script_path = Path("scripts/wavelet_course_make_ppt_figures.py")
    spec = importlib.util.spec_from_file_location("wavelet_course_make_ppt_figures", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_minimal_reports(reports_dir: Path) -> None:
    wide_rows = [
        {
            "block": "A",
            "pair_id": "A_dwt_db2_l2",
            "paired_status": "success",
            "teacher_status": "success",
            "ours_status": "success",
            "teacher_pr_auc": 0.40,
            "ours_pr_auc": 0.52,
            "delta_pr_auc": 0.12,
            "teacher_total_seconds": 100.0,
            "ours_total_seconds": 40.0,
        },
        {
            "block": "A",
            "pair_id": "A_cwt_morl",
            "paired_status": "success",
            "teacher_status": "success",
            "ours_status": "success",
            "teacher_pr_auc": 0.45,
            "ours_pr_auc": 0.42,
            "delta_pr_auc": -0.03,
            "teacher_total_seconds": 120.0,
            "ours_total_seconds": 60.0,
        },
        {
            "block": "B",
            "pair_id": "B_raw_sequence",
            "paired_status": "success",
            "teacher_status": "success",
            "ours_status": "success",
            "teacher_pr_auc": 0.55,
            "ours_pr_auc": 0.62,
            "delta_pr_auc": 0.07,
            "teacher_total_seconds": 80.0,
            "ours_total_seconds": 50.0,
        },
    ]
    _write_csv(reports_dir / "metrics_wide_paired.csv", wide_rows)

    winrate_rows = [
        {
            "block": "A",
            "metric": "pr_auc",
            "comparable_pairs": 2,
            "ours_wins": 1,
            "teacher_wins": 1,
            "ties": 0,
            "ours_winrate": 0.50,
            "teacher_winrate": 0.50,
            "tie_rate": 0.0,
        },
        {
            "block": "B",
            "metric": "pr_auc",
            "comparable_pairs": 1,
            "ours_wins": 1,
            "teacher_wins": 0,
            "ties": 0,
            "ours_winrate": 1.0,
            "teacher_winrate": 0.0,
            "tie_rate": 0.0,
        },
        {
            "block": "overall",
            "metric": "pr_auc",
            "comparable_pairs": 3,
            "ours_wins": 2,
            "teacher_wins": 1,
            "ties": 0,
            "ours_winrate": 2 / 3,
            "teacher_winrate": 1 / 3,
            "tie_rate": 0.0,
        },
    ]
    _write_csv(reports_dir / "pairwise_winrate.csv", winrate_rows)

    long_rows: list[dict[str, object]] = []
    for wide in wide_rows:
        for source, pr_auc in (("Teacher", wide["teacher_pr_auc"]), ("Ours", wide["ours_pr_auc"])):
            long_rows.append(
                {
                    "block": wide["block"],
                    "pair_id": wide["pair_id"],
                    "source": source,
                    "status": "success",
                    "config_name": f"{wide['pair_id']}::{source}",
                    "transform": "dwt" if "dwt" in str(wide["pair_id"]) else "cwt",
                    "feature_protocol": "wavelet_summary_static_pooled",
                    "window_strategy": "full",
                    "window_mode": "global",
                    "threshold": "universal_soft",
                    "classifier": "logreg",
                    "test_pr_auc": pr_auc,
                    "total_seconds": (
                        wide["teacher_total_seconds"] if source == "Teacher" else wide["ours_total_seconds"]
                    ),
                }
            )
    _write_csv(reports_dir / "metrics_long.csv", long_rows)

    domain_rows = [
        {
            "baseline_name": "halp_official_mlp",
            "method_family": "halp_official",
            "source": "halp_official",
            "status": "success",
            "test_pr_auc": 0.58,
            "total_seconds": 200.0,
        },
        {
            "baseline_name": "halp_official_row_protocol",
            "method_family": "halp_official",
            "source": "halp_official_legacy_row_protocol",
            "status": "success",
            "test_pr_auc": 0.71,
            "total_seconds": 240.0,
        },
        {
            "baseline_name": "linear_probe_final_hidden_logreg",
            "method_family": "linear_probe",
            "source": "linear_probe",
            "status": "success",
            "test_pr_auc": 0.49,
            "total_seconds": 8.0,
        },
    ]
    _write_csv(reports_dir / "domain_baselines.csv", domain_rows)

    halp_rows = [
        {"sample_id": 1, "label": 0, "score": 0.02},
        {"sample_id": 2, "label": 0, "score": 0.08},
        {"sample_id": 3, "label": 0, "score": 0.12},
        {"sample_id": 4, "label": 0, "score": 0.18},
        {"sample_id": 5, "label": 1, "score": 0.70},
        {"sample_id": 6, "label": 1, "score": 0.78},
        {"sample_id": 7, "label": 1, "score": 0.86},
        {"sample_id": 8, "label": 1, "score": 0.94},
    ]
    _write_csv(reports_dir / "halp_results.csv", halp_rows)


def test_cli_defaults_and_format_parsing() -> None:
    module = _load_script()

    args = module.build_parser().parse_args(["--format", "png,pdf"])

    assert args.reports_dir == Path("outputs/wavelet_course_v2/reports")
    assert args.output_dir == Path("outputs/wavelet_course_v2/figures/ppt")
    assert args.format == "png,pdf"


def test_script_generates_required_ppt_figures(tmp_path: Path) -> None:
    module = _load_script()
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    _write_minimal_reports(reports_dir)

    assert module.main(
        [
            "--reports-dir",
            str(reports_dir),
            "--output-dir",
            str(output_dir),
            "--format",
            "png,pdf",
        ]
    ) == 0

    expected_stems = {
        "teacher_ours_paired_delta_pr_auc_by_block",
        "block_winrate_bar",
        "best_method_pr_auc_comparison",
        "runtime_teacher_vs_ours",
        "feature_group_attribution_proxy",
        "halp_score_distribution_bell",
    }
    for stem in expected_stems:
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0


def test_report_loading_rejects_missing_required_columns(tmp_path: Path) -> None:
    module = _load_script()
    reports_dir = tmp_path / "reports"
    _write_minimal_reports(reports_dir)
    _write_csv(reports_dir / "pairwise_winrate.csv", [{"block": "A", "metric": "pr_auc"}])

    with pytest.raises(ValueError, match="pairwise_winrate.csv: missing required columns"):
        module.load_reports(reports_dir)


def test_report_loading_rejects_nan_in_required_columns(tmp_path: Path) -> None:
    module = _load_script()
    reports_dir = tmp_path / "reports"
    _write_minimal_reports(reports_dir)
    rows = list(csv.DictReader((reports_dir / "metrics_wide_paired.csv").open(newline="", encoding="utf-8")))
    rows[0]["delta_pr_auc"] = ""
    _write_csv(reports_dir / "metrics_wide_paired.csv", rows)

    with pytest.raises(ValueError, match="metrics_wide_paired.csv: NaN values in required columns"):
        module.load_reports(reports_dir)


def test_halp_score_distribution_title_is_not_shap(tmp_path: Path) -> None:
    module = _load_script()
    reports_dir = tmp_path / "reports"
    _write_minimal_reports(reports_dir)
    tables = module.load_reports(reports_dir)

    fig = module.plot_halp_score_distribution_bell(tables)

    title = fig.axes[0].get_title()
    assert "Score Distribution" in title
    assert "Not SHAP" in title


@pytest.mark.parametrize(
    ("rows", "expected_error", "expected_message"),
    [
        (None, FileNotFoundError, "required HALP result CSV is missing"),
        ([{"label": 0}], ValueError, "halp_results.csv: missing required columns: score"),
        ([{"score": 0.1}], ValueError, "halp_results.csv: missing required columns: label"),
        ([{"label": 0, "score": "inf"}], ValueError, "halp_results.csv: non-finite numeric values"),
        ([{"label": 2, "score": 0.1}], ValueError, "halp_results.csv: label values must be 0 or 1"),
    ],
)
def test_halp_result_loading_fails_closed(
    tmp_path: Path,
    rows: list[dict[str, object]] | None,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    module = _load_script()
    reports_dir = tmp_path / "reports"
    _write_minimal_reports(reports_dir)
    path = reports_dir / "halp_results.csv"
    if rows is None:
        path.unlink()
    else:
        _write_csv(path, rows)

    with pytest.raises(expected_error, match=expected_message):
        module.load_reports(reports_dir)
