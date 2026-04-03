from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_paper_package.py"
SPEC = importlib.util.spec_from_file_location("export_paper_package", SCRIPT_PATH)
paper_export = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(paper_export)


def _write_report(
    reports_root: Path,
    experiment_name: str,
    *,
    metrics: dict[str, float],
    baselines: dict[str, dict[str, float]] | None = None,
    results: pd.DataFrame | None = None,
    subset: str = "popular",
) -> None:
    report_root = reports_root / experiment_name
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {"overall": metrics, "by_subset": {subset: metrics}}
    (report_root / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if baselines is not None:
        (report_root / "baselines.json").write_text(
            json.dumps(baselines, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if results is not None:
        results.to_csv(report_root / "results.csv", index=False)


def _metrics(
    roc_auc: float,
    pr_auc: float,
    tpr_at_fpr_0_01: float,
    f1: float,
    accuracy: float,
    precision: float,
    recall: float,
    false_positive_rate: float,
) -> dict[str, float]:
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tpr_at_fpr_0.01": tpr_at_fpr_0_01,
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
    }


def _results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "s1", "label": 0, "prediction": 0, "score": 0.05, "fold": 0},
            {"sample_id": "s2", "label": 0, "prediction": 0, "score": 0.10, "fold": 0},
            {"sample_id": "s3", "label": 1, "prediction": 1, "score": 0.70, "fold": 1},
            {"sample_id": "s4", "label": 1, "prediction": 1, "score": 0.90, "fold": 1},
        ]
    )


def test_export_paper_package_writes_closeout_tables_and_figures(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    output_root = tmp_path / "paper"
    popular_results = _results_frame()

    qwen_popular = _metrics(0.91, 0.31, 0.11, 0.0, 0.96, 0.0, 0.0, 0.0)
    qwen_shared = _metrics(0.87, 0.24, 0.08, 0.0, 0.95, 0.0, 0.0, 0.0)
    qwen_object = _metrics(0.72, 0.06, 0.02, 0.0, 0.94, 0.0, 0.0, 0.0)
    qwen_shared_object = _metrics(0.79, 0.12, 0.05, 0.0, 0.94, 0.0, 0.0, 0.0)
    qwen_row = _metrics(0.89, 0.28, 0.10, 0.0, 0.96, 0.0, 0.0, 0.0)
    qwen_repope = _metrics(0.90, 0.29, 0.10, 0.0, 0.96, 0.0, 0.0, 0.0)
    qwen_adv = _metrics(0.84, 0.18, 0.06, 0.0, 0.94, 0.0, 0.0, 0.0)

    intern_popular = _metrics(0.92, 0.54, 0.25, 0.37, 0.93, 0.68, 0.26, 0.01)
    intern_shared = _metrics(0.90, 0.49, 0.21, 0.31, 0.92, 0.61, 0.22, 0.01)
    intern_object = _metrics(0.84, 0.41, 0.22, 0.32, 0.92, 0.67, 0.21, 0.01)
    intern_shared_object = _metrics(0.86, 0.44, 0.24, 0.34, 0.92, 0.68, 0.23, 0.01)
    intern_row = _metrics(0.90, 0.55, 0.24, 0.42, 0.93, 0.65, 0.31, 0.02)
    intern_repope = _metrics(0.91, 0.52, 0.23, 0.35, 0.93, 0.66, 0.25, 0.01)
    intern_adv = _metrics(0.86, 0.35, 0.14, 0.20, 0.91, 0.50, 0.13, 0.02)

    qwen_baselines = {
        "full": qwen_popular,
        "drift_only": _metrics(0.85, 0.12, 0.03, 0.0, 0.96, 0.0, 0.0, 0.0),
        "no_manifold": _metrics(0.83, 0.20, 0.08, 0.0, 0.96, 0.0, 0.0, 0.0),
        "linear_probe": _metrics(0.92, 0.38, 0.25, 0.42, 0.96, 0.45, 0.39, 0.02),
        "output_p_yes": _metrics(0.89, 0.24, 0.12, 0.0, 0.95, 0.0, 0.0, 0.0),
        "output_logit_margin": _metrics(0.90, 0.26, 0.15, 0.0, 0.95, 0.0, 0.0, 0.0),
        "output_chosen_answer_confidence": _metrics(0.88, 0.22, 0.10, 0.0, 0.95, 0.0, 0.0, 0.0),
        "raw_curve_only": _metrics(0.86, 0.17, 0.07, 0.0, 0.95, 0.0, 0.0, 0.0),
        "raw_plus_calibrated_simple": _metrics(0.90, 0.27, 0.13, 0.0, 0.95, 0.0, 0.0, 0.0),
        "raw_plus_calibrated_full_curve": _metrics(0.91, 0.29, 0.14, 0.0, 0.95, 0.0, 0.0, 0.0),
        "raw_plus_calibrated_haar": _metrics(0.91, 0.28, 0.14, 0.0, 0.95, 0.0, 0.0, 0.0),
    }
    qwen_object_baselines = {
        "full": qwen_object,
        "linear_probe": _metrics(0.74, 0.08, 0.03, 0.12, 0.93, 0.15, 0.10, 0.03),
    }
    intern_baselines = {
        "full": intern_popular,
        "drift_only": _metrics(0.88, 0.43, 0.17, 0.11, 0.92, 0.64, 0.06, 0.0),
        "no_manifold": _metrics(0.86, 0.40, 0.16, 0.24, 0.92, 0.60, 0.16, 0.01),
        "linear_probe": _metrics(0.94, 0.66, 0.32, 0.64, 0.93, 0.56, 0.73, 0.05),
        "output_p_yes": _metrics(0.90, 0.48, 0.21, 0.0, 0.92, 0.0, 0.0, 0.0),
        "output_logit_margin": _metrics(0.91, 0.50, 0.23, 0.0, 0.92, 0.0, 0.0, 0.0),
        "output_chosen_answer_confidence": _metrics(0.89, 0.47, 0.20, 0.0, 0.92, 0.0, 0.0, 0.0),
        "raw_curve_only": _metrics(0.89, 0.45, 0.18, 0.0, 0.92, 0.0, 0.0, 0.0),
        "raw_plus_calibrated_simple": _metrics(0.91, 0.51, 0.24, 0.0, 0.92, 0.0, 0.0, 0.0),
        "raw_plus_calibrated_full_curve": _metrics(0.92, 0.53, 0.25, 0.0, 0.92, 0.0, 0.0, 0.0),
        "raw_plus_calibrated_haar": _metrics(0.92, 0.52, 0.25, 0.0, 0.92, 0.0, 0.0, 0.0),
    }
    intern_object_baselines = {
        "full": intern_object,
        "linear_probe": _metrics(0.78, 0.27, 0.08, 0.36, 0.87, 0.32, 0.43, 0.08),
    }

    _write_report(reports_root, "correction-qwen3-vl-8b-popular", metrics=qwen_popular, baselines=qwen_baselines, results=popular_results)
    _write_report(reports_root, "correction-qwen3-vl-8b-popular-shared", metrics=qwen_shared)
    _write_report(reports_root, "correction-qwen3-vl-8b-popular-object-heldout", metrics=qwen_object, baselines=qwen_object_baselines)
    _write_report(reports_root, "correction-qwen3-vl-8b-popular-shared-object-heldout", metrics=qwen_shared_object)
    _write_report(reports_root, "correction-qwen3-vl-8b-popular-row", metrics=qwen_row)
    _write_report(reports_root, "correction-qwen3-vl-8b-popular-repope", metrics=qwen_repope)
    _write_report(reports_root, "correction-qwen3-vl-8b-adversarial", metrics=qwen_adv, subset="adversarial")

    _write_report(reports_root, "correction-internvl3.5-8b-popular", metrics=intern_popular, baselines=intern_baselines, results=popular_results)
    _write_report(reports_root, "correction-internvl3.5-8b-popular-shared", metrics=intern_shared)
    _write_report(reports_root, "correction-internvl3.5-8b-popular-object-heldout", metrics=intern_object, baselines=intern_object_baselines)
    _write_report(reports_root, "correction-internvl3.5-8b-popular-shared-object-heldout", metrics=intern_shared_object)
    _write_report(reports_root, "correction-internvl3.5-8b-popular-row", metrics=intern_row)
    _write_report(reports_root, "correction-internvl3.5-8b-popular-repope", metrics=intern_repope)
    _write_report(reports_root, "correction-internvl3.5-8b-adversarial", metrics=intern_adv, subset="adversarial")

    outputs = paper_export.export_paper_package(
        reports_root=reports_root,
        output_root=output_root,
    )

    table1 = pd.read_csv(outputs["table1_csv"])
    table2 = pd.read_csv(outputs["table2_csv"])
    table3 = pd.read_csv(outputs["table3_csv"])
    table4 = pd.read_csv(outputs["table4_csv"])
    manifest = json.loads(outputs["figure_manifest"].read_text(encoding="utf-8"))

    assert list(table1.columns)[1:] == paper_export.METRIC_ORDER
    assert len(table1) == 6
    assert len(table2) == 10
    assert len(table3) == 6
    assert len(table4) == 18

    qwen_repope_row = table1.loc[table1["setting"] == "Qwen popular + RePOPE"].iloc[0]
    assert qwen_repope_row["roc_auc"] == qwen_repope["roc_auc"]
    assert qwen_repope_row["pr_auc"] == qwen_repope["pr_auc"]

    intern_shared_row = table2.loc[
        (table2["model"] == "InternVL3.5-8B") & (table2["variant"] == "full MIND (shared bank)")
    ].iloc[0]
    assert intern_shared_row["roc_auc"] == intern_shared["roc_auc"]
    assert intern_shared_row["pr_auc"] == intern_shared["pr_auc"]

    qwen_linear_object_row = table3.loc[
        (table3["model"] == "Qwen3-VL-8B") & (table3["variant"] == "linear probe")
    ].iloc[0]
    assert qwen_linear_object_row["roc_auc"] == qwen_object_baselines["linear_probe"]["roc_auc"]

    qwen_output_baseline_row = table4.loc[
        (table4["model"] == "Qwen3-VL-8B") & (table4["variant"] == "output baseline: p(yes)")
    ].iloc[0]
    assert qwen_output_baseline_row["roc_auc"] == qwen_baselines["output_p_yes"]["roc_auc"]

    assert sorted(manifest) == ["figure1", "figure2", "figure3"]
    for payload in manifest.values():
        assert Path(payload["path"]).exists()
        assert Path(payload["path"]).stat().st_size > 0
