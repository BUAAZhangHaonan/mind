from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from mind.evaluation.metrics import write_results_table


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_paper_package.py"
SPEC = importlib.util.spec_from_file_location("export_paper_package", SCRIPT_PATH)
paper_export = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = paper_export
SPEC.loader.exec_module(paper_export)


MODELS = list(paper_export.ROUND_TWO_MODEL_LABELS)
POPULAR = "popular"
DASH_B = "dash-b"


def _metric_payload(roc_auc: float, pr_auc: float) -> dict[str, object]:
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confidence_intervals": {
            "roc_auc": {"lower": roc_auc - 0.01, "upper": roc_auc + 0.01},
            "pr_auc": {"lower": pr_auc - 0.02, "upper": pr_auc + 0.02},
        },
    }


def _results_frame(label_shift: float = 0.0, *, extra_column: str | None = None, extra_value: str = "") -> pd.DataFrame:
    rows = [
        {"sample_id": "s1", "image_id": 1, "object_name": "dog", "subset": POPULAR, "label": 0, "prediction": 0, "score": 0.05 + label_shift, "fold": 0},
        {"sample_id": "s2", "image_id": 1, "object_name": "dog", "subset": POPULAR, "label": 1, "prediction": 1, "score": 0.95 - label_shift, "fold": 0},
        {"sample_id": "s3", "image_id": 2, "object_name": "cat", "subset": POPULAR, "label": 0, "prediction": 0, "score": 0.10 + label_shift, "fold": 1},
        {"sample_id": "s4", "image_id": 2, "object_name": "cat", "subset": POPULAR, "label": 1, "prediction": 1, "score": 0.90 - label_shift, "fold": 1},
    ]
    frame = pd.DataFrame(rows)
    if extra_column is not None:
        frame[extra_column] = extra_value
    return frame


def _baseline_variants(base_roc: float, base_pr: float) -> dict[str, dict[str, object]]:
    variant_deltas = {
        "full": (0.00, 0.00),
        "drift_only": (-0.05, -0.06),
        "no_manifold": (-0.08, -0.08),
        "linear_probe": (0.04, 0.03),
        "output_p_yes": (-0.14, -0.15),
        "output_logit_margin": (-0.12, -0.13),
        "output_chosen_answer_confidence": (-0.13, -0.14),
        "raw_curve_only": (-0.03, -0.03),
        "raw_plus_calibrated_simple": (0.01, 0.01),
        "raw_plus_calibrated_full_curve": (0.02, 0.02),
        "raw_plus_calibrated_haar": (-0.01, -0.01),
    }
    payload: dict[str, dict[str, object]] = {}
    for variant, (roc_delta, pr_delta) in variant_deltas.items():
        roc_auc = round(base_roc + roc_delta, 4)
        pr_auc = round(base_pr + pr_delta, 4)
        payload[variant] = {
            **_metric_payload(roc_auc, pr_auc),
            "result_path": f"variant_results/{variant}.csv",
        }
    return payload


def _write_baseline_report(
    reports_root: Path,
    *,
    model_key: str,
    benchmark_key: str,
    protocol: str,
    bank_scope: str,
    base_roc: float,
    base_pr: float,
    label_shift: float,
) -> Path:
    report_root = reports_root / f"round2-{model_key}-{benchmark_key}"
    if bank_scope != "object":
        report_root = reports_root / f"{report_root.name}-{bank_scope}"
    if protocol != "image_grouped":
        report_root = reports_root / f"{report_root.name}-{protocol}"
    report_root.mkdir(parents=True, exist_ok=True)

    variants = _baseline_variants(base_roc, base_pr)
    (report_root / "baselines.json").write_text(
        json.dumps({"bank_scope": bank_scope, "full_variant": "raw_plus_calibrated_simple", **variants}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    ablations = pd.DataFrame(
        [
            {"variant": "raw_curve_only", **_metric_payload(base_roc - 0.03, base_pr - 0.03)},
            {"variant": "raw_plus_calibrated_simple", **_metric_payload(base_roc + 0.01, base_pr + 0.01)},
            {"variant": "raw_plus_calibrated_full_curve", **_metric_payload(base_roc + 0.02, base_pr + 0.02)},
            {"variant": "raw_plus_calibrated_haar", **_metric_payload(base_roc - 0.01, base_pr - 0.01)},
        ]
    )
    ablations.to_csv(report_root / "ablations.csv", index=False)

    split_sensitivity = pd.DataFrame(
        [
            {"variant": "full", "seed": 13, "roc_auc": base_roc, "pr_auc": base_pr},
            {"variant": "full", "seed": 17, "roc_auc": base_roc + 0.01, "pr_auc": base_pr + 0.01},
        ]
    )
    split_sensitivity.to_csv(report_root / "split_sensitivity.csv", index=False)

    variant_frames = {
        "full": _results_frame(label_shift=label_shift),
        "drift_only": _results_frame(label_shift=label_shift + 0.01),
        "no_manifold": _results_frame(label_shift=label_shift + 0.02),
        "linear_probe": _results_frame(label_shift=label_shift - 0.01),
        "output_p_yes": _results_frame(label_shift=label_shift + 0.03),
        "output_logit_margin": _results_frame(label_shift=label_shift + 0.04),
        "output_chosen_answer_confidence": _results_frame(label_shift=label_shift + 0.05),
        "raw_curve_only": _results_frame(label_shift=label_shift + 0.02),
        "raw_plus_calibrated_simple": _results_frame(label_shift=label_shift),
        "raw_plus_calibrated_full_curve": _results_frame(label_shift=label_shift - 0.005),
        "raw_plus_calibrated_haar": _results_frame(label_shift=label_shift + 0.01),
    }
    variant_root = report_root / "variant_results"
    variant_root.mkdir(parents=True, exist_ok=True)
    for variant, frame in variant_frames.items():
        write_results_table(frame, variant_root / f"{variant}.csv")
    return report_root


def _write_halp_report(
    reports_root: Path,
    *,
    model_key: str,
    benchmark_key: str,
    protocol: str,
    base_roc: float,
    base_pr: float,
    label_shift: float,
) -> Path:
    report_root = reports_root / f"round2-{model_key}-{benchmark_key}-halp"
    if protocol != "image_grouped":
        report_root = reports_root / f"{report_root.name}-{protocol}"
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **_metric_payload(round(base_roc + 0.03, 4), round(base_pr + 0.02, 4)),
        "selected_probe_counts": {"vision_only": 2, "query_token_layer_0": 1},
    }
    (report_root / "halp.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    results = _results_frame(label_shift=label_shift, extra_column="selected_probe", extra_value="vision_only")
    write_results_table(results, report_root / "halp_results.csv", extra_columns=("selected_probe",))
    pd.DataFrame(
        [
            {"fold": 0, "selected_probe": "vision_only", "inner_score": 0.9},
            {"fold": 1, "selected_probe": "vision_only", "inner_score": 0.91},
        ]
    ).to_csv(report_root / "halp_selection.csv", index=False)
    return report_root


def _write_glsim_report(
    reports_root: Path,
    *,
    model_key: str,
    benchmark_key: str,
    protocol: str,
    base_roc: float,
    base_pr: float,
    label_shift: float,
) -> Path:
    report_root = reports_root / f"round2-{model_key}-{benchmark_key}-glsim"
    if protocol != "image_grouped":
        report_root = reports_root / f"{report_root.name}-{protocol}"
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **_metric_payload(round(base_roc + 0.02, 4), round(base_pr + 0.01, 4)),
        "selected_config_counts": {"i0_t0_k1_w0.50": 3},
    }
    (report_root / "glsim.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    results = _results_frame(label_shift=label_shift, extra_column="selected_config", extra_value="i0_t0_k1_w0.50")
    write_results_table(results, report_root / "glsim_results.csv", extra_columns=("selected_config",))
    pd.DataFrame(
        [
            {"fold": 0, "selected_config": "i0_t0_k1_w0.50", "inner_score": 0.88},
            {"fold": 1, "selected_config": "i0_t0_k1_w0.50", "inner_score": 0.89},
        ]
    ).to_csv(report_root / "glsim_selection.csv", index=False)
    return report_root


def _seed_round_two_reports(reports_root: Path) -> None:
    base_by_model = {
        "qwen3-vl-8b": (0.89, 0.17),
        "internvl3.5-8b": (0.90, 0.51),
        "llava-onevision-7b": (0.88, 0.42),
        "molmo-7b-d-0924": (0.87, 0.38),
    }
    for index, model_key in enumerate(MODELS):
        base_roc, base_pr = base_by_model[model_key]
        label_shift = 0.01 * index
        for benchmark_key in (POPULAR, DASH_B):
            _write_baseline_report(
                reports_root,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="image_grouped",
                bank_scope="object",
                base_roc=base_roc + (0.01 if benchmark_key == DASH_B else 0.0),
                base_pr=base_pr + (0.02 if benchmark_key == DASH_B else 0.0),
                label_shift=label_shift,
            )
            _write_baseline_report(
                reports_root,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="object_heldout",
                bank_scope="object",
                base_roc=base_roc - 0.02 + (0.01 if benchmark_key == DASH_B else 0.0),
                base_pr=base_pr - 0.02 + (0.02 if benchmark_key == DASH_B else 0.0),
                label_shift=label_shift,
            )
            _write_halp_report(
                reports_root,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="image_grouped",
                base_roc=base_roc,
                base_pr=base_pr,
                label_shift=label_shift,
            )
            _write_halp_report(
                reports_root,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="object_heldout",
                base_roc=base_roc - 0.02,
                base_pr=base_pr - 0.02,
                label_shift=label_shift,
            )
            _write_glsim_report(
                reports_root,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="image_grouped",
                base_roc=base_roc,
                base_pr=base_pr,
                label_shift=label_shift,
            )
            _write_glsim_report(
                reports_root,
                model_key=model_key,
                benchmark_key=benchmark_key,
                protocol="object_heldout",
                base_roc=base_roc - 0.02,
                base_pr=base_pr - 0.02,
                label_shift=label_shift,
            )
        for bank_scope, offset in [("shared", -0.01), ("shuffled_object", -0.03)]:
            _write_baseline_report(
                reports_root,
                model_key=model_key,
                benchmark_key=POPULAR,
                protocol="image_grouped",
                bank_scope=bank_scope,
                base_roc=base_roc + offset,
                base_pr=base_pr + offset,
                label_shift=label_shift,
            )
            _write_baseline_report(
                reports_root,
                model_key=model_key,
                benchmark_key=POPULAR,
                protocol="object_heldout",
                bank_scope=bank_scope,
                base_roc=base_roc + offset - 0.02,
                base_pr=base_pr + offset - 0.02,
                label_shift=label_shift,
            )
            _write_baseline_report(
                reports_root,
                model_key=model_key,
                benchmark_key=DASH_B,
                protocol="image_grouped",
                bank_scope=bank_scope,
                base_roc=base_roc + offset + 0.01,
                base_pr=base_pr + offset + 0.01,
                label_shift=label_shift,
            )
            _write_baseline_report(
                reports_root,
                model_key=model_key,
                benchmark_key=DASH_B,
                protocol="object_heldout",
                bank_scope=bank_scope,
                base_roc=base_roc + offset - 0.01,
                base_pr=base_pr + offset - 0.01,
                label_shift=label_shift,
            )


def _write_table_exports(
    *,
    frame: pd.DataFrame,
    export_csv: Path,
    export_md: Path,
    docs_csv: Path,
    docs_md: Path,
) -> None:
    _write_table_bundle(frame, export_csv=export_csv, export_md=export_md, docs_csv=docs_csv, docs_md=docs_md)


def _write_optional_exports(
    *,
    frame: pd.DataFrame,
    export_csv: Path,
    export_md: Path,
    docs_csv: Path,
    docs_md: Path,
) -> None:
    _write_table_bundle(frame, export_csv=export_csv, export_md=export_md, docs_csv=docs_csv, docs_md=docs_md)


def _build_outputs(reports_root: Path, output_root: Path, tables_root: Path) -> dict[str, Path]:
    reports = discover_round_two_reports(reports_root)

    table1 = build_main_table(reports)
    table2 = build_feature_table(reports)
    table3 = build_transfer_table(reports)
    supp_pope_adversarial = build_benchmark_table(reports, benchmark_key="adversarial")
    supp_repope = build_benchmark_table(reports, benchmark_key="repope")
    supp_dash_b = build_benchmark_table(reports, benchmark_key="dash-b", table_protocol="object_heldout")
    supp_split_sensitivity = build_supp_split_sensitivity_table(reports)

    paths = build_output_paths(output_root)
    _write_table_exports(
        frame=table1,
        export_csv=paths["table1_csv"],
        export_md=paths["table1_md"],
        docs_csv=tables_root / "table1_main.csv",
        docs_md=tables_root / "table1_main.md",
    )
    _write_table_exports(
        frame=table2,
        export_csv=paths["table2_csv"],
        export_md=paths["table2_md"],
        docs_csv=tables_root / "table2_feature_ablation.csv",
        docs_md=tables_root / "table2_feature_ablation.md",
    )
    _write_table_exports(
        frame=table3,
        export_csv=paths["table3_csv"],
        export_md=paths["table3_md"],
        docs_csv=tables_root / "table3_transfer_controls.csv",
        docs_md=tables_root / "table3_transfer_controls.md",
    )
    _write_optional_exports(
        frame=supp_pope_adversarial,
        export_csv=paths["supp_pope_adversarial_csv"],
        export_md=paths["supp_pope_adversarial_md"],
        docs_csv=tables_root / "supp_pope_adversarial.csv",
        docs_md=tables_root / "supp_pope_adversarial.md",
    )
    _write_optional_exports(
        frame=supp_repope,
        export_csv=paths["supp_repope_csv"],
        export_md=paths["supp_repope_md"],
        docs_csv=tables_root / "supp_repope.csv",
        docs_md=tables_root / "supp_repope.md",
    )
    _write_optional_exports(
        frame=supp_dash_b,
        export_csv=paths["supp_dash_b_transfer_csv"],
        export_md=paths["supp_dash_b_transfer_md"],
        docs_csv=tables_root / "supp_dash_b_transfer.csv",
        docs_md=tables_root / "supp_dash_b_transfer.md",
    )
    _write_optional_exports(
        frame=supp_split_sensitivity,
        export_csv=paths["supp_split_sensitivity_csv"],
        export_md=paths["supp_split_sensitivity_md"],
        docs_csv=tables_root / "supp_split_sensitivity.csv",
        docs_md=tables_root / "supp_split_sensitivity.md",
    )

    plot_method_diagram(paths["figure1"])
    _plot_popular_curves(reports, paths["figure2"])
    _plot_transfer_comparison(table3, paths["figure3"])

    manifest = {
        "figure1": {"title": "Round-two method diagram", "path": str(paths["figure1"])},
        "figure2": {"title": "Popular ROC curves", "path": str(paths["figure2"])},
        "figure3": {"title": "Transfer comparison", "path": str(paths["figure3"])},
    }
    paths["figure_manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def test_export_paper_package_reads_round_two_artifacts_only(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    output_root = tmp_path / "paper"
    tables_root = tmp_path / "docs" / "tables" / "round2"

    _seed_round_two_reports(reports_root)

    outputs = paper_export.export_paper_package(
        reports_root=reports_root,
        output_root=output_root,
        tables_root=tables_root,
    )

    table1 = pd.read_csv(outputs["table1_csv"])
    table1_popular = pd.read_csv(outputs["table1_pope_popular_csv"])
    table1_dash_b = pd.read_csv(outputs["table1_dash_b_csv"])
    table2 = pd.read_csv(outputs["table2_csv"])
    table3 = pd.read_csv(outputs["table3_csv"])
    supp_pope_adversarial = pd.read_csv(outputs["supp_pope_adversarial_csv"])
    supp_repope = pd.read_csv(outputs["supp_repope_csv"])
    supp_dash_b = pd.read_csv(outputs["supp_dash_b_transfer_csv"])
    supp_split_sensitivity = pd.read_csv(outputs["supp_split_sensitivity_csv"])

    assert set(table1["method"]) >= {"p_yes", "logit_margin", "chosen_confidence", "drift_only", "no_manifold", "full MIND", "linear_probe", "HALP", "GLSim"}
    assert set(table1["benchmark"]) == {"POPE popular", "DASH-B"}
    assert set(table1["protocol"]) == {"image_grouped"}
    assert set(table1_popular["benchmark"]) == {"POPE popular"}
    assert set(table1_dash_b["benchmark"]) == {"DASH-B"}

    qwen_popular_full = table1.loc[
        (table1["model"] == "Qwen3-VL-8B")
        & (table1["benchmark"] == "POPE popular")
        & (table1["method"] == "full MIND")
    ].iloc[0]
    assert qwen_popular_full["roc_auc"] == 0.89
    assert qwen_popular_full["pr_auc"] == 0.17

    assert set(table2["feature_variant"]) == {"raw_only", "raw_plus_simple_stats", "raw_plus_full_curve", "raw_plus_Haar"}
    assert set(table3["protocol"]) == {"image_grouped", "object_heldout"}
    assert set(table3["bank_scope"]) >= {"object", "shared", "shuffled_object", "linear_probe", "halp", "glsim"}

    assert not supp_split_sensitivity.empty

    assert (tables_root / "table1_main.csv").exists()
    assert (tables_root / "table1_pope_popular.csv").exists()
    assert (tables_root / "table1_dash_b.csv").exists()
    assert (tables_root / "table2_feature_ablation.csv").exists()
    assert (tables_root / "table3_transfer_controls.csv").exists()
    assert (tables_root / "supp_pope_adversarial.csv").exists()
    assert (tables_root / "supp_repope.csv").exists()
    assert (tables_root / "supp_dash_b_transfer.csv").exists()
    assert (tables_root / "supp_split_sensitivity.csv").exists()

    assert "metrics.json" not in {path.name for path in reports_root.rglob("*") if path.is_file()}
    assert "results.csv" not in {path.name for path in reports_root.rglob("*") if path.is_file()}


def test_export_prefers_most_complete_duplicate_report(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    output_root = tmp_path / "paper"
    tables_root = tmp_path / "docs" / "tables" / "round2"

    _seed_round_two_reports(reports_root)

    canonical = reports_root / "round2-qwen3-vl-8b-popular"
    partial_variant = canonical / "variant_results" / "full.csv"
    partial_variant.unlink()
    baseline_payload = json.loads((canonical / "baselines.json").read_text(encoding="utf-8"))
    baseline_payload.pop("full")
    (canonical / "baselines.json").write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    richer = reports_root / "round2-qwen3-vl-8b-popular-final"
    richer.mkdir(parents=True, exist_ok=True)
    (richer / "variant_results").mkdir(parents=True, exist_ok=True)
    for path in canonical.glob("variant_results/*.csv"):
        (richer / "variant_results" / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (richer / "variant_results" / "full.csv").write_text(
        (reports_root / "round2-internvl3.5-8b-popular" / "variant_results" / "full.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    richer_payload = dict(baseline_payload)
    richer_payload["full"] = {
        **_metric_payload(0.91, 0.19),
        "result_path": "variant_results/full.csv",
    }
    (richer / "baselines.json").write_text(
        json.dumps(richer_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (richer / "ablations.csv").write_text((canonical / "ablations.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (richer / "split_sensitivity.csv").write_text(
        (canonical / "split_sensitivity.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    outputs = paper_export.export_paper_package(
        reports_root=reports_root,
        output_root=output_root,
        tables_root=tables_root,
    )

    table1_popular = pd.read_csv(outputs["table1_pope_popular_csv"])
    qwen_popular_full = table1_popular.loc[
        (table1_popular["model"] == "Qwen3-VL-8B")
        & (table1_popular["method"] == "full MIND")
    ].iloc[0]
    assert qwen_popular_full["roc_auc"] == 0.91
    assert qwen_popular_full["pr_auc"] == 0.19
    assert qwen_popular_full["report_path"].endswith("round2-qwen3-vl-8b-popular-final")
