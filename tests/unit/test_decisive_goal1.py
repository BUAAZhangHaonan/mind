from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_decisive_goal1.py"
    assert script_path.exists(), "scripts/prepare_decisive_goal1.py should exist"
    spec = importlib.util.spec_from_file_location("prepare_decisive_goal1", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(root: Path, name: str) -> Path:
    report = root / name
    variant_results = report / "variant_results"
    variant_results.mkdir(parents=True)
    (report / "notes.txt").write_text("copied first\n", encoding="utf-8")
    _write_json(
        report / "baselines.json",
        {
            "bank_scope": "object",
            "full_variant": "raw_plus_calibrated_simple",
            "full": {
                "pr_auc": 0.10,
                "roc_auc": 0.50,
                "result_path": "old/full.csv",
            },
            "linear_probe": {
                "pr_auc": 0.40,
                "roc_auc": 0.80,
                "result_path": "old/linear_probe.csv",
            },
            "no_manifold": {
                "pr_auc": 0.18,
                "roc_auc": 0.65,
                "result_path": "old/no_manifold.csv",
            },
            "raw_plus_calibrated_full_curve": {
                "pr_auc": 0.25,
                "roc_auc": 0.70,
                "result_path": "old/raw_plus_calibrated_full_curve.csv",
            },
        },
    )
    pd.DataFrame(
        [
            {"sample_id": "a", "label": 1, "score": 0.2, "prediction": 0},
            {"sample_id": "b", "label": 0, "score": 0.8, "prediction": 1},
        ]
    ).to_csv(variant_results / "raw_plus_calibrated_full_curve.csv", index=False)
    pd.DataFrame(
        [
            {"sample_id": "a", "label": 1, "score": 0.1, "prediction": 0},
            {"sample_id": "b", "label": 0, "score": 0.4, "prediction": 0},
        ]
    ).to_csv(variant_results / "full.csv", index=False)
    pd.DataFrame(
        [
            {"variant": "full", "pr_auc": 0.10, "roc_auc": 0.50},
            {"variant": "linear_probe", "pr_auc": 0.40, "roc_auc": 0.80},
            {"variant": "no_manifold", "pr_auc": 0.18, "roc_auc": 0.65},
            {"variant": "raw_plus_calibrated_full_curve", "pr_auc": 0.25, "roc_auc": 0.70},
        ]
    ).to_csv(report / "ablations.csv", index=False)
    pd.DataFrame(
        [
            {"variant": "full", "random_state": 13, "pr_auc": 0.11, "roc_auc": 0.51},
            {"variant": "full", "random_state": 17, "pr_auc": 0.12, "roc_auc": 0.52},
            {
                "variant": "raw_plus_calibrated_full_curve",
                "random_state": 13,
                "pr_auc": 0.26,
                "roc_auc": 0.71,
            },
            {
                "variant": "raw_plus_calibrated_full_curve",
                "random_state": 17,
                "pr_auc": 0.27,
                "roc_auc": 0.72,
            },
        ]
    ).to_csv(report / "split_sensitivity.csv", index=False)
    return report


def test_aliases_full_curve_payload_and_writes_comparison(tmp_path: Path) -> None:
    script = _load_script()
    source_root = tmp_path / "round2" / "reports"
    dest_root = tmp_path / "decisive" / "reports"
    comparison_path = tmp_path / "docs" / "tables" / "comparison.md"
    _write_report(source_root, "round2-llava-onevision-7b-popular")

    script.prepare_goal1_reports(
        source_root=source_root,
        dest_root=dest_root,
        comparison_path=comparison_path,
        run_feature_baselines=False,
    )

    copied_report = dest_root / "round2-llava-onevision-7b-popular"
    assert (copied_report / "notes.txt").read_text(encoding="utf-8") == "copied first\n"
    baselines = json.loads((copied_report / "baselines.json").read_text(encoding="utf-8"))
    assert baselines["full_variant"] == "raw_plus_calibrated_full_curve"
    assert baselines["full"]["pr_auc"] == 0.25
    assert baselines["full"]["roc_auc"] == 0.70
    assert baselines["linear_probe"]["pr_auc"] == 0.40
    assert baselines["no_manifold"]["pr_auc"] == 0.18

    raw_results = pd.read_csv(copied_report / "variant_results" / "raw_plus_calibrated_full_curve.csv")
    full_results = pd.read_csv(copied_report / "variant_results" / "full.csv")
    pd.testing.assert_frame_equal(full_results, raw_results)

    ablations = pd.read_csv(copied_report / "ablations.csv")
    full_ablation = ablations.loc[ablations["variant"] == "full"].iloc[0]
    assert full_ablation["pr_auc"] == 0.25
    assert full_ablation["roc_auc"] == 0.70

    split_sensitivity = pd.read_csv(copied_report / "split_sensitivity.csv")
    full_split = split_sensitivity.loc[split_sensitivity["variant"] == "full"].sort_values("random_state")
    assert full_split["pr_auc"].tolist() == [0.26, 0.27]
    assert full_split["roc_auc"].tolist() == [0.71, 0.72]

    markdown = comparison_path.read_text(encoding="utf-8")
    assert markdown.index("old_full_pr_auc") < markdown.index("old_full_roc_auc")
    assert "full_pr_auc_delta" in markdown
    assert "linear_probe_pr_gap_shrink" in markdown
    assert "no_manifold_pr_gap_shrink" in markdown
    assert "+0.1500" in markdown


def test_feature_only_recompute_uses_refresh_features_and_excludes_haar(tmp_path: Path) -> None:
    script = _load_script()
    source_root = tmp_path / "round2" / "reports"
    dest_root = tmp_path / "decisive" / "reports"
    comparison_path = tmp_path / "docs" / "tables" / "comparison.md"
    _write_report(source_root, "round2-qwen3-vl-8b-popular-object-heldout")
    features_path = (
        tmp_path
        / "round2"
        / "features"
        / "round2-qwen3-vl-8b-popular-object-heldout-refresh"
        / "popular.parquet"
    )
    features_path.parent.mkdir(parents=True)
    features_path.write_text("placeholder", encoding="utf-8")
    commands: list[list[str]] = []

    script.prepare_goal1_reports(
        source_root=source_root,
        dest_root=dest_root,
        comparison_path=comparison_path,
        run_feature_baselines=True,
        runner=commands.append,
    )

    assert len(commands) == 1
    command_text = " ".join(commands[0])
    assert str(features_path) in commands[0]
    assert "--split-strategy object_heldout" in command_text
    assert "--num-folds 2" in command_text
    assert "--full-variant raw_plus_calibrated_full_curve" in command_text
    assert "--variants full,raw_curve_only,raw_plus_calibrated_simple,raw_plus_calibrated_full_curve" in command_text
    assert "haar" not in command_text


def test_feature_only_recompute_uses_popular_features_and_repope_overrides(tmp_path: Path) -> None:
    script = _load_script()
    source_root = tmp_path / "round2" / "reports"
    dest_root = tmp_path / "decisive" / "reports"
    comparison_path = tmp_path / "docs" / "tables" / "comparison.md"
    _write_report(source_root, "round2-qwen3-vl-8b-repope")
    features_path = tmp_path / "round2" / "features" / "round2-qwen3-vl-8b-popular" / "popular.parquet"
    features_path.parent.mkdir(parents=True)
    features_path.write_text("placeholder", encoding="utf-8")
    label_overrides = tmp_path / "round2" / "normalized" / "repope" / "popular.jsonl"
    label_overrides.parent.mkdir(parents=True)
    label_overrides.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    script.prepare_goal1_reports(
        source_root=source_root,
        dest_root=dest_root,
        comparison_path=comparison_path,
        run_feature_baselines=True,
        runner=commands.append,
    )

    assert len(commands) == 1
    command_text = " ".join(commands[0])
    assert str(features_path) in commands[0]
    assert "--split-strategy image_grouped" in command_text
    assert "--num-folds 5" in command_text
    assert "--label-overrides" in commands[0]
    assert str(label_overrides) in commands[0]
