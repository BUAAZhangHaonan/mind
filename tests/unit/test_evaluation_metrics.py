from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from mind.evaluation import compute_binary_metrics, evaluate_by_subset, write_metrics_report


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate", SCRIPT_PATH)
evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(evaluate)


def test_compute_binary_metrics_returns_expected_keys() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["tpr_at_fpr_0.01"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_evaluate_by_subset_computes_metrics_per_subset() -> None:
    frame = pd.DataFrame(
        [
            {"subset": "popular", "label": 0, "prediction": 0, "score": 0.1},
            {"subset": "popular", "label": 1, "prediction": 1, "score": 0.9},
            {"subset": "adversarial", "label": 0, "prediction": 0, "score": 0.2},
            {"subset": "adversarial", "label": 1, "prediction": 1, "score": 0.8},
        ]
    )

    metrics = evaluate_by_subset(frame)

    assert sorted(metrics) == ["adversarial", "popular"]
    assert metrics["popular"]["accuracy"] == 1.0


def test_write_metrics_report_writes_json_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    payload = {"popular": {"accuracy": 1.0}}

    write_metrics_report(payload, output_path)

    restored = json.loads(output_path.read_text(encoding="utf-8"))
    assert restored == payload


def test_build_report_paths_returns_metrics_and_results_paths(tmp_path: Path) -> None:
    paths = evaluate.build_report_paths(output_root=tmp_path, experiment_name="smoke-qwen3-vl")

    assert paths["metrics"] == tmp_path / "smoke-qwen3-vl" / "metrics.json"
    assert paths["results"] == tmp_path / "smoke-qwen3-vl" / "results.csv"


def test_apply_label_overrides_relabels_matching_sample_ids() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "subset": "popular",
                "ground_truth_label": 1,
                "answer_label": 1,
                "label": 0,
                "prediction": 1,
                "score": 0.9,
            },
            {
                "sample_id": "sample-2",
                "subset": "popular",
                "ground_truth_label": 1,
                "answer_label": 1,
                "label": 0,
                "prediction": 1,
                "score": 0.8,
            },
        ]
    )
    overrides = pd.DataFrame([{"sample_id": "sample-1", "label": 0}])

    relabeled = evaluate.apply_label_overrides(frame, overrides)

    assert list(relabeled["ground_truth_label"]) == [0, 1]
    assert list(relabeled["label"]) == [1, 0]


def test_apply_label_overrides_accepts_jsonl_path(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "subset": "popular",
                "ground_truth_label": 1,
                "answer_label": 1,
                "label": 0,
                "prediction": 1,
                "score": 0.9,
            },
            {
                "sample_id": "sample-2",
                "subset": "popular",
                "ground_truth_label": 1,
                "answer_label": 1,
                "label": 0,
                "prediction": 1,
                "score": 0.8,
            },
        ]
    )
    overrides_path = tmp_path / "repope.jsonl"
    overrides_path.write_text('{"sample_id":"sample-1","label":0}\n', encoding="utf-8")

    relabeled = evaluate.apply_label_overrides(frame, overrides_path)

    assert list(relabeled["ground_truth_label"]) == [0, 1]
    assert list(relabeled["label"]) == [1, 0]


def test_run_evaluation_writes_metrics_and_results(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.parquet"
    frame = pd.DataFrame(
        [
            {"sample_id": "sample-1", "subset": "popular", "label": 0, "prediction": 0, "score": 0.1},
            {"sample_id": "sample-2", "subset": "popular", "label": 1, "prediction": 1, "score": 0.9},
            {"sample_id": "sample-3", "subset": "adversarial", "label": 0, "prediction": 0, "score": 0.2},
            {"sample_id": "sample-4", "subset": "adversarial", "label": 1, "prediction": 1, "score": 0.8},
        ]
    )
    frame.to_parquet(input_path, index=False)

    outputs = evaluate.run_evaluation(
        input_path=input_path,
        output_root=tmp_path / "reports",
        experiment_name="smoke-qwen3-vl",
    )

    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    restored = pd.read_csv(outputs["results"])

    assert outputs["metrics"].exists()
    assert outputs["results"].exists()
    assert metrics["overall"]["accuracy"] == 1.0
    assert sorted(metrics["by_subset"]) == ["adversarial", "popular"]
    assert list(restored["sample_id"]) == ["sample-1", "sample-2", "sample-3", "sample-4"]
