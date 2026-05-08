from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_script() -> ModuleType:
    script_path = Path("scripts/stage_a_run.py")
    spec = importlib.util.spec_from_file_location("stage_a_run", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_row_reports_one_class_scope_without_crashing() -> None:
    module = _load_script()
    entries = [
        {"stage_a_split": "test", "subset": "random"},
        {"stage_a_split": "test", "subset": "random"},
    ]
    labels = np.array([0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2], dtype=np.float32)
    mask = np.array([True, True])

    row = module._metric_row(
        entries,
        labels,
        scores,
        mask,
        variant="Raw-Static",
        readout="Diag-KNN",
        eval_split="test",
        eval_scope="random",
        bootstrap=10,
        seed=20260506,
        num_bank_correct=12,
        excluded_counts={},
        extra={},
    )

    assert row["metric_status"] == "undefined"
    assert "one class" in row["failure_reason"]
    assert np.isnan(row["pr_auc"])
    assert row["num_test_correct"] == 2
    assert row["num_test_hard_hallucination"] == 0


def test_preflight_failure_summary_is_failed(tmp_path: Path) -> None:
    module = _load_script()

    output = module._write_stage_a_summary(
        tmp_path,
        {"status": "failed"},
        {},
        [],
        "fail",
        dry_run=False,
    )

    payload = json.loads(Path(output).read_text())
    assert payload["status"] == "failed"
    assert payload["overall_decision"] == "fail"
    assert payload["stage_b_started"] is False
