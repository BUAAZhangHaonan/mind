from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script() -> ModuleType:
    script_path = Path("scripts/stage_a_run.py")
    spec = importlib.util.spec_from_file_location("stage_a_run", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_lightweight_stage_a_run(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "run_preflight", lambda **_: {"status": "passed"})
    monkeypatch.setattr(module, "build_family_splits", lambda **_: {"assignments": []})
    monkeypatch.setattr(module, "_write_population_audits", lambda **_: None)
    monkeypatch.setattr(module, "run_model_stage_a", lambda **_: {"overall_decision": "pass"})


def _base_stage_a_args(tmp_path: Path) -> list[str]:
    return [
        "--stage0-root",
        str(tmp_path / "stage0"),
        "--output-root",
        str(tmp_path / "stageA"),
        "--models",
        "qwen3-vl-8b",
        "--device",
        "cpu",
        "--bootstrap",
        "2",
        "--lstm-epochs",
        "1",
        "--knn-k",
        "1",
    ]


def _with_models(args: list[str], models: list[str]) -> list[str]:
    start = args.index("--models")
    end = args.index("--device")
    return [*args[: start + 1], *models, *args[end:]]


def _stage_a_summary(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "stageA" / "manifests" / "stageA_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _lstm_metric_row(
    *,
    readout: str,
    ordered: bool,
    pr_auc: float,
    roc_auc: float,
    pr_auc_ci_low: float,
    pr_auc_ci_high: float,
) -> dict[str, object]:
    return {
        "variant": "Sphere-Traj-LSTM-v0" if ordered else "Sphere-Traj-Shuffled-LSTM",
        "readout": readout,
        "eval_split": "test",
        "eval_scope": "pooled",
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "pr_auc_ci_low": pr_auc_ci_low,
        "pr_auc_ci_high": pr_auc_ci_high,
    }


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


def test_layer_order_gate_requires_both_lstm_readouts_to_support_order() -> None:
    module = _load_script()
    rows = [
        _lstm_metric_row(
            readout="Diag-Classifier",
            ordered=True,
            pr_auc=0.70,
            roc_auc=0.76,
            pr_auc_ci_low=0.68,
            pr_auc_ci_high=0.72,
        ),
        _lstm_metric_row(
            readout="Diag-Classifier",
            ordered=False,
            pr_auc=0.64,
            roc_auc=0.73,
            pr_auc_ci_low=0.61,
            pr_auc_ci_high=0.66,
        ),
        _lstm_metric_row(
            readout="Diag-KNN",
            ordered=True,
            pr_auc=0.51,
            roc_auc=0.51,
            pr_auc_ci_low=0.45,
            pr_auc_ci_high=0.56,
        ),
        _lstm_metric_row(
            readout="Diag-KNN",
            ordered=False,
            pr_auc=0.50,
            roc_auc=0.52,
            pr_auc_ci_low=0.46,
            pr_auc_ci_high=0.55,
        ),
    ]

    gate = module._layer_order_gate(rows)

    assert gate["status"] == "mixed"
    assert len(gate["comparisons"]) == 2


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


def test_dry_run_summary_overwrites_stale_completed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _patch_lightweight_stage_a_run(module, monkeypatch)
    summary_path = tmp_path / "stageA" / "manifests" / "stageA_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"status": "completed", "completed_models": ["qwen3-vl-8b"]}),
        encoding="utf-8",
    )

    result = module.main([*_base_stage_a_args(tmp_path), "--dry-run"])

    payload = _stage_a_summary(tmp_path)
    assert result == 0
    assert payload["status"] == "dry_run"
    assert payload["dry_run"] is True
    assert payload["full_stage_a_run"] is False
    assert "dry_run" in payload["run_scope"]["reasons"]


def test_runtime_model_failure_overwrites_stale_completed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "run_preflight", lambda **_: {"status": "passed"})
    monkeypatch.setattr(module, "build_family_splits", lambda **_: {"assignments": []})
    monkeypatch.setattr(module, "_write_population_audits", lambda **_: None)

    def fail_model(**_: object) -> dict[str, object]:
        raise RuntimeError("model execution exploded")

    monkeypatch.setattr(module, "run_model_stage_a", fail_model)
    summary_path = tmp_path / "stageA" / "manifests" / "stageA_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"status": "completed", "completed_models": ["qwen3-vl-8b"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="model execution exploded"):
        module.main(_base_stage_a_args(tmp_path))

    payload = _stage_a_summary(tmp_path)
    assert payload["status"] == "failed"
    assert payload["overall_decision"] == "fail"
    assert payload["completed_models"] == []
    assert payload["failure_reason"] == (
        "runtime failure while running qwen3-vl-8b: RuntimeError: model execution exploded"
    )
    assert "requested_models_not_completed_or_qwen_gate_skipped" in payload["run_scope"]["reasons"]


def test_split_build_failure_overwrites_stale_completed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "run_preflight", lambda **_: {"status": "passed"})

    def fail_splits(**_: object) -> dict[str, object]:
        raise RuntimeError("split building exploded")

    monkeypatch.setattr(module, "build_family_splits", fail_splits)
    monkeypatch.setattr(module, "_write_population_audits", lambda **_: None)
    summary_path = tmp_path / "stageA" / "manifests" / "stageA_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"status": "completed", "completed_models": ["qwen3-vl-8b"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="split building exploded"):
        module.main(_base_stage_a_args(tmp_path))

    payload = _stage_a_summary(tmp_path)
    assert payload["status"] == "failed"
    assert payload["overall_decision"] == "fail"
    assert payload["stage0_acceptance"] == "passed"
    assert payload["stage_b_started"] is False
    assert payload["completed_models"] == []
    assert payload["failure_reason"] == (
        "runtime failure while preparing Stage A: RuntimeError: split building exploded"
    )
    assert payload["requested_models"] == ["qwen3-vl-8b"]
    assert payload["requested_subsets"] == ["popular", "random", "adversarial"]
    assert payload["run_scope"]["requested_models"] == ["qwen3-vl-8b"]
    assert payload["run_scope"]["requested_subsets"] == ["popular", "random", "adversarial"]
    assert "requested_models_not_completed_or_qwen_gate_skipped" in payload["run_scope"]["reasons"]


def test_failed_summary_overwrites_stale_completed_status(tmp_path: Path) -> None:
    module = _load_script()
    summary_path = tmp_path / "manifests" / "stageA_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"status": "completed", "completed_models": ["qwen3-vl-8b"]}),
        encoding="utf-8",
    )

    output = module._write_stage_a_summary(
        tmp_path,
        {"status": "failed"},
        {},
        [],
        "fail",
        dry_run=False,
    )

    payload = json.loads(Path(output).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["full_stage_a_run"] is False
    assert "stage0_acceptance_not_passed" in payload["run_scope"]["reasons"]


@pytest.mark.parametrize(
    ("extra_args", "reason"),
    [
        (["--skip-lstm"], "skip_lstm"),
        (["--limit-per-subset", "2"], "limit_per_subset"),
        (["--subsets", "popular", "random"], "required_subsets"),
    ],
)
def test_partial_stage_a_runs_are_marked_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    reason: str,
) -> None:
    module = _load_script()
    _patch_lightweight_stage_a_run(module, monkeypatch)

    result = module.main([*_base_stage_a_args(tmp_path), *extra_args])

    payload = _stage_a_summary(tmp_path)
    assert result == 0
    assert payload["status"] == "partial"
    assert payload["full_stage_a_run"] is False
    assert reason in payload["run_scope"]["reasons"]


def test_requested_internvl_without_qwen_gate_skip_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _patch_lightweight_stage_a_run(module, monkeypatch)
    args = _with_models(_base_stage_a_args(tmp_path), ["qwen3-vl-8b", "internvl3.5-8b"])

    result = module.main(args)

    payload = _stage_a_summary(tmp_path)
    assert result == 0
    assert payload["status"] == "partial"
    assert payload["full_stage_a_run"] is False
    assert "requested_models_not_completed_or_qwen_gate_skipped" in payload["run_scope"]["reasons"]


def test_qwen_only_run_writes_internvl_not_requested_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _patch_lightweight_stage_a_run(module, monkeypatch)

    result = module.main(_base_stage_a_args(tmp_path))

    not_run_path = tmp_path / "stageA" / "reports" / "internvl3.5-8b" / "not_run.json"
    not_run = json.loads(not_run_path.read_text(encoding="utf-8"))
    payload = _stage_a_summary(tmp_path)
    assert result == 0
    assert payload["status"] == "completed"
    assert payload["full_stage_a_run"] is True
    assert not_run["model_name"] == "internvl3.5-8b"
    assert not_run["status"] == "not_run"
    assert not_run["skip_type"] == "not_requested"
    assert "not requested" in not_run["reason"]


def test_qwen_gate_skip_counts_as_full_requested_model_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "run_preflight", lambda **_: {"status": "passed"})
    monkeypatch.setattr(module, "build_family_splits", lambda **_: {"assignments": []})
    monkeypatch.setattr(module, "_write_population_audits", lambda **_: None)
    monkeypatch.setattr(module, "run_model_stage_a", lambda **_: {"overall_decision": "fail"})
    args = _with_models(_base_stage_a_args(tmp_path), ["qwen3-vl-8b", "internvl3.5-8b"])

    result = module.main(args)

    payload = _stage_a_summary(tmp_path)
    assert result == 0
    assert payload["status"] == "completed"
    assert payload["full_stage_a_run"] is True
    assert payload["run_scope"]["skipped_models"]["internvl3.5-8b"]["skip_type"] == "qwen_gate"


def test_parser_rejects_non_primary_stage_a_models() -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--models", "qwen3-vl-8b", "llava-7b"])


def test_parser_allows_internvl_only_invocation() -> None:
    module = _load_script()

    args = module.build_parser().parse_args(["--models", "internvl3.5-8b"])

    assert args.models == ["internvl3.5-8b"]
