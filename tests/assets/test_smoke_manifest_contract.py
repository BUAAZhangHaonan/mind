from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mind.models.asset_validation import (
    AssetStatus,
    build_completion_summary,
    validate_smoke_report_contract,
)
from mind.models.registry import REQUIRED_MODEL_ALIASES


def test_smoke_report_includes_every_model_dataset_pair() -> None:
    rows = [
        {"model_alias": alias, "dataset": dataset, "status": AssetStatus.BLOCKED.value, "reason": "blocked"}
        for alias in REQUIRED_MODEL_ALIASES
        for dataset in ("pope", "repope", "dash-b")
    ]

    result = validate_smoke_report_contract(rows, datasets=("pope", "repope", "dash-b"))

    assert result.status == "verified"


def test_final_status_cannot_pass_when_any_model_is_not_verified() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    statuses[REQUIRED_MODEL_ALIASES[0]] = AssetStatus.UNSUPPORTED_BY_POLICY.value

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons={REQUIRED_MODEL_ALIASES[0]: "policy reason"},
        smoke_datasets=["pope", "repope", "dash-b"],
        smoke_limit=2,
        tests_run=["pytest tests/assets"],
        git_commit="abc123",
    )

    assert summary["final_status"] == "blocked"
    assert REQUIRED_MODEL_ALIASES[0] in summary["unsupported_models"]
    assert summary["unsupported_reasons"][REQUIRED_MODEL_ALIASES[0]] == "policy reason"


def test_partial_smoke_failure_summary_uses_worst_status(tmp_path: Path) -> None:
    module_path = Path("scripts/asset_smoke_extract.py")
    spec = importlib.util.spec_from_file_location("asset_smoke_extract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for alias in REQUIRED_MODEL_ALIASES:
        rows.extend(
            [
                {"model_alias": alias, "dataset": "pope", "subset": "popular", "status": AssetStatus.VERIFIED.value, "reason": ""},
                {"model_alias": alias, "dataset": "repope", "subset": "popular", "status": AssetStatus.FAILED_VALIDATION.value, "reason": "later failure"},
            ]
        )

    module.write_summary_from_rows(tmp_path, rows, datasets=("pope", "repope"), smoke_limit=2)
    summary = json.loads((tmp_path / "asset_completion_summary.json").read_text(encoding="utf-8"))

    assert summary["final_status"] == "blocked"
    assert summary["num_failed_validation"] == len(REQUIRED_MODEL_ALIASES)


def test_validation_contract_failure_marks_every_model_failed() -> None:
    module_path = Path("scripts/asset_validate_hidden_states.py")
    spec = importlib.util.spec_from_file_location("asset_validate_hidden_states", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = module._contract_failure_rows("smoke report contract failed: duplicate")
    statuses, reasons = module.aggregate_model_statuses(rows)

    assert set(statuses) == set(REQUIRED_MODEL_ALIASES)
    assert all(status == AssetStatus.FAILED_VALIDATION.value for status in statuses.values())
    assert all("duplicate" in reason for reason in reasons.values())


def test_validation_checks_determinism_and_canary_statuses() -> None:
    module_path = Path("scripts/asset_validate_hidden_states.py")
    spec = importlib.util.spec_from_file_location("asset_validate_hidden_states", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = "qwen3-vl-8b/pope/popular"
    valid = {
        "determinism": {key: {"status": "verified"}},
        "image_sensitivity_canary": {"qwen3-vl-8b": {"status": "verified"}},
    }

    assert module.validate_checksum_statuses(valid, key=key, model_alias="qwen3-vl-8b").status == "verified"
    bad_determinism = {
        "determinism": {key: {"status": "failed_validation"}},
        "image_sensitivity_canary": {"qwen3-vl-8b": {"status": "verified"}},
    }
    assert module.validate_checksum_statuses(bad_determinism, key=key, model_alias="qwen3-vl-8b").status == "failed_validation"
    bad_canary = {
        "determinism": {key: {"status": "verified"}},
        "image_sensitivity_canary": {"qwen3-vl-8b": {"status": "failed_validation"}},
    }
    assert module.validate_checksum_statuses(bad_canary, key=key, model_alias="qwen3-vl-8b").status == "failed_validation"
