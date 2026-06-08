from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


REGRESSION_MODELS = {
    "glm-4.6v-flash",
    "minicpm-v-2_6",
    "minicpm-v-4_5",
    "qwen2.5-vl-7b",
    "qwen3-vl-8b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "internvl3.5-8b",
    "llava-onevision-qwen2-7b-ov-hf",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "phi-3.5-vision-instruct",
}


def test_all_registry_models_appear_exactly_once_in_final_summary() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "" for alias in REQUIRED_MODEL_ALIASES}
    statuses["gemma-4-12b-it"] = AssetStatus.BLOCKED.value
    reasons["gemma-4-12b-it"] = "local path does not exist"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=("pytest tests/assets",),
        git_commit="abc123",
    )

    buckets = [
        summary["verified_models"],
        summary["blocked_models"],
        summary["unsupported_by_wrapper_models"],
        summary["unsupported_by_policy_models"],
        summary["failed_models"],
        summary["not_attempted_due_to_dependency_models"],
    ]
    flattened = [alias for bucket in buckets for alias in bucket]
    assert len(REQUIRED_MODEL_ALIASES) == 16
    assert "gemma-4-12b-it" in REQUIRED_MODEL_ALIASES
    assert sorted(flattened) == sorted(REQUIRED_MODEL_ALIASES)
    assert len(flattened) == len(set(flattened))
    assert REGRESSION_MODELS <= set(summary["verified_models"])


def test_final_summary_distinguishes_status_classes() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "" for alias in REQUIRED_MODEL_ALIASES}
    statuses["gemma-4-12b-it"] = AssetStatus.BLOCKED.value
    reasons["gemma-4-12b-it"] = "blocked"
    statuses["phi-4-multimodal-instruct"] = AssetStatus.UNSUPPORTED_BY_POLICY.value
    reasons["phi-4-multimodal-instruct"] = "policy"
    statuses["molmo-7b-d-0924"] = AssetStatus.UNSUPPORTED_BY_WRAPPER.value
    reasons["molmo-7b-d-0924"] = "wrapper"
    statuses["llava-v1.5-7b"] = AssetStatus.FAILED_VALIDATION.value
    reasons["llava-v1.5-7b"] = "failed"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert "gemma-4-12b-it" in summary["blocked_models"]
    assert "phi-4-multimodal-instruct" in summary["unsupported_by_policy_models"]
    assert "molmo-7b-d-0924" in summary["unsupported_by_wrapper_models"]
    assert "llava-v1.5-7b" in summary["failed_models"]


def test_final_status_passes_only_when_all_registry_models_verified() -> None:
    verified = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    passed = build_completion_summary(
        model_statuses=verified,
        model_reasons={},
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )
    blocked_statuses = dict(verified)
    blocked_statuses["gemma-4-12b-it"] = AssetStatus.BLOCKED.value
    blocked = build_completion_summary(
        model_statuses=blocked_statuses,
        model_reasons={"gemma-4-12b-it": "local path missing"},
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert passed["final_status"] == "passed"
    assert blocked["final_status"] == "blocked"


def test_validation_writes_final_blocked_and_unsupported_json(tmp_path: Path) -> None:
    module_path = Path("scripts/asset_validate_hidden_states.py")
    spec = importlib.util.spec_from_file_location("asset_validate_hidden_states", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "" for alias in REQUIRED_MODEL_ALIASES}
    statuses["molmo-7b-d-0924"] = AssetStatus.BLOCKED.value
    reasons["molmo-7b-d-0924"] = "smoke extraction failed"
    statuses["phi-4-multimodal-instruct"] = AssetStatus.UNSUPPORTED_BY_WRAPPER.value
    reasons["phi-4-multimodal-instruct"] = "wrapper missing"
    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    module.write_final_status_model_lists(tmp_path, summary)

    blocked = json.loads((tmp_path / "blocked_models.json").read_text(encoding="utf-8"))
    unsupported = json.loads((tmp_path / "unsupported_models.json").read_text(encoding="utf-8"))
    assert blocked == [{"alias": "molmo-7b-d-0924", "status": "blocked", "reason": "smoke extraction failed"}]
    assert unsupported == [
        {
            "alias": "phi-4-multimodal-instruct",
            "status": "unsupported_by_wrapper",
            "reason": "wrapper missing",
        }
    ]
