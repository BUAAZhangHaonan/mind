from __future__ import annotations

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


def test_all_16_models_appear_exactly_once_with_verified_separate_env() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    statuses["molmo-7b-d-0924"] = "verified_separate_env"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons={"molmo-7b-d-0924": "accepted from separate env"},
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    buckets = [
        summary["verified_models"],
        summary["verified_separate_env_models"],
        summary["blocked_models"],
        summary["unsupported_by_policy_models"],
        summary["unsupported_by_wrapper_models"],
        summary["failed_models"],
        summary["not_attempted_due_to_dependency_models"],
    ]
    flattened = [alias for bucket in buckets for alias in bucket]
    assert sorted(flattened) == sorted(REQUIRED_MODEL_ALIASES)
    assert len(flattened) == len(set(flattened))
    assert summary["final_status"] == "passed"


def test_final_status_blocks_when_any_model_is_blocked_or_failed() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    statuses["molmo-7b-d-0924"] = "verified_separate_env"
    statuses["phi-4-multimodal-instruct"] = AssetStatus.FAILED_VALIDATION.value

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons={"phi-4-multimodal-instruct": "validation failed"},
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert summary["final_status"] == "blocked"
    assert "phi-4-multimodal-instruct" in summary["failed_models"]


def test_no_final_target_remains_unsupported_by_wrapper() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    statuses["molmo-7b-d-0924"] = "verified_separate_env"
    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons={},
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    final_targets = {"gemma-4-12b-it", "phi-4-multimodal-instruct", "llava-v1.5-7b", "molmo-7b-d-0924"}
    assert final_targets.isdisjoint(set(summary["unsupported_by_wrapper_models"]))
