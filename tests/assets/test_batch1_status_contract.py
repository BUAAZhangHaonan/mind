from __future__ import annotations

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


TARGET_MODELS = {
    "qwen2.5-vl-7b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "internvl3.5-8b",
}
REGRESSION_MODELS = {
    "qwen3-vl-8b",
    "llava-onevision-qwen2-7b-ov-hf",
}


def test_batch1_summary_keeps_targets_regressions_and_non_targets() -> None:
    statuses = {}
    reasons = {}
    for alias in REQUIRED_MODEL_ALIASES:
        if alias in TARGET_MODELS | REGRESSION_MODELS:
            statuses[alias] = AssetStatus.VERIFIED.value
            reasons[alias] = ""
        elif alias == "glm-4.6v-flash":
            statuses[alias] = AssetStatus.UNSUPPORTED_BY_POLICY.value
            reasons[alias] = "policy"
        else:
            statuses[alias] = AssetStatus.UNSUPPORTED_BY_WRAPPER.value
            reasons[alias] = "wrapper"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert set(summary["model_statuses"]) == set(REQUIRED_MODEL_ALIASES)
    assert TARGET_MODELS <= set(summary["verified_models"])
    assert REGRESSION_MODELS <= set(summary["verified_models"])
    assert "glm-4.6v-flash" in summary["unsupported_by_policy_models"]
    assert "gemma-3-12b-it" in summary["unsupported_by_wrapper_models"]
    assert summary["final_status"] == "blocked"


def test_failed_validation_is_distinct_from_blocked_for_target_model() -> None:
    statuses = {alias: AssetStatus.UNSUPPORTED_BY_WRAPPER.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "wrapper" for alias in REQUIRED_MODEL_ALIASES}
    statuses["qwen2.5-vl-7b"] = AssetStatus.FAILED_VALIDATION.value
    reasons["qwen2.5-vl-7b"] = "validation failed"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert "qwen2.5-vl-7b" in summary["failed_models"]
    assert "qwen2.5-vl-7b" not in summary["blocked_models"]
