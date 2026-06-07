from __future__ import annotations

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


BATCH2_TARGET_MODELS = {
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "phi-3.5-vision-instruct",
    "phi-4-multimodal-instruct",
}
REGRESSION_MODELS = {
    "qwen2.5-vl-7b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "internvl3.5-8b",
    "qwen3-vl-8b",
    "llava-onevision-qwen2-7b-ov-hf",
}
NON_TARGET_UNRESOLVED_MODELS = {
    "glm-4.6v-flash",
    "minicpm-v-2_6",
    "minicpm-v-4_5",
    "molmo-7b-d-0924",
    "llava-v1.5-7b",
}


def test_batch2_summary_keeps_targets_regressions_and_non_targets() -> None:
    statuses = {}
    reasons = {}
    for alias in REQUIRED_MODEL_ALIASES:
        if alias in BATCH2_TARGET_MODELS | REGRESSION_MODELS:
            statuses[alias] = AssetStatus.VERIFIED.value
            reasons[alias] = ""
        elif alias == "molmo-7b-d-0924":
            statuses[alias] = AssetStatus.FAILED_VALIDATION.value
            reasons[alias] = "smoke validation failed"
        elif alias == "llava-v1.5-7b":
            statuses[alias] = AssetStatus.BLOCKED.value
            reasons[alias] = "processor/tokenizer metadata is missing"
        else:
            statuses[alias] = AssetStatus.UNSUPPORTED_BY_WRAPPER.value
            reasons[alias] = "wrapper not implemented in this batch"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert set(summary["model_statuses"]) == set(REQUIRED_MODEL_ALIASES)
    assert BATCH2_TARGET_MODELS <= set(summary["verified_models"])
    assert REGRESSION_MODELS <= set(summary["verified_models"])
    assert NON_TARGET_UNRESOLVED_MODELS <= (
        set(summary["blocked_models"])
        | set(summary["unsupported_by_wrapper_models"])
        | set(summary["unsupported_by_policy_models"])
        | set(summary["failed_models"])
    )
    assert summary["final_status"] == "blocked"


def test_batch2_summary_distinguishes_all_non_verified_statuses() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "" for alias in REQUIRED_MODEL_ALIASES}
    statuses["glm-4.6v-flash"] = AssetStatus.UNSUPPORTED_BY_POLICY.value
    reasons["glm-4.6v-flash"] = "policy"
    statuses["minicpm-v-2_6"] = AssetStatus.UNSUPPORTED_BY_WRAPPER.value
    reasons["minicpm-v-2_6"] = "wrapper"
    statuses["llava-v1.5-7b"] = AssetStatus.BLOCKED.value
    reasons["llava-v1.5-7b"] = "blocked"
    statuses["molmo-7b-d-0924"] = AssetStatus.FAILED_VALIDATION.value
    reasons["molmo-7b-d-0924"] = "failed validation"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert "glm-4.6v-flash" in summary["unsupported_by_policy_models"]
    assert "minicpm-v-2_6" in summary["unsupported_by_wrapper_models"]
    assert "llava-v1.5-7b" in summary["blocked_models"]
    assert "molmo-7b-d-0924" in summary["failed_models"]
    assert summary["final_status"] == "blocked"


def test_batch1_regression_models_remain_verified_in_mocked_batch2_summary() -> None:
    statuses = {alias: AssetStatus.UNSUPPORTED_BY_WRAPPER.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "not in mocked verified set" for alias in REQUIRED_MODEL_ALIASES}
    for alias in REGRESSION_MODELS:
        statuses[alias] = AssetStatus.VERIFIED.value
        reasons[alias] = ""

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert REGRESSION_MODELS <= set(summary["verified_models"])
    assert summary["final_status"] == "blocked"
