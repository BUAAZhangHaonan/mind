from __future__ import annotations

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


BATCH3_TARGET_MODELS = {
    "glm-4.6v-flash",
    "minicpm-v-2_6",
    "minicpm-v-4_5",
}
REGRESSION_MODELS = {
    "llava-onevision-qwen2-7b-ov-hf",
    "qwen3-vl-8b",
    "internvl3.5-8b",
    "gemma-3-12b-it",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "phi-3.5-vision-instruct",
    "gemma-3-4b-it",
    "qwen2.5-vl-7b",
}
NON_TARGET_BLOCKED_MODELS = {
    "phi-4-multimodal-instruct",
    "molmo-7b-d-0924",
    "llava-v1.5-7b",
}


def _summary(statuses: dict[str, str], reasons: dict[str, str]) -> dict[str, object]:
    return build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )


def test_batch3_target_models_cannot_disappear_from_final_summary() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REGRESSION_MODELS}
    reasons = {alias: "" for alias in REGRESSION_MODELS}

    summary = _summary(statuses, reasons)

    assert set(summary["model_statuses"]) == set(REQUIRED_MODEL_ALIASES)
    assert BATCH3_TARGET_MODELS <= set(summary["model_statuses"])
    for alias in BATCH3_TARGET_MODELS:
        assert summary["model_statuses"][alias] == AssetStatus.NOT_ATTEMPTED_DUE_TO_DEPENDENCY.value


def test_all_nine_regression_models_remain_verified_in_mocked_batch3_validation() -> None:
    statuses = {alias: AssetStatus.UNSUPPORTED_BY_WRAPPER.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "not selected in mocked Batch3 validation" for alias in REQUIRED_MODEL_ALIASES}
    for alias in REGRESSION_MODELS:
        statuses[alias] = AssetStatus.VERIFIED.value
        reasons[alias] = ""

    summary = _summary(statuses, reasons)

    assert len(REGRESSION_MODELS) == 9
    assert REGRESSION_MODELS <= set(summary["verified_models"])
    assert summary["final_status"] == "blocked"


def test_non_target_blocked_models_remain_represented() -> None:
    statuses = {}
    reasons = {}
    for alias in REQUIRED_MODEL_ALIASES:
        if alias in BATCH3_TARGET_MODELS | REGRESSION_MODELS:
            statuses[alias] = AssetStatus.VERIFIED.value
            reasons[alias] = ""
        else:
            statuses[alias] = AssetStatus.BLOCKED.value
            reasons[alias] = "blocked outside Batch3 target set"

    summary = _summary(statuses, reasons)

    assert NON_TARGET_BLOCKED_MODELS <= set(summary["blocked_models"])
    assert set(summary["model_statuses"]) == set(REQUIRED_MODEL_ALIASES)
    assert summary["final_status"] == "blocked"


def test_batch3_summary_distinguishes_all_final_status_groups() -> None:
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

    summary = _summary(statuses, reasons)

    assert "qwen2.5-vl-7b" in summary["verified_models"]
    assert "llava-v1.5-7b" in summary["blocked_models"]
    assert "minicpm-v-2_6" in summary["unsupported_by_wrapper_models"]
    assert "glm-4.6v-flash" in summary["unsupported_by_policy_models"]
    assert "molmo-7b-d-0924" in summary["failed_models"]
    assert summary["final_status"] == "blocked"


def test_final_status_remains_blocked_if_any_registry_model_is_non_verified() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    reasons = {alias: "" for alias in REQUIRED_MODEL_ALIASES}
    statuses["llava-v1.5-7b"] = AssetStatus.BLOCKED.value
    reasons["llava-v1.5-7b"] = "blocked"

    summary = _summary(statuses, reasons)

    assert summary["num_verified"] == len(REQUIRED_MODEL_ALIASES) - 1
    assert summary["final_status"] == "blocked"
