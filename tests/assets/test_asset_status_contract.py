from __future__ import annotations

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


def test_every_requested_model_has_exactly_one_final_status() -> None:
    statuses = {}
    reasons = {}
    for index, alias in enumerate(REQUIRED_MODEL_ALIASES):
        if index == 0:
            statuses[alias] = AssetStatus.FAILED_VALIDATION.value
            reasons[alias] = "failed"
        elif index == 1:
            statuses[alias] = AssetStatus.UNSUPPORTED_BY_POLICY.value
            reasons[alias] = "policy"
        elif index == 2:
            statuses[alias] = AssetStatus.UNSUPPORTED_BY_WRAPPER.value
            reasons[alias] = "wrapper"
        elif index == 3:
            statuses[alias] = AssetStatus.NOT_ATTEMPTED_DUE_TO_DEPENDENCY.value
            reasons[alias] = "dependency"
        else:
            statuses[alias] = AssetStatus.BLOCKED.value
            reasons[alias] = "blocked"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    model_lists = [
        summary["verified_models"],
        summary["blocked_models"],
        summary["unsupported_by_policy_models"],
        summary["unsupported_by_wrapper_models"],
        summary["failed_models"],
        summary["not_attempted_due_to_dependency_models"],
    ]
    flattened = [alias for group in model_lists for alias in group]
    assert sorted(flattened) == sorted(REQUIRED_MODEL_ALIASES)
    assert len(flattened) == len(set(flattened))
    assert summary["num_failed_validation"] == 1
    assert summary["num_unsupported_by_policy"] == 1
    assert summary["num_unsupported_by_wrapper"] == 1
    assert summary["num_not_attempted_due_to_dependency"] == 1
    assert summary["training_started"] is False
