from __future__ import annotations

from .conftest import full_cache_attr, synthetic_unified_manifest


def test_report_summary_contains_route_table_and_per_model_counts() -> None:
    render_full_cache_report = full_cache_attr("render_full_cache_report")

    report = render_full_cache_report(synthetic_unified_manifest())

    normalized = report.lower()
    assert "| model_alias | route | status | total_entries |" in normalized
    assert "qwen3-vl-8b" in report
    assert "accepted_existing_stage0" in report
    assert "molmo-7b-d-0924" in report
    assert "failed_validation" in report
    assert "total_models: 16" in normalized
    assert "total_entries:" in normalized


def test_report_summary_contains_no_scientific_claims() -> None:
    render_full_cache_report = full_cache_attr("render_full_cache_report")

    report = render_full_cache_report(synthetic_unified_manifest())

    forbidden_phrases = (
        "accuracy",
        "auroc",
        "p-value",
        "statistically significant",
        "outperforms",
        "improves",
        "causal",
        "hypothesis",
        "finding",
        "supports the claim",
    )
    normalized = report.lower()
    for phrase in forbidden_phrases:
        assert phrase not in normalized
