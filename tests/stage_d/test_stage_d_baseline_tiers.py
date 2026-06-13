from __future__ import annotations

import numpy as np

from .conftest import stage_d_attr, stage_d_script_attr


def test_stage_d_tier_methods_are_frozen_and_halp_variants_are_distinct() -> None:
    tiers = stage_d_attr("stage_d_protocols", "stage_d_baseline_tiers")()

    assert tiers["tierA"] == [
        "MIND-main",
        "MIND-param",
        "logistic(z)",
        "final-hidden linear probe",
        "output-confidence",
        "HALP-lite",
    ]
    assert tiers["tierB"] == ["official HALP"]
    assert tiers["roles"]["MIND-main"] == "frozen_main_method"
    assert tiers["roles"]["MIND-param"] == "parametric_secondary"
    assert tiers["roles"]["logistic(z)"] == "same_embedding_supervised_comparator"
    assert tiers["roles"]["official HALP"] == "ceiling_broader_access"
    assert tiers["roles"]["HALP-lite"] != tiers["roles"]["official HALP"]


def test_related_method_feasibility_records_required_methods() -> None:
    feasibility = stage_d_attr("stage_d_protocols", "related_method_feasibility_payload")()

    methods = [row["method"] for row in feasibility["methods"]]
    assert methods == ["HALP", "EnsemHalDet", "VIB-Probe", "HaloProbe"]
    for row in feasibility["methods"]:
        assert row["detection_granularity"]
        assert row["required_access_type"]
        assert row["executable_with_current_cache"] in {True, False}
        if not row["executable_with_current_cache"]:
            assert row["incompatibility_reason"]


def test_stage_d_linear_baselines_use_lightweight_sgd_logistic() -> None:
    fit_linear_models = stage_d_script_attr("stage_d_run", "_fit_linear_models")

    features = np.asarray(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.1, 0.0],
            [1.0, 1.1, 1.2],
            [1.2, 1.1, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    models = fit_linear_models(features, labels, seed=20260506)

    assert sorted(models) == [0.1, 1.0, 10.0]
    for model in models.values():
        classifier = model.named_steps["classifier"]
        assert classifier.__class__.__name__ == "SGDClassifier"
        assert classifier.loss == "log_loss"
