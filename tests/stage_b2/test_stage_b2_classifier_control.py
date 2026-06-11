from __future__ import annotations

from .conftest import stage_b2_attr


def test_classifier_control_is_lightweight_logistic_and_secondary() -> None:
    classifier_control_config = stage_b2_attr(
        "stage_b2_status",
        "classifier_control_config",
    )

    config = classifier_control_config()

    assert config["readout"] == "Diag-Classifier"
    assert config["model"] == "logistic_regression"
    assert config["role"] == "secondary_control"
    assert config["uses_large_mlp"] is False
    assert config["primary_decision_signal"] is False
