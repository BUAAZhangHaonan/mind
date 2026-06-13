from __future__ import annotations

from .conftest import stage_d_attr


def test_stage_d_protocol_set_is_exact_and_optional_protocol_is_labeled() -> None:
    protocol_contract = stage_d_attr("stage_d_protocols", "stage_d_protocol_contract")()

    assert protocol_contract["primary_protocols"] == [
        "repope_to_repope",
        "repope_to_pope",
        "repope_to_dashb",
        "pope_to_dashb",
    ]
    assert protocol_contract["optional_protocols"] == ["pope_to_repope"]
    assert protocol_contract["default_protocols"] == [
        "repope_to_repope",
        "repope_to_pope",
        "repope_to_dashb",
        "pope_to_dashb",
    ]
    assert all(row["calibration_split"] == "source/cal" for row in protocol_contract["protocol_rows"])


def test_oracle_target_calibration_rows_are_diagnostic_only() -> None:
    make_rows = stage_d_attr("stage_d_protocols", "build_stage_d_calibration_scopes")

    rows = make_rows("repope_to_dashb")

    assert rows == [
        {
            "protocol": "repope_to_dashb",
            "calibration_scope": "source_calibration",
            "diagnostic_only": False,
        },
        {
            "protocol": "repope_to_dashb",
            "calibration_scope": "oracle_target_calibration",
            "diagnostic_only": True,
        },
    ]
