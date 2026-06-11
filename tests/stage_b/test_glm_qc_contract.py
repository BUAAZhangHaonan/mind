from __future__ import annotations

from copy import deepcopy

from .conftest import GLM_MODEL_ALIAS, PANEL_MODELS, stage_b_attr


def test_glm_qc_classifies_answer_text_without_mutating_cached_parsed_answer() -> None:
    classify_glm_answer_qc = stage_b_attr(
        "stage_b_glm_qc",
        "classify_glm_answer_qc",
    )
    parseable_entry = {
        "model_alias": GLM_MODEL_ALIAS,
        "sample_id": "glm-parseable",
        "answer_text": "Yes, the object is visible.",
        "parsed_answer": None,
    }
    nonparseable_entry = {
        "model_alias": GLM_MODEL_ALIAS,
        "sample_id": "glm-nonparseable",
        "answer_text": "The image is unclear, so I cannot determine.",
        "parsed_answer": None,
    }
    parseable_before = deepcopy(parseable_entry)
    nonparseable_before = deepcopy(nonparseable_entry)

    parseable = classify_glm_answer_qc(parseable_entry)
    nonparseable = classify_glm_answer_qc(nonparseable_entry)

    assert parseable["parseable"] is True
    assert parseable["derived_parsed_answer"] == 1
    assert nonparseable["parseable"] is False
    assert nonparseable["derived_parsed_answer"] is None
    assert parseable_entry == parseable_before
    assert nonparseable_entry == nonparseable_before


def test_glm_can_be_excluded_without_blocking_rest_of_panel() -> None:
    apply_glm_qc_exclusion = stage_b_attr(
        "stage_b_glm_qc",
        "apply_glm_qc_exclusion",
    )

    status = apply_glm_qc_exclusion(
        panel_models=PANEL_MODELS,
        qc_rows=[
            {
                "model_alias": GLM_MODEL_ALIAS,
                "parseable": False,
                "reason": "GLM answer_text is not parseable on the checked cache rows",
            }
        ],
    )

    assert status["blocked"] is False
    assert status["excluded_models"][GLM_MODEL_ALIAS]
    assert set(status["included_models"]) == set(PANEL_MODELS) - {GLM_MODEL_ALIAS}
