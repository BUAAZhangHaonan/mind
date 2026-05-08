from __future__ import annotations

from mind.trajectory.stage_a_population import (
    PopulationClass,
    classify_entry,
    summarize_population,
)


def _entry(label: int, parsed_answer: int | None) -> dict[str, object]:
    return {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "pope",
        "subset": "popular",
        "sample_id": "sample-1",
        "image_id": "image-1",
        "object_name": "cat",
        "label": label,
        "parsed_answer": parsed_answer,
    }


def test_correct_label_construction() -> None:
    assert classify_entry(_entry(1, 1)) == PopulationClass.CORRECT
    assert classify_entry(_entry(0, 0)) == PopulationClass.CORRECT


def test_hard_hallucination_construction() -> None:
    assert classify_entry(_entry(0, 1)) == PopulationClass.HARD_HALLUCINATION


def test_false_negative_exclusion() -> None:
    assert classify_entry(_entry(1, 0)) == PopulationClass.FALSE_NEGATIVE_ERROR


def test_parsed_none_exclusion() -> None:
    assert classify_entry(_entry(0, None)) == PopulationClass.PARSED_NONE


def test_primary_population_counts() -> None:
    rows = [
        _entry(1, 1),
        _entry(0, 0),
        _entry(0, 1),
        _entry(1, 0),
        _entry(0, None),
    ]

    summary = summarize_population(rows)

    assert summary["num_correct"] == 2
    assert summary["num_hard_hallucination"] == 1
    assert summary["num_false_negative_error"] == 1
    assert summary["num_parsed_none"] == 1
    assert summary["num_primary_population"] == 3
    assert summary["hallucination_rate_in_primary_population"] == 1 / 3
