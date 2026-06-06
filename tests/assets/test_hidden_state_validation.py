from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from mind.models.asset_validation import (
    DeterminismPair,
    validate_determinism_pair,
    validate_hidden_state_entries,
)


def _entry() -> dict[str, object]:
    return {
        "sample_id": "s0",
        "image_id": 1,
        "image_path": "/tmp/image.jpg",
        "question": "Is there a dog in the image?",
        "label": 1,
        "object_name": "dog",
        "source_dataset": "pope",
        "subset": "popular",
        "answer_text": "Yes",
        "parsed_answer": 1,
        "first_token_logits": torch.tensor([0.1, 0.2, 0.3]),
        "selected_layers": [0, 1, 2],
        "layer_vectors": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "model_name": "demo",
        "model_family": "qwen_vl",
        "token_index": 4,
        "prompt_template_id": "qwen_vl_chat_single_image_raw_question_v1",
    }


def _sidecar() -> dict[str, object]:
    return {
        "total_layers": 3,
        "hidden_dim": 3,
        "token_index": 4,
        "prompt_template_id": "qwen_vl_chat_single_image_raw_question_v1",
        "hidden_state_index_offset": 1,
        "hidden_state_count": 4,
        "selected_layer_hidden_state_indices": [1, 2, 3],
    }


def test_valid_full_layer_shard_passes() -> None:
    result = validate_hidden_state_entries([_entry()], _sidecar())

    assert result.status == "verified"


def test_valid_shard_allows_per_entry_token_index_variation() -> None:
    first = _entry()
    second = _entry()
    second["sample_id"] = "s1"
    second["token_index"] = 9

    result = validate_hidden_state_entries([first, second], _sidecar())

    assert result.status == "verified"


@pytest.mark.parametrize(
    "mutator, reason",
    [
        (lambda entry: entry.pop("layer_vectors"), "layer_vectors"),
        (lambda entry: entry.update({"selected_layers": [0, 2, 3]}), "selected_layers"),
        (lambda entry: entry.update({"selected_layers": [0, 1], "layer_vectors": entry["layer_vectors"][:2]}), "total_layers"),
        (lambda entry: entry.update({"layer_vectors": torch.tensor([[1.0, float("nan"), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])}), "finite"),
        (lambda entry: entry.update({"layer_vectors": torch.ones((3, 3))}), "constant"),
    ],
)
def test_invalid_structural_cases_fail(mutator, reason: str) -> None:
    entry = _entry()
    mutator(entry)

    result = validate_hidden_state_entries([entry], _sidecar())

    assert result.status == "failed_validation"
    assert reason in result.reason


def test_nondeterminism_above_tolerance_fails() -> None:
    first = _entry()
    second = deepcopy(first)
    second["layer_vectors"] = torch.as_tensor(second["layer_vectors"]) + 0.01

    result = validate_determinism_pair(
        DeterminismPair(first=first, second=second),
        layer_tolerance=1e-3,
        logits_tolerance=1e-3,
    )

    assert result.status == "failed_validation"
    assert "layer_vectors" in result.reason


def test_missing_selected_layer_hidden_state_mapping_fails() -> None:
    sidecar = _sidecar()
    sidecar.pop("selected_layer_hidden_state_indices")

    result = validate_hidden_state_entries([_entry()], sidecar)

    assert result.status == "failed_validation"
    assert "sidecar metadata" in result.reason
