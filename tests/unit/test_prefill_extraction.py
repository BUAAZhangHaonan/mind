from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mind.data import HallucinationRecord
from mind.extractors import (
    build_prefill_cache_entry,
    build_prefill_readout_entry,
    extract_prefill_entry,
    extract_prefill_vectors,
    save_prefill_cache_shard,
    select_layer_range,
    select_middle_layers,
    stack_prefill_hidden_states,
)


def _record() -> HallucinationRecord:
    return HallucinationRecord(
        sample_id="popular-1",
        image_id=101,
        image_path="COCO_val2014_000000000101.jpg",
        question="Is there a dog in the image?",
        label=1,
        object_name="dog",
        split="val",
        subset="popular",
        source_dataset="pope",
    )


def test_select_middle_layers_picks_middle_half_evenly() -> None:
    assert select_middle_layers(total_layers=32, count=4) == [8, 13, 18, 23]


def test_select_layer_range_supports_early_middle_late_and_all() -> None:
    assert select_layer_range(total_layers=32, count=4, range_name="early") == [0, 5, 10, 15]
    assert select_layer_range(total_layers=32, count=4, range_name="middle") == [8, 13, 18, 23]
    assert select_layer_range(total_layers=32, count=4, range_name="late") == [16, 21, 26, 31]
    assert select_layer_range(total_layers=8, count=8, range_name="all") == list(range(8))


def test_extract_prefill_vectors_uses_last_prefill_token_from_selected_layers() -> None:
    hidden_states = tuple(torch.full((1, 3, 2), fill_value=float(index)) for index in range(5))

    vectors = extract_prefill_vectors(hidden_states, selected_layers=[0, 2, 3], token_index=-1)

    assert vectors.shape == (3, 2)
    assert torch.equal(vectors[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(vectors[1], torch.tensor([3.0, 3.0]))
    assert torch.equal(vectors[2], torch.tensor([4.0, 4.0]))


def test_stack_prefill_hidden_states_keeps_full_prompt_sequence() -> None:
    hidden_states = tuple(torch.full((2, 3, 2), fill_value=float(index)) for index in range(5))

    stacked = stack_prefill_hidden_states(hidden_states, batch_index=1)

    assert stacked.shape == (4, 3, 2)
    assert torch.equal(stacked[0], torch.full((3, 2), 1.0))
    assert torch.equal(stacked[-1], torch.full((3, 2), 4.0))


def test_build_prefill_cache_entry_captures_metadata_and_logits() -> None:
    entry = build_prefill_cache_entry(
        record=_record(),
        answer_text="Yes",
        parsed_answer=1,
        selected_layers=[8, 13],
        layer_vectors=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        first_token_logits=torch.tensor([0.2, 0.8]),
    )

    assert entry["sample_id"] == "popular-1"
    assert entry["selected_layers"] == [8, 13]
    assert entry["parsed_answer"] == 1
    assert entry["layer_vectors"].shape == (2, 2)
    assert torch.equal(entry["first_token_logits"], torch.tensor([0.2, 0.8]))


def test_build_prefill_readout_entry_captures_boundaries_and_full_states() -> None:
    entry = build_prefill_readout_entry(
        record=_record(),
        answer_text="Yes",
        parsed_answer=1,
        full_hidden_states=torch.ones((4, 3, 2)),
        query_token_index=2,
        vision_token_span=(0, 1),
        first_token_logits=torch.tensor([0.2, 0.8]),
        vision_features=torch.tensor([1.0, 2.0]),
    )

    assert entry["sample_id"] == "popular-1"
    assert entry["query_token_index"] == 2
    assert entry["vision_token_span"] == [0, 1]
    assert entry["full_hidden_states"].shape == (4, 3, 2)
    assert torch.equal(entry["vision_features"], torch.tensor([1.0, 2.0]))


def test_save_prefill_cache_shard_writes_torch_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "shard-00000.pt"
    payload = [{"sample_id": "sample-1", "selected_layers": [1], "layer_vectors": torch.ones(1, 2)}]

    save_prefill_cache_shard(payload, output_path)

    restored = torch.load(output_path, weights_only=False)
    assert restored[0]["sample_id"] == "sample-1"


def test_save_prefill_cache_shard_casts_layer_vectors_but_preserves_logits_dtype(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "shard-00000.pt"
    payload = [
        {
            "sample_id": "sample-1",
            "selected_layers": [1],
            "layer_vectors": torch.ones((1, 2), dtype=torch.float32),
            "first_token_logits": torch.tensor([0.2, 0.8], dtype=torch.float32),
        }
    ]

    save_prefill_cache_shard(payload, output_path)

    restored = torch.load(output_path, weights_only=False)
    assert restored[0]["layer_vectors"].dtype == torch.float16
    assert restored[0]["first_token_logits"].dtype == torch.float32


def test_extract_prefill_entry_runs_wrapper_and_model_contract() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeWrapper:
        def prepare_batch_inputs(self, processor, *, questions, image_paths, device: str):
            assert processor == "processor"
            assert questions == ["Is there a dog in the image?"]
            assert image_paths == ["COCO_val2014_000000000101.jpg"]
            return FakeBatch(
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            ).to(device)

        def decode_generation(self, processor, *, generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
            assert processor == "processor"
            assert torch.equal(prompt_input_ids, torch.tensor([[10, 11, 12]]))
            assert torch.equal(generated_ids, torch.tensor([[10, 11, 12, 42]]))
            return "Yes"

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["return_dict_in_generate"] is True
            assert kwargs["output_scores"] is True
            assert kwargs["output_hidden_states"] is True
            return type(
                "FakeGenerationOutput",
                (),
                {
                    "sequences": torch.tensor([[10, 11, 12, 42]]),
                    "scores": [torch.tensor([[0.3, 0.7]], dtype=torch.float32)],
                    "hidden_states": [
                        tuple(torch.full((1, 3, 2), fill_value=float(index)) for index in range(5))
                    ],
                },
            )()

    entry = extract_prefill_entry(
        model=FakeModel(),
        processor="processor",
        wrapper=FakeWrapper(),
        record=_record(),
        selected_layers=[0, 2],
        device="cuda:0",
    )

    assert entry["parsed_answer"] == 1
    assert entry["selected_layers"] == [0, 2]
    assert entry["layer_vectors"].shape == (2, 2)
    assert torch.equal(entry["first_token_logits"], torch.tensor([0.3, 0.7]))


def test_extract_prefill_entry_rejects_non_finite_first_token_logits() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeWrapper:
        def prepare_batch_inputs(self, processor, *, questions, image_paths, device: str):
            return FakeBatch(
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            ).to(device)

        def decode_generation(self, processor, *, generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
            return "Yes"

    class FakeModel:
        def generate(self, **kwargs):
            return type(
                "FakeGenerationOutput",
                (),
                {
                    "sequences": torch.tensor([[10, 11, 12, 42]]),
                    "scores": [torch.tensor([[float("nan"), 0.7]], dtype=torch.float32)],
                    "hidden_states": [
                        tuple(torch.full((1, 3, 2), fill_value=float(index)) for index in range(5))
                    ],
                },
            )()

    with pytest.raises(RuntimeError) as exc_info:
        extract_prefill_entry(
            model=FakeModel(),
            processor="processor",
            wrapper=FakeWrapper(),
            record=_record(),
            selected_layers=[0, 2],
            device="cuda:0",
        )

    message = str(exc_info.value)
    assert "popular-1" in message
    assert "COCO_val2014_000000000101.jpg" in message
    assert "finite_logits=1/2" in message
