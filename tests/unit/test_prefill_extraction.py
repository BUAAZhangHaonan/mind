from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from mind.data import HallucinationRecord
from mind.extractors import (
    build_prefill_cache_entry,
    extract_prefill_entry,
    extract_prefill_vectors,
    save_prefill_cache_shard,
    select_middle_layers,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_eval_states.py"
SPEC = importlib.util.spec_from_file_location("extract_eval_states", SCRIPT_PATH)
extract_eval_states = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(extract_eval_states)


def test_select_middle_layers_picks_middle_half_evenly() -> None:
    selected = select_middle_layers(total_layers=32, count=4)

    assert selected == [8, 13, 18, 23]


def test_extract_prefill_vectors_uses_last_prefill_token_from_selected_layers() -> None:
    hidden_states = tuple(
        torch.full((1, 3, 2), fill_value=float(index))
        for index in range(5)
    )

    vectors = extract_prefill_vectors(hidden_states, selected_layers=[0, 2, 3], token_index=-1)

    assert vectors.shape == (3, 2)
    assert torch.equal(vectors[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(vectors[1], torch.tensor([3.0, 3.0]))
    assert torch.equal(vectors[2], torch.tensor([4.0, 4.0]))


def test_build_prefill_cache_entry_captures_metadata_and_logits() -> None:
    record = HallucinationRecord(
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

    entry = build_prefill_cache_entry(
        record=record,
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


def test_save_prefill_cache_shard_writes_torch_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "shard-00000.pt"
    payload = [{"sample_id": "sample-1", "selected_layers": [1], "layer_vectors": torch.ones(1, 2)}]

    save_prefill_cache_shard(payload, output_path)

    restored = torch.load(output_path)
    assert restored[0]["sample_id"] == "sample-1"


def test_build_cache_output_path_uses_expected_layout(tmp_path: Path) -> None:
    output_path = extract_eval_states.build_cache_output_path(
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        dataset_name="pope",
        split="popular",
        shard_index=3,
    )

    assert output_path == tmp_path / "qwen3-vl-8b" / "pope" / "popular" / "shard-00003.pt"


def test_load_normalized_records_reads_jsonl_rows(tmp_path: Path) -> None:
    source = tmp_path / "popular.jsonl"
    source.write_text(
        '{"sample_id":"popular-1","image_id":101,"image_path":"a.jpg","question":"Q?","label":1,"object_name":"dog","split":"val","subset":"popular","source_dataset":"pope"}\n',
        encoding="utf-8",
    )

    records = extract_eval_states.load_normalized_records(source)

    assert len(records) == 1
    assert records[0].sample_id == "popular-1"
    assert records[0].object_name == "dog"


def test_iter_record_shards_preserves_order_and_size() -> None:
    records = [
        HallucinationRecord(
            sample_id=f"sample-{index}",
            image_id=index,
            image_path=f"{index}.jpg",
            question="Q?",
            label=index % 2,
            object_name="dog",
            split="val",
            subset="popular",
            source_dataset="pope",
        )
        for index in range(5)
    ]

    shards = list(extract_eval_states.iter_record_shards(records, shard_size=2))

    assert [len(shard) for shard in shards] == [2, 2, 1]
    assert shards[0][0].sample_id == "sample-0"
    assert shards[2][0].sample_id == "sample-4"


def test_extract_prefill_entry_runs_wrapper_and_model_contract() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeWrapper:
        def prepare_inputs(self, processor, *, question: str, image_path: str | None, device: str):
            assert processor == "processor"
            assert question == "Is there a dog in the image?"
            assert image_path == "COCO_val2014_000000000101.jpg"
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

    class FakeOutput:
        def __init__(self):
            self.hidden_states = tuple(
                torch.full((1, 3, 2), fill_value=float(index))
                for index in range(5)
            )
            self.logits = torch.tensor(
                [[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]],
                dtype=torch.float32,
            )

    class FakeModel:
        def __call__(self, **kwargs):
            assert kwargs["output_hidden_states"] is True
            assert kwargs["return_dict"] is True
            return FakeOutput()

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 4
            assert kwargs["do_sample"] is False
            return torch.tensor([[10, 11, 12, 42]])

    record = HallucinationRecord(
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

    entry = extract_prefill_entry(
        model=FakeModel(),
        processor="processor",
        wrapper=FakeWrapper(),
        record=record,
        selected_layers=[0, 2],
        device="cuda:0",
    )

    assert entry["parsed_answer"] == 1
    assert entry["selected_layers"] == [0, 2]
    assert entry["layer_vectors"].shape == (2, 2)
    assert torch.equal(entry["first_token_logits"], torch.tensor([0.3, 0.7]))
