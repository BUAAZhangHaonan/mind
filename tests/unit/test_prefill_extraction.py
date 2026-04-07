from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from mind.data import HallucinationRecord
from mind.extractors import (
    build_prefill_cache_entry,
    build_prefill_readout_entry,
    extract_prefill_entry,
    extract_prefill_entries,
    extract_prefill_readout_entries,
    extract_prefill_vectors,
    save_prefill_cache_shard,
    select_layer_range,
    select_middle_layers,
    stack_prefill_hidden_states,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_eval_states.py"
SPEC = importlib.util.spec_from_file_location("extract_eval_states", SCRIPT_PATH)
extract_eval_states = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(extract_eval_states)

READOUT_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_readout_states.py"
READOUT_SPEC = importlib.util.spec_from_file_location("extract_readout_states", READOUT_SCRIPT_PATH)
extract_readout_states = importlib.util.module_from_spec(READOUT_SPEC)
assert READOUT_SPEC is not None and READOUT_SPEC.loader is not None
READOUT_SPEC.loader.exec_module(extract_readout_states)


def test_select_middle_layers_picks_middle_half_evenly() -> None:
    selected = select_middle_layers(total_layers=32, count=4)

    assert selected == [8, 13, 18, 23]


def test_select_layer_range_supports_early_middle_and_late() -> None:
    assert select_layer_range(total_layers=32, count=4, range_name="early") == [0, 5, 10, 15]
    assert select_layer_range(total_layers=32, count=4, range_name="middle") == [8, 13, 18, 23]
    assert select_layer_range(total_layers=32, count=4, range_name="late") == [16, 21, 26, 31]


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


def test_stack_prefill_hidden_states_keeps_full_prompt_sequence() -> None:
    hidden_states = tuple(
        torch.full((2, 3, 2), fill_value=float(index))
        for index in range(5)
    )

    stacked = stack_prefill_hidden_states(hidden_states, batch_index=1)

    assert stacked.shape == (4, 3, 2)
    assert torch.equal(stacked[0], torch.full((3, 2), 1.0))
    assert torch.equal(stacked[-1], torch.full((3, 2), 4.0))


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


def test_build_prefill_readout_entry_captures_boundaries_and_full_states() -> None:
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

    entry = build_prefill_readout_entry(
        record=record,
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


def test_extract_readout_states_reuses_same_output_layout(tmp_path: Path) -> None:
    output_path = extract_readout_states.build_cache_output_path(
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        dataset_name="pope",
        split="popular",
        shard_index=4,
    )

    assert output_path == tmp_path / "qwen3-vl-8b" / "pope" / "popular" / "shard-00004.pt"


def test_extract_readout_run_extraction_batches_records(tmp_path: Path, monkeypatch) -> None:
    records = [
        HallucinationRecord(
            sample_id=f"sample-{index}",
            image_id=index,
            image_path=f"{index}.jpg",
            question=f"Q{index}?",
            label=index % 2,
            object_name="dog",
            split="val",
            subset="popular",
            source_dataset="pope",
        )
        for index in range(5)
    ]
    batch_sizes: list[int] = []

    class FakeWrapper:
        def load_processor(self):
            return "processor"

        def load_model(self, device: str):
            assert device == "cuda:0"
            return "model"

    monkeypatch.setattr(
        extract_readout_states,
        "load_yaml_config",
        lambda path, config_cls: type("Config", (), {"name": "qwen3-vl-8b"})(),
    )
    monkeypatch.setattr(extract_readout_states, "create_model_wrapper", lambda config: FakeWrapper())
    monkeypatch.setattr(extract_readout_states, "load_normalized_records", lambda path: list(records))
    monkeypatch.setattr(
        extract_readout_states,
        "extract_prefill_readout_entries",
        lambda **kwargs: (
            batch_sizes.append(len(kwargs["records"]))
            or [{"sample_id": record.sample_id} for record in kwargs["records"]]
        ),
    )

    output_paths = extract_readout_states.run_extraction(
        records_path=tmp_path / "popular.jsonl",
        model_config_path=tmp_path / "model.yaml",
        output_root=tmp_path / "readouts",
        dataset_name="pope",
        split="popular",
        image_root=None,
        device="cuda:0",
        shard_size=4,
        batch_size=2,
        max_new_tokens=1,
        limit=0,
    )

    assert batch_sizes == [2, 2, 1]
    assert len(output_paths) == 2
    first_shard = torch.load(output_paths[0], weights_only=False)
    second_shard = torch.load(output_paths[1], weights_only=False)
    assert [row["sample_id"] for row in first_shard] == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert [row["sample_id"] for row in second_shard] == ["sample-4"]


def test_extract_readout_run_extraction_resumes_from_contiguous_partial_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    records = [
        HallucinationRecord(
            sample_id=f"sample-{index}",
            image_id=index,
            image_path=f"{index}.jpg",
            question=f"Q{index}?",
            label=index % 2,
            object_name="dog",
            split="val",
            subset="popular",
            source_dataset="pope",
        )
        for index in range(5)
    ]
    batch_sizes: list[int] = []
    model_load_calls: list[str] = []

    class FakeWrapper:
        def load_processor(self):
            return "processor"

        def load_model(self, device: str):
            model_load_calls.append(device)
            return "model"

    monkeypatch.setattr(
        extract_readout_states,
        "load_yaml_config",
        lambda path, config_cls: type("Config", (), {"name": "qwen3-vl-8b"})(),
    )
    monkeypatch.setattr(extract_readout_states, "create_model_wrapper", lambda config: FakeWrapper())
    monkeypatch.setattr(extract_readout_states, "load_normalized_records", lambda path: list(records))
    monkeypatch.setattr(
        extract_readout_states,
        "extract_prefill_readout_entries",
        lambda **kwargs: (
            batch_sizes.append(len(kwargs["records"]))
            or [{"sample_id": record.sample_id} for record in kwargs["records"]]
        ),
    )

    existing_shard = tmp_path / "readouts" / "qwen3-vl-8b" / "pope" / "popular" / "shard-00000.pt"
    existing_shard.parent.mkdir(parents=True, exist_ok=True)
    torch.save([{"sample_id": "sample-0"}, {"sample_id": "sample-1"}], existing_shard)

    output_paths = extract_readout_states.run_extraction(
        records_path=tmp_path / "popular.jsonl",
        model_config_path=tmp_path / "model.yaml",
        output_root=tmp_path / "readouts",
        dataset_name="pope",
        split="popular",
        image_root=None,
        device="cuda:0",
        shard_size=2,
        batch_size=2,
        max_new_tokens=1,
        limit=0,
    )

    assert model_load_calls == ["cuda:0"]
    assert batch_sizes == [2, 1]
    assert [path.name for path in output_paths] == [
        "shard-00000.pt",
        "shard-00001.pt",
        "shard-00002.pt",
    ]
    resumed_shard = torch.load(output_paths[1], weights_only=False)
    final_shard = torch.load(output_paths[2], weights_only=False)
    assert [row["sample_id"] for row in resumed_shard] == ["sample-2", "sample-3"]
    assert [row["sample_id"] for row in final_shard] == ["sample-4"]


def test_extract_readout_run_extraction_rejects_non_contiguous_partial_shards(
    tmp_path: Path, monkeypatch
) -> None:
    records = [
        HallucinationRecord(
            sample_id=f"sample-{index}",
            image_id=index,
            image_path=f"{index}.jpg",
            question=f"Q{index}?",
            label=index % 2,
            object_name="dog",
            split="val",
            subset="popular",
            source_dataset="pope",
        )
        for index in range(6)
    ]

    monkeypatch.setattr(
        extract_readout_states,
        "load_yaml_config",
        lambda path, config_cls: type("Config", (), {"name": "qwen3-vl-8b"})(),
    )
    monkeypatch.setattr(extract_readout_states, "load_normalized_records", lambda path: list(records))

    shard_root = tmp_path / "readouts" / "qwen3-vl-8b" / "pope" / "popular"
    shard_root.mkdir(parents=True, exist_ok=True)
    torch.save([{"sample_id": "sample-0"}], shard_root / "shard-00000.pt")
    torch.save([{"sample_id": "sample-4"}], shard_root / "shard-00002.pt")

    with pytest.raises(ValueError, match="contiguous shard prefix"):
        extract_readout_states.run_extraction(
            records_path=tmp_path / "popular.jsonl",
            model_config_path=tmp_path / "model.yaml",
            output_root=tmp_path / "readouts",
            dataset_name="pope",
            split="popular",
            image_root=None,
            device="cuda:0",
            shard_size=2,
            batch_size=2,
            max_new_tokens=1,
            limit=0,
        )


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


def test_resolve_image_paths_joins_relative_paths_with_image_root(tmp_path: Path) -> None:
    records = [
        HallucinationRecord(
            sample_id="sample-0",
            image_id=0,
            image_path="COCO_val2014_000000000000.jpg",
            question="Q?",
            label=0,
            object_name="dog",
            split="val",
            subset="popular",
            source_dataset="pope",
        ),
        HallucinationRecord(
            sample_id="sample-1",
            image_id=1,
            image_path=str(tmp_path / "already-absolute.jpg"),
            question="Q?",
            label=1,
            object_name="cat",
            split="val",
            subset="popular",
            source_dataset="pope",
        ),
    ]

    resolved = extract_eval_states.resolve_image_paths(records, image_root=tmp_path / "val2014")

    assert resolved[0].image_path == str(tmp_path / "val2014" / "COCO_val2014_000000000000.jpg")
    assert resolved[1].image_path == str(tmp_path / "already-absolute.jpg")


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

        def generate(self, model, processor, *, model_inputs, max_new_tokens: int):
            assert processor == "processor"
            assert max_new_tokens == 4
            assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12]]))
            return model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 4
            assert kwargs["do_sample"] is False
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
                        tuple(
                            torch.full((1, 3, 2), fill_value=float(index))
                            for index in range(5)
                        )
                    ],
                },
            )()

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


def test_extract_prefill_entries_decode_each_sample_from_true_prompt_length() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []

        def batch_decode(self, token_ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool):
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is True
            self.calls.append(token_ids)
            return ["Yes"] if token_ids == [[42]] else ["No"]

    class FakeWrapper:
        def prepare_batch_inputs(self, processor, *, questions, image_paths, device: str):
            assert processor.__class__.__name__ == "FakeProcessor"
            assert questions == ["Q1", "Q2"]
            assert image_paths == ["a.jpg", "b.jpg"]
            return FakeBatch(
                {
                    "input_ids": torch.tensor([[10, 11, 12], [0, 20, 21]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]]),
                }
            ).to(device)

        def decode_generation(self, processor, *, generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
            prompt_length = int(prompt_input_ids.shape[-1])
            continuation = generated_ids[:, prompt_length:]
            decoded = processor.batch_decode(
                continuation.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return str(decoded[0]).strip()

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["return_dict_in_generate"] is True
            assert kwargs["output_scores"] is True
            assert kwargs["output_hidden_states"] is True
            return type(
                "FakeGenerationOutput",
                (),
                {
                    "sequences": torch.tensor([[10, 11, 12, 42], [0, 20, 21, 43]]),
                    "scores": [torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)],
                    "hidden_states": [
                        tuple(
                            torch.tensor(
                                [
                                    [[100.0 + layer, 101.0 + layer], [110.0 + layer, 111.0 + layer], [120.0 + layer, 121.0 + layer]],
                                    [[200.0 + layer, 201.0 + layer], [210.0 + layer, 211.0 + layer], [220.0 + layer, 221.0 + layer]],
                                ]
                            )
                            for layer in range(5)
                        )
                    ],
                },
            )()

    records = [
        HallucinationRecord(
            sample_id="sample-1",
            image_id=1,
            image_path="a.jpg",
            question="Q1",
            label=1,
            object_name="dog",
            split="val",
            subset="popular",
            source_dataset="pope",
        ),
        HallucinationRecord(
            sample_id="sample-2",
            image_id=2,
            image_path="b.jpg",
            question="Q2",
            label=0,
            object_name="cat",
            split="val",
            subset="popular",
            source_dataset="pope",
        ),
    ]

    processor = FakeProcessor()
    entries = extract_prefill_entries(
        model=FakeModel(),
        processor=processor,
        wrapper=FakeWrapper(),
        records=records,
        selected_layers=[0, 2],
        device="cuda:0",
    )

    assert processor.calls == [[[42]], [[43]]]
    assert [entry["parsed_answer"] for entry in entries] == [1, 0]
    assert torch.equal(entries[0]["layer_vectors"][0], torch.tensor([121.0, 122.0]))
    assert torch.equal(entries[1]["layer_vectors"][0], torch.tensor([221.0, 222.0]))
    assert torch.equal(entries[0]["first_token_logits"], torch.tensor([0.2, 0.8]))
    assert torch.equal(entries[1]["first_token_logits"], torch.tensor([0.7, 0.3]))


def test_extract_prefill_entries_falls_back_to_wrapper_prefill_forward_states() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeWrapper:
        def prepare_batch_inputs(self, processor, *, questions, image_paths, device: str):
            assert processor == "processor"
            assert questions == ["Q1"]
            assert image_paths == ["a.jpg"]
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

        def extract_prefill_hidden_states(self, model, processor, *, model_inputs):
            assert processor == "processor"
            assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12]]))
            return tuple(
                torch.full((1, 3, 2), fill_value=float(index))
                for index in range(5)
            )

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["output_hidden_states"] is True
            return type(
                "FakeGenerationOutput",
                (),
                {
                    "sequences": torch.tensor([[10, 11, 12, 42]]),
                    "scores": [torch.tensor([[0.3, 0.7]], dtype=torch.float32)],
                    "hidden_states": [None],
                },
            )()

    record = HallucinationRecord(
        sample_id="sample-1",
        image_id=1,
        image_path="a.jpg",
        question="Q1",
        label=1,
        object_name="dog",
        split="val",
        subset="popular",
        source_dataset="pope",
    )

    entries = extract_prefill_entries(
        model=FakeModel(),
        processor="processor",
        wrapper=FakeWrapper(),
        records=[record],
        selected_layers=[0, 2],
        device="cuda:0",
    )

    assert len(entries) == 1
    assert torch.equal(entries[0]["layer_vectors"][0], torch.tensor([1.0, 1.0]))
    assert torch.equal(entries[0]["layer_vectors"][1], torch.tensor([3.0, 3.0]))


def test_extract_prefill_readout_entries_capture_query_and_vision_boundaries() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeWrapper:
        def prepare_batch_inputs(self, processor, *, questions, image_paths, device: str):
            assert processor == "processor"
            assert questions == ["Q1"]
            assert image_paths == ["a.jpg"]
            return FakeBatch(
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            ).to(device)

        def resolve_query_token_index(self, processor, *, model_inputs, batch_index: int) -> int:
            assert processor == "processor"
            assert batch_index == 0
            return 2

        def resolve_vision_token_span(self, model, processor, *, model_inputs, batch_index: int):
            assert processor == "processor"
            assert batch_index == 0
            return (0, 1)

        def extract_preprojector_vision_features(self, model, processor, *, model_inputs, batch_index: int):
            assert processor == "processor"
            assert batch_index == 0
            return torch.tensor([1.0, 2.0, 3.0])

        def decode_generation(self, processor, *, generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
            assert processor == "processor"
            assert torch.equal(prompt_input_ids, torch.tensor([[10, 11, 12]]))
            assert torch.equal(generated_ids, torch.tensor([[10, 11, 12, 42]]))
            return "Yes"

        def generate(self, model, processor, *, model_inputs, max_new_tokens: int):
            assert processor == "processor"
            assert max_new_tokens == 1
            return model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 1
            return type(
                "FakeGenerationOutput",
                (),
                {
                    "sequences": torch.tensor([[10, 11, 12, 42]]),
                    "scores": [torch.tensor([[0.3, 0.7]], dtype=torch.float32)],
                    "hidden_states": [
                        tuple(
                            torch.full((1, 3, 2), fill_value=float(index))
                            for index in range(5)
                        )
                    ],
                },
            )()

    record = HallucinationRecord(
        sample_id="popular-1",
        image_id=101,
        image_path="a.jpg",
        question="Q1",
        label=1,
        object_name="dog",
        split="val",
        subset="popular",
        source_dataset="pope",
    )

    entries = extract_prefill_readout_entries(
        model=FakeModel(),
        processor="processor",
        wrapper=FakeWrapper(),
        records=[record],
        device="cuda:0",
    )

    assert len(entries) == 1
    assert entries[0]["query_token_index"] == 2
    assert entries[0]["vision_token_span"] == [0, 1]
    assert entries[0]["full_hidden_states"].shape == (4, 3, 2)
    assert torch.equal(entries[0]["vision_features"], torch.tensor([1.0, 2.0, 3.0]))


def test_extract_prefill_readout_entries_fall_back_to_wrapper_prefill_forward_states() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeWrapper:
        def prepare_batch_inputs(self, processor, *, questions, image_paths, device: str):
            assert processor == "processor"
            assert questions == ["Q1"]
            assert image_paths == ["a.jpg"]
            return FakeBatch(
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            ).to(device)

        def resolve_query_token_index(self, processor, *, model_inputs, batch_index: int) -> int:
            assert processor == "processor"
            assert batch_index == 0
            return 2

        def decode_generation(self, processor, *, generated_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
            assert processor == "processor"
            assert torch.equal(prompt_input_ids, torch.tensor([[10, 11, 12]]))
            assert torch.equal(generated_ids, torch.tensor([[10, 11, 12, 42]]))
            return "Yes"

        def extract_prefill_hidden_states(self, model, processor, *, model_inputs):
            assert processor == "processor"
            assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12]]))
            return tuple(
                torch.full((1, 3, 2), fill_value=float(index))
                for index in range(5)
            )

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["output_hidden_states"] is True
            return type(
                "FakeGenerationOutput",
                (),
                {
                    "sequences": torch.tensor([[10, 11, 12, 42]]),
                    "scores": [torch.tensor([[0.3, 0.7]], dtype=torch.float32)],
                    "hidden_states": [None],
                },
            )()

    record = HallucinationRecord(
        sample_id="sample-1",
        image_id=1,
        image_path="a.jpg",
        question="Q1",
        label=1,
        object_name="dog",
        split="val",
        subset="popular",
        source_dataset="pope",
    )

    entries = extract_prefill_readout_entries(
        model=FakeModel(),
        processor="processor",
        wrapper=FakeWrapper(),
        records=[record],
        device="cuda:0",
    )

    assert len(entries) == 1
    assert entries[0]["query_token_index"] == 2
    assert entries[0]["full_hidden_states"].shape == (4, 3, 2)
    assert torch.equal(entries[0]["full_hidden_states"][0], torch.full((3, 2), 1.0))
