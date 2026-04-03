from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from mind.config import ModelConfig
from mind.models import (
    InternVLWrapper,
    LlavaOnevisionWrapper,
    LoadedModelBundle,
    MolmoWrapper,
    QwenWrapper,
    QwenVLWrapper,
    create_model_wrapper,
)


def test_create_model_wrapper_selects_expected_family() -> None:
    qwen_config = ModelConfig(
        name="qwen3-vl-8b",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen",
    )
    internvl_config = ModelConfig(
        name="internvl3.5-8b",
        model_id="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
    )

    qwen_wrapper = create_model_wrapper(qwen_config)
    internvl_wrapper = create_model_wrapper(internvl_config)

    assert isinstance(qwen_wrapper, QwenWrapper)
    assert isinstance(internvl_wrapper, InternVLWrapper)


def test_create_model_wrapper_supports_llava_onevision_and_molmo() -> None:
    llava_config = ModelConfig(
        name="llava-onevision-7b",
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        family="llava_onevision",
    )
    molmo_config = ModelConfig(
        name="molmo-7b-d-0924",
        model_id="allenai/Molmo-7B-D-0924",
        family="molmo",
    )

    llava_wrapper = create_model_wrapper(llava_config)
    molmo_wrapper = create_model_wrapper(molmo_config)

    assert isinstance(llava_wrapper, LlavaOnevisionWrapper)
    assert isinstance(molmo_wrapper, MolmoWrapper)


def test_parse_yes_no_response_strips_thinking_and_punctuation() -> None:
    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    assert wrapper.parse_yes_no_response("<think>reasoning</think>\nYes, there is.") == 1
    assert wrapper.parse_yes_no_response("No.") == 0


def test_qwen_wrapper_formats_yes_no_question_once() -> None:
    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    assert (
        wrapper.format_yes_no_question("Is there a dog in the image?")
        == "Is there a dog in the image? Respond with only one word: yes or no."
    )
    assert (
        wrapper.format_yes_no_question("Is there a dog in the image? Respond with only one word: yes or no.")
        == "Is there a dog in the image? Respond with only one word: yes or no."
    )


def test_load_bundle_uses_standard_transformers_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["processor"] = (model_id, kwargs)
            return SimpleNamespace(name="processor")

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["model"] = (model_id, kwargs)
            return SimpleNamespace(name="model")

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            family="qwen",
            dtype="float16",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    )

    bundle = wrapper.load_bundle(
        model_factory=FakeModelFactory,
        processor_factory=FakeProcessorFactory,
    )

    assert isinstance(bundle, LoadedModelBundle)
    assert captured["processor"] == (
        "Qwen/Qwen3-VL-8B-Instruct",
        {"trust_remote_code": True},
    )
    assert captured["model"][0] == "Qwen/Qwen3-VL-8B-Instruct"
    assert captured["model"][1]["device_map"] == "auto"
    assert captured["model"][1]["attn_implementation"] == "sdpa"
    assert captured["model"][1]["torch_dtype"] == torch.float16


def test_molmo_wrapper_forces_eager_attention_for_model_load(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_id: str, **kwargs):
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return SimpleNamespace(name="molmo-model")

    monkeypatch.setattr(
        "mind.models.wrappers.AutoModelForCausalLM.from_pretrained",
        fake_from_pretrained,
    )

    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
            dtype="float16",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    )

    model = wrapper.load_model(device="cuda")

    assert model.name == "molmo-model"
    assert captured["model_id"] == "allenai/Molmo-7B-D-0924"
    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["attn_implementation"] == "eager"


def test_molmo_wrapper_normalizes_placeholder_past_key_values(monkeypatch) -> None:
    class FakeMolmoModel:
        def __init__(self) -> None:
            self.calls = []

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
            self.calls.append(
                {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "kwargs": kwargs,
                }
            )
            return {"input_ids": input_ids, "past_key_values": past_key_values, **kwargs}

    fake_model = FakeMolmoModel()

    monkeypatch.setattr(
        "mind.models.wrappers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
            dtype="float16",
            attn_implementation="eager",
            trust_remote_code=True,
        )
    )

    model = wrapper.load_model(device="cuda")
    placeholder_cache = [(None, None)] * 4
    result = model.prepare_inputs_for_generation(
        torch.tensor([[1, 2, 3]]),
        past_key_values=placeholder_cache,
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.bool),
    )

    assert fake_model.calls[0]["past_key_values"] is None
    assert result["past_key_values"] is None


def test_base_wrapper_extract_prefill_hidden_states_uses_forward_pass() -> None:
    class FakeForwardModel:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                hidden_states=tuple(
                    torch.full((1, 3, 2), fill_value=float(index))
                    for index in range(5)
                )
            )

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            family="qwen",
        )
    )
    model = FakeForwardModel()

    hidden_states = wrapper.extract_prefill_hidden_states(
        model,
        "processor",
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        },
    )

    assert len(hidden_states) == 5
    assert torch.equal(hidden_states[0], torch.zeros((1, 3, 2)))
    assert model.calls[0]["return_dict"] is True
    assert model.calls[0]["output_hidden_states"] is True


def test_qwen_wrapper_decode_generation_removes_prompt_tokens() -> None:
    class FakeProcessor:
        def batch_decode(self, token_ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool):
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is True
            assert token_ids == [[42, 43]]
            return ["Yes"]

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    answer = wrapper.decode_generation(
        FakeProcessor(),
        generated_ids=torch.tensor([[10, 11, 42, 43]]),
        prompt_input_ids=torch.tensor([[10, 11]]),
    )

    assert answer == "Yes"


def test_qwen_wrapper_resolves_query_token_index_from_last_non_padding_token() -> None:
    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    index = wrapper.resolve_query_token_index(
        "processor",
        model_inputs={
            "input_ids": torch.tensor([[0, 10, 11, 12]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        },
        batch_index=0,
    )

    assert index == 3


def test_qwen_wrapper_resolves_query_token_index_from_mapping_like_inputs() -> None:
    class FakeBatchFeature:
        def __init__(self, payload):
            self.payload = payload

        def get(self, key, default=None):
            return self.payload.get(key, default)

        def __contains__(self, key):
            return key in self.payload

        def __getitem__(self, key):
            return self.payload[key]

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    index = wrapper.resolve_query_token_index(
        "processor",
        model_inputs=FakeBatchFeature(
            {
                "input_ids": torch.tensor([[0, 10, 11, 12]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1]]),
            }
        ),
        batch_index=0,
    )

    assert index == 3


def test_qwen_text_prepare_inputs_uses_chat_template_and_device() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            assert (
                messages[0]["content"][0]["text"]
                == "Answer yes or no. Respond with only one word: yes or no."
            )
            return "<prompt>"

        def __call__(self, text, return_tensors: str):
            assert text == ["<prompt>"]
            assert return_tensors == "pt"
            return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    batch = wrapper.prepare_inputs(
        FakeProcessor(),
        question="Answer yes or no.",
        image_path=None,
        device="cuda:1",
    )

    assert batch["device"] == "cuda:1"


def test_internvl_prepare_inputs_uses_image_and_prompt(tmp_path: Path) -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            self.calls.append(("template", messages))
            return "<vision-prompt>"

        def __call__(self, *, text, images, return_tensors: str):
            self.calls.append(("call", text, images, return_tensors))
            return FakeBatch({"input_ids": torch.tensor([[4, 5, 6]])})

    image_path = tmp_path / "demo.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    processor = FakeProcessor()
    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    batch = wrapper.prepare_inputs(
        processor,
        question="Is there a plane in the image?",
        image_path=str(image_path),
        device="cuda:0",
    )

    assert batch["device"] == "cuda:0"
    assert processor.calls[1][1] == ["<vision-prompt>"]
    assert len(processor.calls[1][2]) == 1
    assert processor.calls[0][1][0]["content"][1]["text"].endswith("Respond with only one word: yes or no.")


def test_internvl_prepare_batch_inputs_uses_padding_and_multiple_images(tmp_path: Path) -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            prompt = messages[0]["content"][1]["text"]
            self.calls.append(("template", prompt))
            return f"<vision-prompt:{prompt}>"

        def __call__(self, *, text, images, return_tensors: str, padding: bool):
            self.calls.append(("call", text, images, return_tensors, padding))
            return FakeBatch({"input_ids": torch.tensor([[4, 5, 6], [7, 8, 9]])})

    image_path_a = tmp_path / "demo-a.png"
    image_path_b = tmp_path / "demo-b.png"
    Image.new("RGB", (2, 2), color="white").save(image_path_a)
    Image.new("RGB", (2, 2), color="black").save(image_path_b)
    processor = FakeProcessor()
    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    batch = wrapper.prepare_batch_inputs(
        processor,
        questions=["Is there a plane in the image?", "Is there a dog in the image?"],
        image_paths=[str(image_path_a), str(image_path_b)],
        device="cuda:2",
    )

    assert batch["device"] == "cuda:2"
    assert processor.calls[2][1] == [
        "<vision-prompt:Is there a plane in the image? Respond with only one word: yes or no.>",
        "<vision-prompt:Is there a dog in the image? Respond with only one word: yes or no.>",
    ]
    assert len(processor.calls[2][2]) == 2
    assert processor.calls[2][4] is True


def test_qwenvl_wrapper_resolves_vision_token_span_from_image_token_indices() -> None:
    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    span = wrapper.resolve_vision_token_span(
        SimpleNamespace(config=SimpleNamespace(image_token_index=99)),
        "processor",
        model_inputs={"input_ids": torch.tensor([[7, 99, 99, 15]])},
        batch_index=0,
    )

    assert span == (1, 2)


def test_qwenvl_wrapper_resolves_vision_token_span_from_mapping_like_inputs() -> None:
    class FakeBatchFeature:
        def __init__(self, payload):
            self.payload = payload

        def __contains__(self, key):
            return key in self.payload

        def __getitem__(self, key):
            return self.payload[key]

    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    span = wrapper.resolve_vision_token_span(
        SimpleNamespace(config=SimpleNamespace(image_token_index=99)),
        "processor",
        model_inputs=FakeBatchFeature({"input_ids": torch.tensor([[7, 99, 99, 15]])}),
        batch_index=0,
    )

    assert span == (1, 2)


def test_qwenvl_extract_preprojector_features_uses_visual_blocks_before_merger() -> None:
    class FakeBlock:
        def __call__(self, hidden_states, **kwargs):
            del kwargs
            return hidden_states + 1

    class FakeMerger:
        def __call__(self, hidden_states):
            raise AssertionError("merger should not be called for preprojector features")

    visual = SimpleNamespace(
        dtype=torch.float32,
        patch_embed=lambda tensor: tensor,
        fast_pos_embed_interpolate=lambda grid_thw: torch.zeros((4, 2), dtype=torch.float32),
        rot_pos_emb=lambda grid_thw: torch.zeros((4, 1), dtype=torch.float32),
        blocks=[FakeBlock()],
        merger=FakeMerger(),
    )
    wrapper = QwenVLWrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            family="qwen_vl",
        )
    )

    features = wrapper.extract_preprojector_vision_features(
        SimpleNamespace(visual=visual),
        "processor",
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        },
        batch_index=0,
    )

    assert torch.equal(features, torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1)


def test_llava_onevision_extract_preprojector_features_accepts_mapping_like_inputs() -> None:
    class FakeBatchFeature:
        def __init__(self, payload):
            self.payload = payload

        def get(self, key, default=None):
            return self.payload.get(key, default)

    class FakeVisionTower:
        def __call__(self, pixel_values, output_hidden_states: bool):
            assert output_hidden_states is False
            assert tuple(pixel_values.shape) == (2, 3, 4, 4)
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                    ]
                )
            )

    wrapper = LlavaOnevisionWrapper(
        ModelConfig(
            name="llava-onevision-7b",
            model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
            family="llava_onevision",
        )
    )

    features = wrapper.extract_preprojector_vision_features(
        SimpleNamespace(vision_tower=FakeVisionTower()),
        "processor",
        model_inputs=FakeBatchFeature(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.ones((1, 2, 3, 4, 4), dtype=torch.float32),
            }
        ),
        batch_index=0,
    )

    assert torch.equal(
        features,
        torch.tensor(
            [
                [3.0, 4.0],
                [5.0, 6.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        ),
    )


def test_internvl_extract_preprojector_features_flattens_visual_tokens() -> None:
    class FakeVisionTower:
        def __call__(self, *, pixel_values):
            assert tuple(pixel_values.shape) == (2, 3, 4, 4)
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [
                        [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]],
                        [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]],
                    ]
                )
            )

    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    features = wrapper.extract_preprojector_vision_features(
        SimpleNamespace(vision_tower=FakeVisionTower()),
        "processor",
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones((2, 3, 4, 4), dtype=torch.float32),
        },
        batch_index=0,
    )

    assert torch.equal(
        features,
        torch.tensor(
            [
                [12.0, 13.0],
                [14.0, 15.0],
                [22.0, 23.0],
                [24.0, 25.0],
            ]
        ),
    )


def test_llava_onevision_extract_preprojector_features_flattens_patch_tokens() -> None:
    class FakeVisionTower:
        def __call__(self, pixel_values, output_hidden_states: bool):
            assert output_hidden_states is False
            assert tuple(pixel_values.shape) == (2, 3, 4, 4)
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                    ]
                )
            )

    wrapper = LlavaOnevisionWrapper(
        ModelConfig(
            name="llava-onevision-7b",
            model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
            family="llava_onevision",
        )
    )

    features = wrapper.extract_preprojector_vision_features(
        SimpleNamespace(vision_tower=FakeVisionTower()),
        "processor",
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones((1, 2, 3, 4, 4), dtype=torch.float32),
        },
        batch_index=0,
    )

    assert torch.equal(
        features,
        torch.tensor(
            [
                [3.0, 4.0],
                [5.0, 6.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        ),
    )


def test_molmo_prepare_batch_inputs_collates_variable_length_fields(tmp_path: Path) -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []
            self.tokenizer = SimpleNamespace(name="molmo-tokenizer")

        def process(self, *, images, text: str):
            self.calls.append((images, text))
            if "plane" in text:
                return {
                    "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
                    "image_input_idx": torch.tensor([[0, 1]], dtype=torch.long),
                    "images": torch.ones((1, 2, 2), dtype=torch.float32),
                    "image_masks": torch.ones((1, 2), dtype=torch.bool),
                }
            return {
                "input_ids": torch.tensor([21, 22], dtype=torch.long),
                "image_input_idx": torch.tensor([[0]], dtype=torch.long),
                "images": torch.zeros((1, 1, 2), dtype=torch.float32),
                "image_masks": torch.ones((1, 1), dtype=torch.bool),
            }

    image_path_a = tmp_path / "demo-a.png"
    image_path_b = tmp_path / "demo-b.png"
    Image.new("RGB", (2, 2), color="white").save(image_path_a)
    Image.new("RGB", (2, 2), color="black").save(image_path_b)
    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )

    batch = wrapper.prepare_batch_inputs(
        FakeProcessor(),
        questions=["Is there a plane in the image?", "Is there a dog in the image?"],
        image_paths=[str(image_path_a), str(image_path_b)],
        device="cpu",
    )

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert batch["input_ids"][1].tolist() == [21, 22, -1]
    assert tuple(batch["image_input_idx"].shape) == (2, 1, 2)
    assert batch["image_input_idx"][1, 0].tolist() == [0, -1]
    assert batch["images"].dtype == torch.float16


def test_molmo_wrapper_resolves_vision_token_span_from_image_input_idx() -> None:
    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )

    span = wrapper.resolve_vision_token_span(
        SimpleNamespace(),
        "processor",
        model_inputs={
            "image_input_idx": torch.tensor(
                [
                    [[0, 1, 2]],
                    [[3, 4, -1]],
                ],
                dtype=torch.long,
            )
        },
        batch_index=1,
    )

    assert span == (3, 4)


def test_molmo_extract_preprojector_features_filters_masked_tokens() -> None:
    class FakeVisionBackbone:
        def encode_image(self, images):
            assert tuple(images.shape) == (1, 2, 3, 2)
            return (
                torch.tensor(
                    [
                        [
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                        ]
                    ]
                ),
                None,
            )

    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )

    features = wrapper.extract_preprojector_vision_features(
        SimpleNamespace(model=SimpleNamespace(vision_backbone=FakeVisionBackbone())),
        "processor",
        model_inputs={
            "images": torch.ones((1, 2, 3, 2), dtype=torch.float32),
            "image_masks": torch.tensor([[[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]], dtype=torch.float32),
        },
        batch_index=0,
    )

    assert torch.equal(
        features,
        torch.tensor(
            [
                [1.0, 2.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ]
        ),
    )


def test_molmo_generate_uses_generate_from_batch_with_tokenizer() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def generate_from_batch(self, batch, generation_config, **kwargs):
            self.calls.append((batch, generation_config, kwargs))
            return "ok"

    processor = SimpleNamespace(tokenizer=SimpleNamespace(name="molmo-tokenizer"))
    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )
    model = FakeModel()

    result = wrapper.generate(
        model,
        processor,
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        max_new_tokens=4,
    )

    assert result == "ok"
    assert model.calls[0][1].max_new_tokens == 4
    assert model.calls[0][2]["tokenizer"] is processor.tokenizer
    assert model.calls[0][2]["return_dict_in_generate"] is True
    assert model.calls[0][2]["output_scores"] is True
    assert model.calls[0][2]["output_hidden_states"] is True


def test_molmo_load_processor_uses_local_processing_modules(tmp_path: Path, monkeypatch) -> None:
    snapshot_root = tmp_path / "molmo"
    snapshot_root.mkdir()
    (snapshot_root / "image_preprocessing_molmo.py").write_text(
        "\n".join(
            [
                "class MolmoImageProcessor:",
                "    @classmethod",
                "    def from_pretrained(cls, path):",
                "        return cls()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot_root / "preprocessing_molmo.py").write_text(
        "\n".join(
            [
                "from .image_preprocessing_molmo import MolmoImageProcessor",
                "class MolmoProcessor:",
                "    def __init__(self, image_processor=None, tokenizer=None):",
                "        self.image_processor = image_processor",
                "        self.tokenizer = tokenizer",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    monkeypatch.setattr("mind.models.wrappers.snapshot_download", lambda **_: str(snapshot_root))
    monkeypatch.setattr(
        "mind.models.wrappers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )

    processor = wrapper.load_processor()

    assert type(processor).__name__ == "MolmoProcessor"
    assert processor.tokenizer.padding_side == "left"
