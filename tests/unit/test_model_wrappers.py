from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from mind.config import ModelConfig
from mind.models import (
    InternVLWrapper,
    LoadedModelBundle,
    QwenWrapper,
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


def test_qwen_text_prepare_inputs_uses_chat_template_and_device() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["content"][0]["text"] == "Answer yes or no."
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
