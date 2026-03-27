from __future__ import annotations

from types import SimpleNamespace

import torch

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
