from __future__ import annotations

from mind.config import ModelConfig
from mind.models import (
    InternVLWrapper,
    QwenTextWrapper,
    QwenVLWrapper,
    create_model_wrapper,
    parse_yes_no_answer,
    resolve_torch_dtype,
)


def test_parse_yes_no_answer_handles_common_variants() -> None:
    assert parse_yes_no_answer("Yes, there is a dog.") == 1
    assert parse_yes_no_answer("No. There is no train.") == 0
    assert parse_yes_no_answer("The answer is yes") == 1
    assert parse_yes_no_answer("unclear response") is None


def test_create_model_wrapper_picks_expected_wrapper_classes() -> None:
    qwen_vl = ModelConfig(
        name="qwen3-vl-8b",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen_vl",
    )
    qwen_text = ModelConfig(
        name="qwen3.5-4b",
        model_id="Qwen/Qwen3.5-4B",
        family="qwen",
    )
    internvl = ModelConfig(
        name="internvl3.5-8b",
        model_id="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
    )

    assert isinstance(create_model_wrapper(qwen_vl), QwenVLWrapper)
    assert isinstance(create_model_wrapper(qwen_text), QwenTextWrapper)
    assert isinstance(create_model_wrapper(internvl), InternVLWrapper)


def test_resolve_torch_dtype_maps_supported_values() -> None:
    assert str(resolve_torch_dtype("float16")) == "torch.float16"
    assert str(resolve_torch_dtype("bfloat16")) == "torch.bfloat16"
    assert str(resolve_torch_dtype("float32")) == "torch.float32"


def test_wrapper_builds_expected_model_load_kwargs() -> None:
    wrapper = QwenVLWrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            family="qwen_vl",
            dtype="bfloat16",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    )

    kwargs = wrapper.model_load_kwargs(device="cuda")

    assert kwargs["trust_remote_code"] is True
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["device_map"] == "auto"
    assert str(kwargs["torch_dtype"]) == "torch.bfloat16"


def test_qwen_vl_build_messages_includes_image_and_text() -> None:
    wrapper = QwenVLWrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            family="qwen_vl",
        )
    )

    messages = wrapper.build_messages(
        question="Is there a bus in the image?",
        image_path="demo.jpg",
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "demo.jpg"},
                {"type": "text", "text": "Is there a bus in the image?"},
            ],
        }
    ]
