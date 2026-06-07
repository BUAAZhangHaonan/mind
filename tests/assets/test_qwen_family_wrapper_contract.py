from __future__ import annotations

from types import SimpleNamespace

import pytest

from mind.config import ModelConfig
from mind.models.factory import create_model_wrapper
from mind.models import wrappers


def _config(*, name: str, family: str, thinking: dict[str, object] | None = None, offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_id=f"/models/{name}",
        local_path=f"/models/{name}",
        family=family,
        dtype="float16",
        trust_remote_code=False,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking=thinking or {"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id=f"{family}_prompt",
        prompt_template_text="single image plus normalized question",
        hidden_state_index_offset=offset,
    )


@pytest.mark.parametrize(
    ("name", "family", "wrapper_class", "model_class", "processor_class"),
    [
        ("qwen2.5-vl-7b", "qwen2_5_vl", "Qwen25VLWrapper", "Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLProcessor"),
        ("qwen3-vl-8b", "qwen3_vl", "QwenVLWrapper", "AutoModelForImageTextToText", "AutoProcessor"),
        ("qwen3.5-4b", "qwen3_5", "Qwen35VLWrapper", "Qwen3_5ForConditionalGeneration", "Qwen3VLProcessor"),
        ("qwen3.5-9b", "qwen3_5", "Qwen35VLWrapper", "Qwen3_5ForConditionalGeneration", "Qwen3VLProcessor"),
    ],
)
def test_qwen_aliases_map_to_explicit_wrappers(
    name: str,
    family: str,
    wrapper_class: str,
    model_class: str,
    processor_class: str,
) -> None:
    wrapper = create_model_wrapper(_config(name=name, family=family))

    assert type(wrapper).__name__ == wrapper_class
    assert wrapper.expected_model_class_name() == model_class
    assert wrapper.expected_processor_class_name() == processor_class
    assert wrapper.resolve_hidden_state_index_offset() == 1


def test_qwen_generation_kwargs_are_deterministic_and_explicit() -> None:
    wrapper = create_model_wrapper(_config(name="qwen2.5-vl-7b", family="qwen2_5_vl"))

    kwargs = wrapper.deterministic_generation_kwargs(max_new_tokens=1)

    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True
    assert kwargs["output_hidden_states"] is True


def test_qwen35_thinking_is_disabled_with_template_argument() -> None:
    wrapper = create_model_wrapper(
        _config(
            name="qwen3.5-4b",
            family="qwen3_5",
            thinking={"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        )
    )

    assert wrapper.disable_thinking_kwargs() == {"enable_thinking": False}


def test_qwen_hidden_state_index_offset_cannot_be_unknown() -> None:
    wrapper = create_model_wrapper(_config(name="qwen3.5-4b", family="qwen3_5", offset="unknown"))

    with pytest.raises(ValueError, match="hidden_state_index_offset"):
        wrapper.resolve_hidden_state_index_offset()


def test_qwen_load_processor_does_not_require_minicpm_tokenizer_contract(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    wrapper = create_model_wrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id=str(tmp_path),
            local_path=str(tmp_path),
            family="qwen3_vl",
            dtype="float16",
            trust_remote_code=False,
            attn_implementation="eager",
            deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
            thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
            policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
            prompt_template_id="qwen3_vl_prompt",
            prompt_template_text="single image plus normalized question",
            hidden_state_index_offset=1,
        )
    )
    fake_processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    monkeypatch.setattr(wrappers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: fake_processor)

    loaded = wrapper.load_processor()

    assert loaded is fake_processor
    assert loaded.tokenizer.padding_side == "left"
