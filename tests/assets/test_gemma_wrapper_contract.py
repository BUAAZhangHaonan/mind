from __future__ import annotations

from pathlib import Path

import pytest

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import AssetModel


def _config(*, name: str, offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_id=f"/models/{name}",
        local_path=f"/models/{name}",
        family="gemma3",
        dtype="bfloat16",
        trust_remote_code=False,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="gemma3_chat_single_image_raw_question_v1",
        prompt_template_text="Gemma3 user chat message with one image item followed by the normalized question unchanged.",
        hidden_state_index_offset=offset,
    )


def _asset(path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "gemma-3-4b-it",
        "local_path": str(path),
        "model_config_path": "configs/models/gemma_3_4b_it.yaml",
        "model_id_or_family_name": "gemma3",
        "family": "gemma3",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": False, "disabled_by_default": True, "disable_argument": None},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "gemma3_chat_single_image_raw_question_v1",
        "prompt_template_text": "Gemma3 user chat message with one image item followed by the normalized question unchanged.",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_gemma3_layout(path: Path, *, multimodal: bool = True, processor: bool = True) -> None:
    config = {
        "model_type": "gemma3" if multimodal else "gemma",
        "architectures": ["Gemma3ForConditionalGeneration" if multimodal else "GemmaForCausalLM"],
        "text_config": {"num_hidden_layers": 2, "hidden_size": 4},
        "torch_dtype": "bfloat16",
    }
    if multimodal:
        config["vision_config"] = {"num_hidden_layers": 1, "hidden_size": 3}
        config["image_token_index"] = 262144
        config["mm_tokens_per_image"] = 256
    (path / "config.json").write_text(__import__("json").dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if processor:
        (path / "processor_config.json").write_text('{"processor_class":"Gemma3Processor"}', encoding="utf-8")
        (path / "preprocessor_config.json").write_text(
            '{"processor_class":"Gemma3Processor","image_processor_type":"Gemma3ImageProcessor"}',
            encoding="utf-8",
        )


@pytest.mark.parametrize("name", ["gemma-3-4b-it", "gemma-3-12b-it"])
def test_gemma3_aliases_map_to_explicit_gemma_wrapper(name: str) -> None:
    wrapper = create_model_wrapper(_config(name=name))

    assert type(wrapper).__name__ == "Gemma3Wrapper"
    assert wrapper.expected_model_class_name() == "Gemma3ForConditionalGeneration"
    assert wrapper.expected_processor_class_name() == "Gemma3Processor"
    assert wrapper.resolve_hidden_state_index_offset() == 1


def test_text_only_gemma_config_is_rejected(tmp_path: Path) -> None:
    _write_gemma3_layout(tmp_path, multimodal=False)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "Gemma3 multimodal" in result.reason


def test_missing_gemma_multimodal_processor_blocks(tmp_path: Path) -> None:
    _write_gemma3_layout(tmp_path, processor=False)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "Gemma3Processor" in result.reason


def test_missing_installed_gemma3_transformers_class_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gemma3_layout(tmp_path)
    monkeypatch.setattr(
        "mind.models.asset_validation._missing_transformers_classes",
        lambda class_names: ["transformers.Gemma3Processor"],
    )

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "transformers.Gemma3Processor" in result.reason


def test_gemma_generation_kwargs_are_deterministic_and_explicit() -> None:
    wrapper = create_model_wrapper(_config(name="gemma-3-4b-it"))

    kwargs = wrapper.deterministic_generation_kwargs(max_new_tokens=1)

    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True
    assert kwargs["output_hidden_states"] is True


def test_gemma_hidden_state_index_offset_cannot_be_unknown() -> None:
    wrapper = create_model_wrapper(_config(name="gemma-3-4b-it", offset="unknown"))

    with pytest.raises(ValueError, match="hidden_state_index_offset"):
        wrapper.resolve_hidden_state_index_offset()


def test_gemma_load_kwargs_require_existing_local_path() -> None:
    wrapper = create_model_wrapper(_config(name="gemma-3-4b-it"))

    with pytest.raises(FileNotFoundError, match="local model path"):
        wrapper.model_load_kwargs(device="cpu")


def test_gemma_load_processor_requires_existing_local_path(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class FakeGemma3Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("from_pretrained must not run for a missing local path")

    monkeypatch.setattr("transformers.Gemma3Processor", FakeGemma3Processor)
    wrapper = create_model_wrapper(_config(name="gemma-3-4b-it"))

    with pytest.raises(FileNotFoundError, match="local model path"):
        wrapper.load_processor()
    assert called is False
