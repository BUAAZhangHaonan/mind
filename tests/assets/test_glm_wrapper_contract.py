from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import AssetModel


def _config(*, offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name="glm-4.6v-flash",
        model_id="/models/glm-4.6v-flash",
        local_path="/models/glm-4.6v-flash",
        family="glm4v",
        dtype="float16",
        trust_remote_code=False,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="glm4v_chat_single_image_raw_question_no_thinking_v1",
        prompt_template_text="single image plus normalized question with enable_thinking=false",
        hidden_state_index_offset=offset,
    )


def _asset(path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "glm-4.6v-flash",
        "local_path": str(path),
        "model_config_path": "configs/models/glm_4_6v_flash.yaml",
        "model_id_or_family_name": "glm4v",
        "family": "glm4v",
        "dtype": "float16",
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "glm4v_chat_single_image_raw_question_no_thinking_v1",
        "prompt_template_text": "single image plus normalized question with enable_thinking=false",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_glm_layout(
    path: Path,
    *,
    moe: bool = False,
    thinking_template: bool = False,
    processor: bool = True,
) -> None:
    text_config: dict[str, object] = {"num_hidden_layers": 2, "hidden_size": 4}
    if moe:
        text_config["num_experts"] = 8
    config = {
        "model_type": "glm4v",
        "architectures": ["GLM4VForConditionalGeneration"],
        "text_config": text_config,
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 4},
        "image_token_index": 151343,
        "torch_dtype": "float16",
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenizer_config = {"chat_template": "{{ messages }}"}
    if thinking_template:
        tokenizer_config["chat_template"] = "{{ reasoning_content }} <think>{{ messages }}</think>"
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    if processor:
        (path / "processor_config.json").write_text(
            json.dumps({"processor_class": "GLM4VProcessor"}),
            encoding="utf-8",
        )
        (path / "preprocessor_config.json").write_text(
            json.dumps({"processor_class": "GLM4VProcessor", "image_processor_type": "GLM4VImageProcessor"}),
            encoding="utf-8",
        )


def test_glm_alias_maps_to_explicit_wrapper_or_policy_block(tmp_path: Path) -> None:
    _write_glm_layout(tmp_path)

    try:
        wrapper = create_model_wrapper(_config())
    except ValueError:
        result = audit_asset_metadata(_asset(tmp_path))
        assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
        assert "glm" in result.reason.lower()
    else:
        assert "glm" in type(wrapper).__name__.lower()
        assert "glm" in wrapper.expected_model_class_name().lower()
        assert "glm" in wrapper.expected_processor_class_name().lower()
        assert wrapper.resolve_hidden_state_index_offset() == 1


def test_glm_moe_indicators_block_loading_by_policy(tmp_path: Path) -> None:
    _write_glm_layout(tmp_path, moe=True)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "num_experts" in result.reason


def test_glm_thinking_or_reasoning_cannot_remain_unresolved(tmp_path: Path) -> None:
    _write_glm_layout(tmp_path, thinking_template=True)
    asset = _asset(
        tmp_path,
        thinking={"supported": "unknown", "disabled_by_default": "unknown", "disable_argument": None},
    )

    result = audit_asset_metadata(asset)

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "thinking" in result.reason.lower() or "reasoning" in result.reason.lower()


def test_glm_generation_kwargs_are_deterministic_and_explicit() -> None:
    try:
        wrapper = create_model_wrapper(_config())
    except ValueError as error:
        pytest.fail(f"Batch3 GLM contract needs an explicit wrapper with deterministic kwargs: {error}")

    kwargs = wrapper.deterministic_generation_kwargs(max_new_tokens=1)

    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True
    assert kwargs["output_hidden_states"] is True


def test_glm_hidden_state_index_offset_cannot_be_unknown(tmp_path: Path) -> None:
    try:
        wrapper = create_model_wrapper(_config(offset="unknown"))
    except ValueError:
        _write_glm_layout(tmp_path)
        result = audit_asset_metadata(_asset(tmp_path, hidden_state_index_offset="unknown"))
        assert result.status == AssetStatus.FAILED_VALIDATION
        assert "hidden_state_index_offset" in result.reason
    else:
        with pytest.raises(ValueError, match="hidden_state_index_offset"):
            wrapper.resolve_hidden_state_index_offset()


def test_glm_processor_and_model_classes_are_recorded_when_verified(tmp_path: Path) -> None:
    _write_glm_layout(tmp_path)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status in {AssetStatus.VERIFIED, AssetStatus.UNSUPPORTED_BY_POLICY}
    if result.status == AssetStatus.VERIFIED:
        assert result.local_loading_class_candidate not in {"generic_unknown", "unknown", "unsupported"}
        assert result.image_processor_candidate != "unknown"
