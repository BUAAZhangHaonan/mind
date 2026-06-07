from __future__ import annotations

from pathlib import Path

import pytest

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import AssetModel


def _asset(path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "internvl3.5-8b",
        "local_path": str(path),
        "model_config_path": "configs/models/internvl3_5_8b_asset.yaml",
        "model_id_or_family_name": "internvl",
        "family": "internvl",
        "dtype": "float16",
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": False, "disabled_by_default": True, "disable_argument": None},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "internvl_chat_single_image_raw_question_v1",
        "prompt_template_text": "single image plus normalized question",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_hf_compatible_layout(path: Path) -> None:
    (path / "config.json").write_text(
        """
        {
          "model_type": "internvl_chat",
          "architectures": ["InternVLChatModel"],
          "auto_map": {"AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"},
          "llm_config": {"num_hidden_layers": 2, "hidden_size": 4},
          "vision_config": {"num_hidden_layers": 1, "hidden_size": 4}
        }
        """,
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "processor_config.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")


def test_hf_compatible_internvl_local_layout_is_accepted(tmp_path: Path) -> None:
    _write_hf_compatible_layout(tmp_path)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.VERIFIED
    assert result.local_loading_class_candidate == "AutoModel"


def test_custom_internvl_layout_without_processor_or_tokenizer_is_blocked(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"internvl_chat","llm_config":{"num_hidden_layers":2,"hidden_size":4}}',
        encoding="utf-8",
    )

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "processor/tokenizer" in result.reason


def test_internvl_wrapper_is_not_generic_unknown() -> None:
    wrapper = create_model_wrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="/models/internvl",
            local_path="/models/internvl",
            family="internvl",
            dtype="float16",
            trust_remote_code=True,
            attn_implementation="eager",
            deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
            thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
            policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
            prompt_template_id="internvl_prompt",
            prompt_template_text="single image plus normalized question",
            hidden_state_index_offset=1,
        )
    )

    assert type(wrapper).__name__ == "InternVLWrapper"
    assert wrapper.expected_model_class_name() == "InternVLChatModel"
    assert wrapper.expected_processor_class_name() == "InternVLLocalProcessor"
    assert wrapper.resolve_hidden_state_index_offset() == 1


def test_internvl_hidden_state_index_offset_must_be_explicit() -> None:
    wrapper = create_model_wrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="/models/internvl",
            local_path="/models/internvl",
            family="internvl",
            dtype="float16",
            trust_remote_code=True,
            hidden_state_index_offset="unknown",
        )
    )

    with pytest.raises(ValueError, match="hidden_state_index_offset"):
        wrapper.resolve_hidden_state_index_offset()
