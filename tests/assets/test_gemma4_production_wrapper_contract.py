from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import AssetModel


def _config(*, family: str = "gemma4_unified", offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name="gemma-4-12b-it",
        model_id="/home/team/lvshuyang/Models/gemma-4-12B-it",
        local_path="/home/team/lvshuyang/Models/gemma-4-12B-it",
        family=family,
        dtype="bfloat16",
        trust_remote_code=False,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="gemma4_unified_single_image_raw_question_no_thinking_v1",
        prompt_template_text="Gemma4 Unified chat template with one image item, raw question text, and enable_thinking=False.",
        hidden_state_index_offset=offset,
    )


def _asset(path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "gemma-4-12b-it",
        "local_path": str(path),
        "hf_model_id": "google/gemma-4-12B-it",
        "model_config_path": "configs/models/gemma_4_12b_it.yaml",
        "model_id_or_family_name": "gemma4_unified",
        "family": "gemma4_unified",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "gemma4_unified_single_image_raw_question_no_thinking_v1",
        "prompt_template_text": "Gemma4 Unified chat template with one image item, raw question text, and enable_thinking=False.",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_gemma4_layout(path: Path, *, moe_variant: bool = False) -> None:
    text_config: dict[str, object] = {"num_hidden_layers": 48, "hidden_size": 3840}
    if moe_variant:
        text_config.update({"num_experts": 8, "top_k_experts": 2, "enable_moe_block": True})
    else:
        text_config.update({"num_experts": None, "top_k_experts": None, "enable_moe_block": False})
    config = {
        "model_type": "gemma4_unified",
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "text_config": text_config,
        "image_token_id": 258880,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(json.dumps({"processor_class": "Gemma4UnifiedProcessor"}), encoding="utf-8")
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4UnifiedProcessor",
                "image_processor": {"image_processor_type": "Gemma4UnifiedImageProcessor"},
            }
        ),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"synthetic")


def test_gemma4_maps_to_explicit_unified_wrapper() -> None:
    wrapper = create_model_wrapper(_config())

    assert type(wrapper).__name__ == "Gemma4UnifiedWrapper"
    assert type(wrapper).__name__ != "Gemma3Wrapper"
    assert wrapper.expected_processor_class_name() == "Gemma4UnifiedProcessor"
    assert wrapper.expected_model_class_name() in {"AutoModelForMultimodalLM", "Gemma4UnifiedForConditionalGeneration"}
    assert wrapper.disable_thinking_kwargs() == {"enable_thinking": False}
    assert wrapper.prompt_template_id() == "gemma4_unified_single_image_raw_question_no_thinking_v1"
    assert wrapper.requires_image_sensitivity_canary() is True
    assert wrapper.has_separate_vision_encoder() is False


def test_gemma4_nested_inactive_expert_fields_do_not_trigger_moe_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gemma4_layout(tmp_path, moe_variant=False)
    monkeypatch.setattr("mind.models.asset_validation._missing_transformers_classes", lambda class_names: [])

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.VERIFIED
    assert result.moe_indicators == []
    assert result.as_dict()["moe_policy_decision"] == "non_moe"
    assert "text_config.num_experts" in result.as_dict()["moe_indicators_ignored_with_reason"]


def test_gemma4_explicit_moe_variant_still_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gemma4_layout(tmp_path, moe_variant=True)
    monkeypatch.setattr("mind.models.asset_validation._missing_transformers_classes", lambda class_names: [])

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "MoE" in result.reason


def test_gemma4_thinking_must_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gemma4_layout(tmp_path, moe_variant=False)
    monkeypatch.setattr("mind.models.asset_validation._missing_transformers_classes", lambda class_names: [])

    result = audit_asset_metadata(_asset(tmp_path, thinking={"supported": True, "disabled_by_default": False, "disable_argument": None}))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "thinking" in result.reason
