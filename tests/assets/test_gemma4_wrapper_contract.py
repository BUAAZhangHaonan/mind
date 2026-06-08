from __future__ import annotations

import json
from pathlib import Path
import importlib.util

import pytest

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import REQUIRED_MODEL_ALIASES, AssetModel


def _config(*, offset: int | str = 1, thinking: dict[str, object] | None = None) -> ModelConfig:
    return ModelConfig(
        name="gemma-4-12b-it",
        model_id="/models/gemma-4-12B-it",
        local_path="/models/gemma-4-12B-it",
        family="gemma4",
        dtype="bfloat16",
        trust_remote_code=False,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking=thinking or {"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="gemma4_chat_single_image_raw_question_no_thinking_v1",
        prompt_template_text="Gemma 4 user chat message with one image item followed by the normalized question unchanged; chat template receives enable_thinking=false.",
        hidden_state_index_offset=offset,
    )


def _asset(path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "gemma-4-12b-it",
        "local_path": str(path),
        "model_config_path": "configs/models/gemma_4_12b_it.yaml",
        "model_id_or_family_name": "gemma4",
        "family": "gemma4",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": True, "disabled_by_default": True, "disable_argument": "enable_thinking=false"},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "gemma4_chat_single_image_raw_question_no_thinking_v1",
        "prompt_template_text": "Gemma 4 user chat message with one image item followed by the normalized question unchanged; chat template receives enable_thinking=false.",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_gemma4_layout(path: Path, *, moe: bool = False, processor: bool = True) -> None:
    config = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"num_hidden_layers": 48, "hidden_size": 4096},
        "image_token_index": 262144,
        "torch_dtype": "bfloat16",
        "supported_modalities": ["text", "image", "audio"],
    }
    if moe:
        config["num_experts"] = 8
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ enable_thinking }}"}),
        encoding="utf-8",
    )
    (path / "model-00001-of-00001.safetensors").write_text("", encoding="utf-8")
    if processor:
        (path / "processor_config.json").write_text(
            json.dumps({"processor_class": "Gemma4Processor"}),
            encoding="utf-8",
        )
        (path / "preprocessor_config.json").write_text(
            json.dumps({"processor_class": "Gemma4Processor", "image_processor_type": "Gemma4ImageProcessor"}),
            encoding="utf-8",
        )


def test_gemma4_alias_is_represented_in_registry_contract() -> None:
    assert "gemma-4-12b-it" in REQUIRED_MODEL_ALIASES
    assert len(REQUIRED_MODEL_ALIASES) == 16


def test_gemma4_alias_maps_to_explicit_non_gemma3_wrapper() -> None:
    wrapper = create_model_wrapper(_config())

    assert type(wrapper).__name__ == "Gemma4Wrapper"
    assert type(wrapper).__name__ != "Gemma3Wrapper"
    assert wrapper.expected_model_class_name() == "Gemma4ForConditionalGeneration"
    assert wrapper.expected_processor_class_name() == "Gemma4Processor"
    assert wrapper.disable_thinking_kwargs() == {"enable_thinking": False}


def test_gemma4_moe_indicators_block_loading(tmp_path: Path) -> None:
    _write_gemma4_layout(tmp_path, moe=True)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "num_experts" in result.reason


def test_gemma4_thinking_mode_must_be_disabled(tmp_path: Path) -> None:
    _write_gemma4_layout(tmp_path)
    asset = _asset(
        tmp_path,
        thinking={"supported": True, "disabled_by_default": False, "disable_argument": None},
    )

    result = audit_asset_metadata(asset)

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "thinking" in result.reason


def test_gemma4_missing_processor_blocks(tmp_path: Path) -> None:
    _write_gemma4_layout(tmp_path, processor=False)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "Gemma4Processor" in result.reason


def test_gemma4_missing_transformers_class_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gemma4_layout(tmp_path)
    monkeypatch.setattr(
        "mind.models.asset_validation._missing_transformers_classes",
        lambda class_names: ["transformers.AutoModelForMultimodalLM"],
    )

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "AutoModelForMultimodalLM" in result.reason


def test_gemma4_incomplete_safetensors_index_blocks(tmp_path: Path) -> None:
    _write_gemma4_layout(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    module_path = Path("scripts/asset_audit.py")
    spec = importlib.util.spec_from_file_location("asset_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "safetensors" in result.reason
    assert module.gemma4_local_asset_complete(tmp_path) is False


def test_gemma4_hidden_state_index_offset_cannot_be_unknown() -> None:
    wrapper = create_model_wrapper(_config(offset="unknown"))

    with pytest.raises(ValueError, match="hidden_state_index_offset"):
        wrapper.resolve_hidden_state_index_offset()
