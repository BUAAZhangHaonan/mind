from __future__ import annotations

from pathlib import Path

from mind.models.asset_validation import (
    AssetStatus,
    audit_asset_metadata,
    detect_moe_indicators,
)
from mind.models.registry import AssetModel


def _asset(tmp_path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "demo-model",
        "local_path": str(tmp_path),
        "model_config_path": "configs/models/demo.yaml",
        "model_id_or_family_name": "demo",
        "family": "qwen_vl",
        "dtype": "float16",
        "trust_remote_code": False,
        "deterministic_generation": {
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": 1,
        },
        "thinking": {
            "supported": False,
            "disabled_by_default": True,
            "disable_argument": None,
        },
        "policy": {
            "allow_moe": False,
            "allow_thinking": False,
            "allow_video_only": False,
            "allow_audio_only": False,
        },
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def test_moe_indicators_trigger_unsupported_by_policy(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"demo","num_experts":8}', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "num_experts" in result.reason
    assert detect_moe_indicators({"nested": {"experts_per_tok": 2}}) == ["nested.experts_per_tok"]


def test_thinking_unsupported_triggers_unsupported_by_policy(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"glm4v","text_config":{"num_hidden_layers":2,"hidden_size":4}}', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template":"<think>"}', encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    asset = _asset(
        tmp_path,
        family="glm4v",
        thinking={
            "supported": True,
            "disabled_by_default": False,
            "disable_argument": None,
        },
    )

    result = audit_asset_metadata(asset)

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "thinking" in result.reason


def test_missing_local_path_triggers_blocked(tmp_path: Path) -> None:
    result = audit_asset_metadata(_asset(tmp_path / "missing"))

    assert result.status == AssetStatus.BLOCKED
    assert "does not exist" in result.reason


def test_missing_processor_or_tokenizer_metadata_triggers_blocked(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"demo","num_hidden_layers":2,"hidden_size":4}', encoding="utf-8")

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "processor/tokenizer" in result.reason


def test_local_custom_internvl_asset_is_unsupported_by_wrapper(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"internvl_chat","llm_config":{"num_hidden_layers":2,"hidden_size":4}}',
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")

    result = audit_asset_metadata(_asset(tmp_path, alias="internvl3.5-8b", family="internvl"))

    assert result.status == AssetStatus.UNSUPPORTED_BY_WRAPPER
    assert "custom InternVL" in result.reason
