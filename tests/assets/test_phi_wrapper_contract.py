from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import AssetModel


def _config(*, name: str, family: str, offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_id=f"/models/{name}",
        local_path=f"/models/{name}",
        family=family,
        dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id=f"{family}_single_image_raw_question_v1",
        prompt_template_text="Phi image-text prompt inserts one image token and the normalized question unchanged.",
        hidden_state_index_offset=offset,
    )


def _asset(path: Path, *, family: str, alias: str | None = None, **overrides: object) -> AssetModel:
    payload = {
        "alias": alias or ("phi-4-multimodal-instruct" if family == "phi4mm" else "phi-3.5-vision-instruct"),
        "local_path": str(path),
        "model_config_path": f"configs/models/{family}.yaml",
        "model_id_or_family_name": family,
        "family": family,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": False, "disabled_by_default": True, "disable_argument": None},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": f"{family}_single_image_raw_question_v1",
        "prompt_template_text": "Phi image-text prompt inserts one image token and the normalized question unchanged.",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_phi_layout(
    path: Path,
    *,
    family: str,
    image_processor: bool = True,
    audio_required_only: bool = False,
) -> None:
    architecture = "Phi4MMForCausalLM" if family == "phi4mm" else "Phi3VForCausalLM"
    processor = "Phi4MMProcessor" if family == "phi4mm" else "Phi3VProcessor"
    image_processor_type = "Phi4MMImageProcessor" if family == "phi4mm" else "Phi3VImageProcessor"
    config = {
        "model_type": family,
        "architectures": [architecture],
        "auto_map": {"AutoModelForCausalLM": f"modeling_{family}.{architecture}"},
        "num_hidden_layers": 2,
        "hidden_size": 4,
        "torch_dtype": "bfloat16",
    }
    if family == "phi3_v":
        config["img_processor"] = {"image_dim_out": 4}
    else:
        config["embd_layer"] = {
            "image_embd_layer": {"embedding_cls": "image"},
            "audio_embd_layer": {"embedding_cls": "audio"},
        }
    if audio_required_only:
        config.pop("img_processor", None)
        config["embd_layer"] = {"audio_embd_layer": {"embedding_cls": "audio"}}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "processor_config.json").write_text(json.dumps({"processor_class": processor}), encoding="utf-8")
    if image_processor:
        (path / "preprocessor_config.json").write_text(
            json.dumps({"processor_class": processor, "image_processor_type": image_processor_type}),
            encoding="utf-8",
        )


def test_phi35_vision_alias_maps_to_explicit_phi_wrapper() -> None:
    wrapper = create_model_wrapper(_config(name="phi-3.5-vision-instruct", family="phi3_v"))

    assert type(wrapper).__name__ == "Phi35VisionWrapper"
    assert wrapper.expected_model_class_name() == "Phi3VForCausalLM"
    assert wrapper.expected_processor_class_name() == "Phi3VProcessor"
    assert wrapper.resolve_hidden_state_index_offset() == 1


def test_phi4_multimodal_alias_maps_to_explicit_phi_wrapper() -> None:
    wrapper = create_model_wrapper(_config(name="phi-4-multimodal-instruct", family="phi4mm"))

    assert type(wrapper).__name__ == "Phi4MultimodalWrapper"
    assert wrapper.expected_model_class_name() == "Phi4MMForCausalLM"
    assert wrapper.expected_processor_class_name() == "Phi4MMProcessor"
    assert wrapper.resolve_hidden_state_index_offset() == 1


def test_audio_only_or_audio_required_phi_blocks_image_text_smoke(tmp_path: Path) -> None:
    _write_phi_layout(tmp_path, family="phi4mm", audio_required_only=True)

    result = audit_asset_metadata(_asset(tmp_path, family="phi4mm"))

    assert result.status == AssetStatus.UNSUPPORTED_BY_POLICY
    assert "image-text" in result.reason


def test_missing_phi_image_processor_blocks(tmp_path: Path) -> None:
    _write_phi_layout(tmp_path, family="phi3_v", image_processor=False)

    result = audit_asset_metadata(_asset(tmp_path, family="phi3_v"))

    assert result.status == AssetStatus.BLOCKED
    assert "image processor" in result.reason


@pytest.mark.parametrize("family", ["phi3_v", "phi4mm"])
def test_phi_generation_kwargs_are_deterministic_and_explicit(family: str) -> None:
    name = "phi-4-multimodal-instruct" if family == "phi4mm" else "phi-3.5-vision-instruct"
    wrapper = create_model_wrapper(_config(name=name, family=family))

    kwargs = wrapper.deterministic_generation_kwargs(max_new_tokens=1)

    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True
    assert kwargs["output_hidden_states"] is True
    assert kwargs["use_cache"] is False


def test_phi_hidden_state_index_offset_cannot_be_unknown() -> None:
    wrapper = create_model_wrapper(_config(name="phi-3.5-vision-instruct", family="phi3_v", offset="unknown"))

    with pytest.raises(ValueError, match="hidden_state_index_offset"):
        wrapper.resolve_hidden_state_index_offset()
