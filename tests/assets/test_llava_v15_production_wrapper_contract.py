from __future__ import annotations

from mind.config import ModelConfig
from mind.models.factory import create_model_wrapper
from mind.models.registry import load_asset_registry


def _config(offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name="llava-v1.5-7b",
        model_id="/home/team/lvshuyang/Models/llava-1.5-7b-hf",
        local_path="/home/team/lvshuyang/Models/llava-1.5-7b-hf",
        family="llava_v15",
        dtype="float16",
        trust_remote_code=False,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="llava_v15_single_image_raw_question_v1",
        prompt_template_text="LLaVA-v1.5 USER image prompt with normalized question unchanged.",
        hidden_state_index_offset=offset,
    )


def test_llava_v15_maps_to_explicit_wrapper() -> None:
    wrapper = create_model_wrapper(_config())

    assert type(wrapper).__name__ == "LlavaV15Wrapper"
    assert type(wrapper).__name__ != "LlavaOnevisionWrapper"
    assert wrapper.expected_processor_class_name() == "LlavaProcessor"
    assert wrapper.expected_model_class_name() == "LlavaForConditionalGeneration"
    assert wrapper.resolve_hidden_state_index_offset() == 1


def test_llava_v15_registry_uses_complete_hf_path() -> None:
    registry = load_asset_registry("configs/assets/model_assets.yaml")
    asset = next(model for model in registry.models if model.alias == "llava-v1.5-7b")

    assert asset.local_path == "/home/team/lvshuyang/Models/llava-1.5-7b-hf"
    assert asset.hidden_state_index_offset == 1
    assert asset.family == "llava_v15"


def test_llava_v15_sidecar_policy_is_explicit() -> None:
    wrapper = create_model_wrapper(_config())
    metadata = wrapper.production_sidecar_metadata()

    assert metadata["hf_complete_asset_path"] == "/home/team/lvshuyang/Models/llava-1.5-7b-hf"
    assert metadata["vision_tower_status"] == "local_hf_checkpoint_contains_vision_tower"
    assert metadata["image_token_prompt_policy"] == "llava_v15_user_image_newline_question"
    assert metadata["copied_metadata_from_onevision"] is False
