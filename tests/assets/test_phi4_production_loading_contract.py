from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from mind.config import ModelConfig
from mind.models.factory import create_model_wrapper
from mind.models.wrappers import Phi4MultimodalWrapper


def _config() -> ModelConfig:
    return ModelConfig(
        name="phi-4-multimodal-instruct",
        model_id="/home/team/lvshuyang/Models/Phi-4-multimodal-instruct",
        local_path="/home/team/lvshuyang/Models/Phi-4-multimodal-instruct",
        family="phi4mm",
        dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="phi4mm_single_image_raw_question_v1",
        prompt_template_text="Phi4 image-text prompt inserts one image token and the normalized question unchanged.",
        hidden_state_index_offset=1,
    )


def test_phi4_disables_flash_attention_and_records_safe_loading_policy() -> None:
    wrapper = create_model_wrapper(_config())

    kwargs = wrapper.model_load_kwargs(device="cuda:0")
    metadata = wrapper.production_sidecar_metadata()

    assert type(wrapper).__name__ == "Phi4MultimodalWrapper"
    assert kwargs["attn_implementation"] in {"eager", "sdpa"}
    assert "_attn_implementation" not in kwargs
    assert kwargs["low_cpu_mem_usage"] is False
    assert kwargs["device_map"] is None
    assert metadata["disabled_flash_attention_2"] is True
    assert metadata["attn_implementation_effective"] in {"eager", "sdpa"}
    assert metadata["low_cpu_mem_usage"] is False
    assert metadata["device_map_policy"] == "manual_to_single_device"


def test_phi4_requires_no_meta_tensor_confirmation_for_verification() -> None:
    wrapper = create_model_wrapper(_config())

    with pytest.raises(ValueError, match="meta"):
        wrapper.validate_no_meta_tensors_after_load(object())


def test_phi4_loading_policy_comments_are_present() -> None:
    source = inspect.getsource(Phi4MultimodalWrapper)

    assert "Flash Attention 2" in source
    assert "low_cpu_mem_usage=False" in source
    assert "meta tensors" in source
    assert "asset-loading policy" in source


def test_phi4_records_local_flash_attention_override_even_when_registry_is_eager() -> None:
    config = _config().model_copy(update={"attn_implementation": "eager"})
    wrapper = Phi4MultimodalWrapper(config)

    local_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    original_attn = str(
        getattr(local_config, "_attn_implementation", None)
        or getattr(local_config, "attn_implementation", None)
        or wrapper.config.attn_implementation
        or "unknown"
    )
    wrapper._phi4_original_attn_implementation = original_attn
    wrapper._phi4_disabled_flash_attention_2 = original_attn == "flash_attention_2"

    metadata = wrapper.production_sidecar_metadata()

    assert metadata["attn_implementation_effective"] == "eager"
    assert metadata["attn_implementation_original"] == "flash_attention_2"
    assert metadata["disabled_flash_attention_2"] is True
