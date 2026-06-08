from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.modeling_utils import PreTrainedModel

from mind.config import ModelConfig
from mind.models.factory import create_model_wrapper
from mind.models import wrappers


def _config() -> ModelConfig:
    return ModelConfig(
        name="molmo-7b-d-0924",
        model_id="/models/molmo",
        local_path="/models/molmo",
        family="molmo",
        dtype="float16",
        trust_remote_code=True,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="molmo_single_image_raw_question_v1",
        prompt_template_text="Molmo processor receives the normalized question text unchanged with one PIL image.",
        hidden_state_index_offset=1,
    )


def test_molmo_all_tied_weights_keys_shim_is_wrapper_local(monkeypatch, tmp_path: Path) -> None:
    class LocalMolmoForCausalLM:
        def __init__(self) -> None:
            self.tie_calls = 0

        def tie_weights(self) -> str:
            self.tie_calls += 1
            return "tied"

        pass

    had_global_attr = hasattr(PreTrainedModel, "all_tied_weights_keys")
    original_global = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if had_global_attr:
        delattr(PreTrainedModel, "all_tied_weights_keys")

    monkeypatch.setattr(
        wrappers,
        "get_class_from_dynamic_module",
        lambda class_reference, local_path, local_files_only=True, trust_remote_code=True: LocalMolmoForCausalLM,
    )

    wrappers.ensure_molmo_remote_class_contract(tmp_path)

    assert LocalMolmoForCausalLM.all_tied_weights_keys == {}
    instance = LocalMolmoForCausalLM()
    assert instance.tie_weights(missing_keys=[], recompute_mapping=True) == "tied"
    assert instance.tie_calls == 1
    assert not hasattr(PreTrainedModel, "all_tied_weights_keys")

    if had_global_attr:
        setattr(PreTrainedModel, "all_tied_weights_keys", original_global)


def test_molmo_generate_from_batch_uses_generation_mixin_locally(monkeypatch, tmp_path: Path) -> None:
    class LocalMolmoForCausalLM:
        config = SimpleNamespace(use_position_ids=True)

        def tie_weights(self) -> str:
            return "tied"

        def generate_from_batch(self, *args, **kwargs):
            raise AssertionError("original generate_from_batch should be replaced")

    calls: list[dict[str, object]] = []
    had_global_generate = hasattr(PreTrainedModel, "generate")
    original_global_generate = getattr(PreTrainedModel, "generate", None)
    if had_global_generate:
        delattr(PreTrainedModel, "generate")

    def fake_generate(self, input_ids, generation_config, **kwargs):
        calls.append({"input_ids": input_ids, "generation_config": generation_config, "kwargs": kwargs})
        return "generated"

    monkeypatch.setattr(wrappers.GenerationMixin, "generate", fake_generate)
    monkeypatch.setattr(
        wrappers,
        "get_class_from_dynamic_module",
        lambda class_reference, local_path, local_files_only=True, trust_remote_code=True: LocalMolmoForCausalLM,
    )

    wrappers.ensure_molmo_remote_class_contract(tmp_path)
    result = LocalMolmoForCausalLM().generate_from_batch(
        {"input_ids": torch.tensor([[1, 2]])},
        SimpleNamespace(max_new_tokens=1, use_cache=True),
        return_dict_in_generate=True,
    )

    assert result == "generated"
    assert calls[0]["kwargs"]["attention_mask"].shape == (1, 3)
    assert calls[0]["kwargs"]["position_ids"].shape == (1, 2)
    assert not hasattr(PreTrainedModel, "generate")

    if had_global_generate:
        setattr(PreTrainedModel, "generate", original_global_generate)


def test_molmo_load_model_applies_shim_before_from_pretrained(monkeypatch, tmp_path: Path) -> None:
    wrapper = create_model_wrapper(
        ModelConfig(
            **{
                **_config().model_dump(),
                "local_path": str(tmp_path),
                "model_id": str(tmp_path),
            }
        )
    )
    calls: list[str] = []

    monkeypatch.setattr(wrappers, "require_local_model_path", lambda path: Path(path))
    monkeypatch.setattr(wrappers, "ensure_molmo_remote_class_contract", lambda path: calls.append("shim"))

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append("load")
            return SimpleNamespace(eval=lambda: "eval-model")

    monkeypatch.setattr(wrappers, "AutoModelForCausalLM", FakeAutoModelForCausalLM)

    wrapper.load_model(device="cpu")

    assert calls[:2] == ["shim", "load"]


def test_molmo_hidden_state_extraction_required_for_verification() -> None:
    wrapper = create_model_wrapper(_config())

    assert wrapper.expected_model_class_name() == "AutoModelForCausalLM"
    assert wrapper.expected_processor_class_name() == "MolmoProcessor"
    assert wrapper.resolve_hidden_state_index_offset() == 1
