from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mind.config import ModelConfig
from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.factory import create_model_wrapper
from mind.models.registry import AssetModel


MINICPM_ALIASES = ("minicpm-v-2_6", "minicpm-v-4_5")


def _config(*, name: str, offset: int | str = 1) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_id=f"/models/{name}",
        local_path=f"/models/{name}",
        family="minicpmv",
        dtype="float16",
        trust_remote_code=True,
        attn_implementation="eager",
        deterministic_generation={"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        policy={"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        prompt_template_id="minicpmv_single_image_raw_question_v1",
        prompt_template_text="single image plus normalized question",
        hidden_state_index_offset=offset,
    )


def _asset(path: Path, *, alias: str = "minicpm-v-2_6", **overrides: object) -> AssetModel:
    payload = {
        "alias": alias,
        "local_path": str(path),
        "model_config_path": f"configs/models/{alias.replace('-', '_')}.yaml",
        "model_id_or_family_name": "minicpmv",
        "family": "minicpmv",
        "dtype": "float16",
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": False, "disabled_by_default": True, "disable_argument": None},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "minicpmv_single_image_raw_question_v1",
        "prompt_template_text": "single image plus normalized question",
        "hidden_state_index_offset": 1,
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_minicpm_layout(
    path: Path,
    *,
    alias: str = "minicpm-v-2_6",
    processor: bool = True,
    image_processor: bool = True,
    generation_only_chat: bool = False,
) -> None:
    architecture = "MiniCPMV4_5ForCausalLM" if alias == "minicpm-v-4_5" else "MiniCPMVForCausalLM"
    processor_class = "MiniCPMVProcessor"
    config = {
        "model_type": "minicpmv",
        "architectures": [architecture],
        "auto_map": {"AutoModel": f"modeling_minicpmv.{architecture}"},
        "llm_config": {"num_hidden_layers": 2, "hidden_size": 4},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 4},
        "torch_dtype": "float16",
    }
    if generation_only_chat:
        config["auto_map"] = {"AutoModel": "modeling_minicpmv.GenerationOnlyChatModel"}
        config["custom_chat_api"] = {"method": "chat", "returns_hidden_states": False}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if processor:
        (path / "processor_config.json").write_text(
            json.dumps({"processor_class": processor_class}),
            encoding="utf-8",
        )
    if image_processor:
        (path / "preprocessor_config.json").write_text(
            json.dumps({"processor_class": processor_class, "image_processor_type": "MiniCPMVImageProcessor"}),
            encoding="utf-8",
        )


@pytest.mark.parametrize("alias", MINICPM_ALIASES)
def test_minicpm_aliases_map_to_explicit_wrapper_or_unsupported_by_wrapper(tmp_path: Path, alias: str) -> None:
    _write_minicpm_layout(tmp_path, alias=alias)

    try:
        wrapper = create_model_wrapper(_config(name=alias))
    except ValueError:
        result = audit_asset_metadata(_asset(tmp_path, alias=alias))
        assert result.status == AssetStatus.UNSUPPORTED_BY_WRAPPER
        assert "wrapper" in result.reason.lower()
    else:
        assert "minicpm" in type(wrapper).__name__.lower()
        assert "minicpm" in wrapper.expected_model_class_name().lower()
        assert "minicpm" in wrapper.expected_processor_class_name().lower()


def test_minicpm_generation_only_custom_chat_without_hidden_state_access_is_insufficient(tmp_path: Path) -> None:
    _write_minicpm_layout(tmp_path, generation_only_chat=True)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status in {AssetStatus.BLOCKED, AssetStatus.FAILED_VALIDATION, AssetStatus.UNSUPPORTED_BY_WRAPPER}
    assert "hidden" in result.reason.lower()


def test_missing_minicpm_image_processor_blocks_verification(tmp_path: Path) -> None:
    _write_minicpm_layout(tmp_path, image_processor=False)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "image processor" in result.reason.lower()


@pytest.mark.parametrize("alias", MINICPM_ALIASES)
def test_minicpm_generation_kwargs_are_deterministic_and_explicit(tmp_path: Path, alias: str) -> None:
    try:
        wrapper = create_model_wrapper(_config(name=alias))
    except ValueError:
        _write_minicpm_layout(tmp_path, alias=alias)
        result = audit_asset_metadata(_asset(tmp_path, alias=alias))
        assert result.status == AssetStatus.UNSUPPORTED_BY_WRAPPER
        assert result.reason
        return

    kwargs = wrapper.deterministic_generation_kwargs(max_new_tokens=1)

    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] == 0
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True
    assert kwargs["output_hidden_states"] is True


def test_minicpm_hidden_state_index_offset_cannot_be_unknown(tmp_path: Path) -> None:
    try:
        wrapper = create_model_wrapper(_config(name="minicpm-v-2_6", offset="unknown"))
    except ValueError:
        _write_minicpm_layout(tmp_path)
        result = audit_asset_metadata(_asset(tmp_path, hidden_state_index_offset="unknown"))
        assert result.status == AssetStatus.FAILED_VALIDATION
        assert "hidden_state_index_offset" in result.reason
    else:
        with pytest.raises(ValueError, match="hidden_state_index_offset"):
            wrapper.resolve_hidden_state_index_offset()


def test_minicpm_wrapper_attaches_tokenizer_special_id_contract() -> None:
    wrapper = create_model_wrapper(_config(name="minicpm-v-2_6"))

    class BackendTokenizer:
        bos_token_id = 101
        eos_token_id = 102
        unk_token_id = 103

        def convert_tokens_to_ids(self, token: str) -> int:
            return {
                "<image>": 201,
                "</image>": 202,
                "<slice>": 203,
                "</slice>": 204,
                "<image_id>": 205,
                "</image_id>": 206,
                "\n": 207,
            }[token]

    tokenizer = BackendTokenizer()

    wrapper.configure_minicpm_tokenizer(tokenizer)

    assert tokenizer.im_start == "<image>"
    assert tokenizer.im_start_id == 201
    assert tokenizer.im_end_id == 202
    assert tokenizer.slice_start_id == 203
    assert tokenizer.slice_end_id == 204
    assert tokenizer.im_id_start_id == 205
    assert tokenizer.im_id_end_id == 206
    assert tokenizer.newline_id == 207
    assert tokenizer.bos_id == 101
    assert tokenizer.eos_id == 102
    assert tokenizer.unk_id == 103


def test_minicpm_wrapper_patches_remote_tied_weight_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wrapper = create_model_wrapper(_config(name="minicpm-v-2_6"))

    class RemoteMiniCPMV:
        pass

    def fake_get_class(class_reference: str, model_path: str, **kwargs: object) -> type:
        assert class_reference == "modeling_minicpmv.MiniCPMV"
        assert model_path == str(tmp_path)
        assert kwargs["local_files_only"] is True
        return RemoteMiniCPMV

    monkeypatch.setattr("mind.models.wrappers.get_class_from_dynamic_module", fake_get_class, raising=False)

    wrapper.ensure_minicpm_remote_class_contract(tmp_path)

    assert RemoteMiniCPMV.all_tied_weights_keys == {}


def test_minicpm_generate_passes_max_new_tokens_once() -> None:
    wrapper = create_model_wrapper(_config(name="minicpm-v-2_6"))
    recorded: dict[str, object] = {}

    class FakeModel:
        def generate(self, *, max_new_tokens: int, **kwargs: object) -> object:
            recorded["max_new_tokens"] = max_new_tokens
            recorded["kwargs"] = kwargs
            return object()

    class FakeProcessor:
        tokenizer = object()

    model_inputs = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        "pixel_values": [[torch.zeros((1, 3, 2, 2))]],
        "tgt_sizes": [[torch.tensor([[1, 1]])]],
        "image_bound": [torch.tensor([[1, 2]])],
    }

    wrapper.generate(FakeModel(), FakeProcessor(), model_inputs=model_inputs, max_new_tokens=1)

    assert recorded["max_new_tokens"] == 1
    assert "max_new_tokens" not in recorded["kwargs"]
    assert recorded["kwargs"]["do_sample"] is False


def test_minicpm_decode_generation_passes_tensor_to_processor() -> None:
    wrapper = create_model_wrapper(_config(name="minicpm-v-2_6"))
    observed: dict[str, object] = {}

    class FakeProcessor:
        def batch_decode(self, ids: object, **kwargs: object) -> list[str]:
            observed["is_tensor"] = isinstance(ids, torch.Tensor)
            observed["shape"] = tuple(ids.shape)
            observed["kwargs"] = kwargs
            return ["yes"]

    text = wrapper.decode_generation(
        FakeProcessor(),
        generated_ids=torch.tensor([[101, 102, 103, 104]]),
        prompt_input_ids=torch.tensor([[101, 102, 103]]),
    )

    assert text == "yes"
    assert observed["is_tensor"] is True
    assert observed["shape"] == (1, 1)
    assert observed["kwargs"]["skip_special_tokens"] is True
