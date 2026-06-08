from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_module():
    path = Path("tmp/asset_repair/run_llava_v15_tmp_smoke.py")
    spec = importlib.util.spec_from_file_location("run_llava_v15_tmp_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeProcessor:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        assert content[0] == {"type": "image"}
        assert content[1]["type"] == "text"
        return "<image>\nUSER: " + content[1]["text"] + " ASSISTANT:"


def test_tmp_smoke_script_defaults_to_dry_run() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])
    assert args.execute is False
    assert args.dry_run is True


def test_prompt_uses_image_and_question_without_rewriting() -> None:
    module = _load_module()
    prompt = module.build_prompt(FakeProcessor(), "Is there a snowboard in the image?")
    assert "<image>" in prompt
    assert "Is there a snowboard in the image?" in prompt
    assert "Respond with only one word" not in prompt


def test_select_full_layer_vectors_requires_explicit_offset() -> None:
    module = _load_module()
    hidden_states = tuple(torch.full((1, 4, 3), float(index)) for index in range(4))

    vectors, selected_indices = module.select_full_layer_vectors(
        hidden_states,
        token_index=2,
        total_layers=3,
        hidden_dim=3,
    )

    assert vectors.shape == (3, 3)
    assert selected_indices == [1, 2, 3]
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_select_full_layer_vectors_fails_unknown_count() -> None:
    module = _load_module()
    hidden_states = tuple(torch.zeros((1, 4, 3)) for _ in range(2))

    with pytest.raises(ValueError, match="hidden_state_index_offset"):
        module.select_full_layer_vectors(hidden_states, token_index=2, total_layers=3, hidden_dim=3)


def test_sidecar_metadata_is_validation_ready() -> None:
    module = _load_module()
    sidecar = module.build_sidecar_metadata(
        dataset_name="pope",
        subset="popular",
        total_layers=32,
        hidden_dim=4096,
        hidden_state_count=33,
        selected_indices=list(range(1, 33)),
        token_index=10,
        processor_class="LlavaProcessor",
        model_class="LlavaForConditionalGeneration",
        local_path="/tmp/llava",
    )

    assert sidecar["model_alias"] == "llava-v1.5-7b"
    assert sidecar["model_family"] == "llava_v15_tmp"
    assert sidecar["wrapper_class"] == "TmpLlavaV15SmokeRunner"
    assert sidecar["processor_class"] == "LlavaProcessor"
    assert sidecar["model_class"] == "LlavaForConditionalGeneration"
    assert sidecar["hidden_state_index_offset"] == 1
    assert sidecar["selected_layers"] == list(range(32))
    assert sidecar["deterministic_generation_kwargs"]["max_new_tokens"] == 1
    assert sidecar["deterministic_generation_kwargs"]["do_sample"] is False
    assert sidecar["deterministic_generation_kwargs"]["temperature"] == 0
