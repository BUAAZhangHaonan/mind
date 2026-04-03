from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from mind.config import ModelConfig
from mind.models import (
    InternVLWrapper,
    LlavaOnevisionWrapper,
    LoadedModelBundle,
    MolmoWrapper,
    QwenWrapper,
    create_model_wrapper,
)


def test_create_model_wrapper_selects_expected_family() -> None:
    qwen_config = ModelConfig(
        name="qwen3-vl-8b",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen",
    )
    internvl_config = ModelConfig(
        name="internvl3.5-8b",
        model_id="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
    )

    qwen_wrapper = create_model_wrapper(qwen_config)
    internvl_wrapper = create_model_wrapper(internvl_config)

    assert isinstance(qwen_wrapper, QwenWrapper)
    assert isinstance(internvl_wrapper, InternVLWrapper)


def test_create_model_wrapper_supports_llava_onevision_and_molmo() -> None:
    llava_config = ModelConfig(
        name="llava-onevision-7b",
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        family="llava_onevision",
    )
    molmo_config = ModelConfig(
        name="molmo-7b-d-0924",
        model_id="allenai/Molmo-7B-D-0924",
        family="molmo",
    )

    llava_wrapper = create_model_wrapper(llava_config)
    molmo_wrapper = create_model_wrapper(molmo_config)

    assert isinstance(llava_wrapper, LlavaOnevisionWrapper)
    assert isinstance(molmo_wrapper, MolmoWrapper)


def test_parse_yes_no_response_strips_thinking_and_punctuation() -> None:
    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    assert wrapper.parse_yes_no_response("<think>reasoning</think>\nYes, there is.") == 1
    assert wrapper.parse_yes_no_response("No.") == 0


def test_qwen_wrapper_formats_yes_no_question_once() -> None:
    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    assert (
        wrapper.format_yes_no_question("Is there a dog in the image?")
        == "Is there a dog in the image? Respond with only one word: yes or no."
    )
    assert (
        wrapper.format_yes_no_question("Is there a dog in the image? Respond with only one word: yes or no.")
        == "Is there a dog in the image? Respond with only one word: yes or no."
    )


def test_load_bundle_uses_standard_transformers_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["processor"] = (model_id, kwargs)
            return SimpleNamespace(name="processor")

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["model"] = (model_id, kwargs)
            return SimpleNamespace(name="model")

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3-vl-8b",
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            family="qwen",
            dtype="float16",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    )

    bundle = wrapper.load_bundle(
        model_factory=FakeModelFactory,
        processor_factory=FakeProcessorFactory,
    )

    assert isinstance(bundle, LoadedModelBundle)
    assert captured["processor"] == (
        "Qwen/Qwen3-VL-8B-Instruct",
        {"trust_remote_code": True},
    )
    assert captured["model"][0] == "Qwen/Qwen3-VL-8B-Instruct"
    assert captured["model"][1]["device_map"] == "auto"
    assert captured["model"][1]["attn_implementation"] == "sdpa"
    assert captured["model"][1]["torch_dtype"] == torch.float16


def test_qwen_wrapper_decode_generation_removes_prompt_tokens() -> None:
    class FakeProcessor:
        def batch_decode(self, token_ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool):
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is True
            assert token_ids == [[42, 43]]
            return ["Yes"]

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    answer = wrapper.decode_generation(
        FakeProcessor(),
        generated_ids=torch.tensor([[10, 11, 42, 43]]),
        prompt_input_ids=torch.tensor([[10, 11]]),
    )

    assert answer == "Yes"


def test_qwen_text_prepare_inputs_uses_chat_template_and_device() -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            assert (
                messages[0]["content"][0]["text"]
                == "Answer yes or no. Respond with only one word: yes or no."
            )
            return "<prompt>"

        def __call__(self, text, return_tensors: str):
            assert text == ["<prompt>"]
            assert return_tensors == "pt"
            return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    wrapper = QwenWrapper(
        ModelConfig(
            name="qwen3.5-4b",
            model_id="Qwen/Qwen3.5-4B",
            family="qwen",
        )
    )

    batch = wrapper.prepare_inputs(
        FakeProcessor(),
        question="Answer yes or no.",
        image_path=None,
        device="cuda:1",
    )

    assert batch["device"] == "cuda:1"


def test_internvl_prepare_inputs_uses_image_and_prompt(tmp_path: Path) -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            self.calls.append(("template", messages))
            return "<vision-prompt>"

        def __call__(self, *, text, images, return_tensors: str):
            self.calls.append(("call", text, images, return_tensors))
            return FakeBatch({"input_ids": torch.tensor([[4, 5, 6]])})

    image_path = tmp_path / "demo.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    processor = FakeProcessor()
    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    batch = wrapper.prepare_inputs(
        processor,
        question="Is there a plane in the image?",
        image_path=str(image_path),
        device="cuda:0",
    )

    assert batch["device"] == "cuda:0"
    assert processor.calls[1][1] == ["<vision-prompt>"]
    assert len(processor.calls[1][2]) == 1
    assert processor.calls[0][1][0]["content"][1]["text"].endswith("Respond with only one word: yes or no.")


def test_internvl_prepare_batch_inputs_uses_padding_and_multiple_images(tmp_path: Path) -> None:
    class FakeBatch(dict):
        def to(self, device: str):
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            prompt = messages[0]["content"][1]["text"]
            self.calls.append(("template", prompt))
            return f"<vision-prompt:{prompt}>"

        def __call__(self, *, text, images, return_tensors: str, padding: bool):
            self.calls.append(("call", text, images, return_tensors, padding))
            return FakeBatch({"input_ids": torch.tensor([[4, 5, 6], [7, 8, 9]])})

    image_path_a = tmp_path / "demo-a.png"
    image_path_b = tmp_path / "demo-b.png"
    Image.new("RGB", (2, 2), color="white").save(image_path_a)
    Image.new("RGB", (2, 2), color="black").save(image_path_b)
    processor = FakeProcessor()
    wrapper = InternVLWrapper(
        ModelConfig(
            name="internvl3.5-8b",
            model_id="OpenGVLab/InternVL3_5-8B-HF",
            family="internvl",
        )
    )

    batch = wrapper.prepare_batch_inputs(
        processor,
        questions=["Is there a plane in the image?", "Is there a dog in the image?"],
        image_paths=[str(image_path_a), str(image_path_b)],
        device="cuda:2",
    )

    assert batch["device"] == "cuda:2"
    assert processor.calls[2][1] == [
        "<vision-prompt:Is there a plane in the image? Respond with only one word: yes or no.>",
        "<vision-prompt:Is there a dog in the image? Respond with only one word: yes or no.>",
    ]
    assert len(processor.calls[2][2]) == 2
    assert processor.calls[2][4] is True


def test_molmo_prepare_batch_inputs_collates_variable_length_fields(tmp_path: Path) -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []
            self.tokenizer = SimpleNamespace(name="molmo-tokenizer")

        def process(self, *, images, text: str):
            self.calls.append((images, text))
            if "plane" in text:
                return {
                    "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
                    "image_input_idx": torch.tensor([[0, 1]], dtype=torch.long),
                    "images": torch.ones((1, 2, 2), dtype=torch.float32),
                    "image_masks": torch.ones((1, 2), dtype=torch.bool),
                }
            return {
                "input_ids": torch.tensor([21, 22], dtype=torch.long),
                "image_input_idx": torch.tensor([[0]], dtype=torch.long),
                "images": torch.zeros((1, 1, 2), dtype=torch.float32),
                "image_masks": torch.ones((1, 1), dtype=torch.bool),
            }

    image_path_a = tmp_path / "demo-a.png"
    image_path_b = tmp_path / "demo-b.png"
    Image.new("RGB", (2, 2), color="white").save(image_path_a)
    Image.new("RGB", (2, 2), color="black").save(image_path_b)
    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )

    batch = wrapper.prepare_batch_inputs(
        FakeProcessor(),
        questions=["Is there a plane in the image?", "Is there a dog in the image?"],
        image_paths=[str(image_path_a), str(image_path_b)],
        device="cpu",
    )

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert batch["input_ids"][1].tolist() == [21, 22, -1]
    assert tuple(batch["image_input_idx"].shape) == (2, 1, 2)
    assert batch["image_input_idx"][1, 0].tolist() == [0, -1]
    assert tuple(batch["images"].shape) == (2, 1, 2, 2)
    assert tuple(batch["image_masks"].shape) == (2, 1, 2)


def test_molmo_generate_uses_generate_from_batch_with_tokenizer() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def generate_from_batch(self, batch, generation_config, **kwargs):
            self.calls.append((batch, generation_config, kwargs))
            return "ok"

    processor = SimpleNamespace(tokenizer=SimpleNamespace(name="molmo-tokenizer"))
    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )
    model = FakeModel()

    result = wrapper.generate(
        model,
        processor,
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        max_new_tokens=4,
    )

    assert result == "ok"
    assert model.calls[0][1].max_new_tokens == 4
    assert model.calls[0][2]["tokenizer"] is processor.tokenizer
    assert model.calls[0][2]["return_dict_in_generate"] is True
    assert model.calls[0][2]["output_scores"] is True
    assert model.calls[0][2]["output_hidden_states"] is True


def test_molmo_load_processor_uses_local_processing_modules(tmp_path: Path, monkeypatch) -> None:
    snapshot_root = tmp_path / "molmo"
    snapshot_root.mkdir()
    (snapshot_root / "image_preprocessing_molmo.py").write_text(
        "\n".join(
            [
                "class MolmoImageProcessor:",
                "    @classmethod",
                "    def from_pretrained(cls, path):",
                "        return cls()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot_root / "preprocessing_molmo.py").write_text(
        "\n".join(
            [
                "from .image_preprocessing_molmo import MolmoImageProcessor",
                "class MolmoProcessor:",
                "    def __init__(self, image_processor=None, tokenizer=None):",
                "        self.image_processor = image_processor",
                "        self.tokenizer = tokenizer",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    monkeypatch.setattr("mind.models.wrappers.snapshot_download", lambda **_: str(snapshot_root))
    monkeypatch.setattr(
        "mind.models.wrappers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    wrapper = MolmoWrapper(
        ModelConfig(
            name="molmo-7b-d-0924",
            model_id="allenai/Molmo-7B-D-0924",
            family="molmo",
        )
    )

    processor = wrapper.load_processor()

    assert type(processor).__name__ == "MolmoProcessor"
    assert processor.tokenizer.padding_side == "left"
