"""Standardized Hugging Face model wrappers for MIND."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from mind.config import ModelConfig

from .types import parse_yes_no_answer, resolve_torch_dtype


def configure_left_padding(processor: Any) -> Any:
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return processor


@dataclass
class BaseModelWrapper:
    """Base wrapper that normalizes model loading and prompt shape."""

    config: ModelConfig

    def model_load_kwargs(self, *, device: str = "cuda") -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "attn_implementation": self.config.attn_implementation,
            "torch_dtype": resolve_torch_dtype(self.config.dtype),
        }
        if device.startswith("cuda"):
            kwargs["device_map"] = "auto"
        return kwargs

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def format_yes_no_question(self, question: str) -> str:
        normalized = question.strip()
        suffix = "Respond with only one word: yes or no."
        if suffix.lower() in normalized.lower():
            return normalized
        punctuation = "" if normalized.endswith(("?", ".", "!")) else "."
        return f"{normalized}{punctuation} {suffix}"

    def prepare_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        raise NotImplementedError

    def parse_yes_no_response(self, text: str) -> int | None:
        cleaned = text.replace("<think>", " ").replace("</think>", " ").strip()
        return parse_yes_no_answer(cleaned)

    def _move_batch_to_device(self, batch: Any, device: str) -> Any:
        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    def decode_generation(
        self,
        processor: Any,
        *,
        generated_ids: Any,
        prompt_input_ids: Any,
    ) -> str:
        prompt_length = int(prompt_input_ids.shape[-1])
        continuation = generated_ids[:, prompt_length:]
        decoded = processor.batch_decode(
            continuation.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return str(decoded[0]).strip()

    def load_processor(self):
        return configure_left_padding(
            AutoProcessor.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
            )
        )

    def load_model(self, *, device: str = "cuda"):
        raise NotImplementedError


@dataclass
class LoadedModelBundle:
    processor: Any
    model: Any


class QwenWrapper(BaseModelWrapper):
    def prepare_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        del image_paths
        prompts = [
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.format_yes_no_question(question),
                            }
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in questions
        ]
        batch = processor(text=prompts, return_tensors="pt", padding=True)
        return self._move_batch_to_device(batch, device)

    def prepare_inputs(
        self,
        processor: Any,
        *,
        question: str,
        image_path: str | None,
        device: str,
    ) -> Any:
        del image_path
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.format_yes_no_question(question),
                    }
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = processor(text=[prompt], return_tensors="pt")
        return self._move_batch_to_device(batch, device)

    def load_bundle(
        self,
        *,
        model_factory: Any = AutoModelForImageTextToText,
        processor_factory: Any = AutoProcessor,
        device: str = "cuda",
    ) -> LoadedModelBundle:
        processor = processor_factory.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        model = model_factory.from_pretrained(
            self.config.model_id,
            **self.model_load_kwargs(device=device),
        )
        return LoadedModelBundle(processor=processor, model=model)


class QwenVLWrapper(QwenWrapper):
    def prepare_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        if len(questions) != len(image_paths):
            raise ValueError("questions and image_paths must have the same length.")
        prompts = []
        images = []
        for question, image_path in zip(questions, image_paths):
            if image_path is None:
                raise ValueError("QwenVLWrapper requires an image path.")
            messages = self.build_messages(
                question=self.format_yes_no_question(question),
                image_path=image_path,
            )
            prompts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            images.append(Image.open(image_path).convert("RGB"))
        batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        return self._move_batch_to_device(batch, device)

    def prepare_inputs(
        self,
        processor: Any,
        *,
        question: str,
        image_path: str | None,
        device: str,
    ) -> Any:
        messages = self.build_messages(
            question=self.format_yes_no_question(question),
            image_path=image_path,
        )
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if image_path is None:
            raise ValueError("QwenVLWrapper requires an image path.")
        image = Image.open(image_path).convert("RGB")
        batch = processor(text=[prompt], images=[image], return_tensors="pt")
        return self._move_batch_to_device(batch, device)

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        if image_path is None:
            raise ValueError("QwenVLWrapper requires an image path.")
        image = str(Path(image_path))
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def load_model(self, *, device: str = "cuda"):
        return AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            **self.model_load_kwargs(device=device),
        )


class QwenTextWrapper(QwenWrapper):
    def load_processor(self):
        return configure_left_padding(
            AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
            )
        )

    def load_bundle(
        self,
        *,
        model_factory: Any = AutoModelForCausalLM,
        processor_factory: Any = AutoTokenizer,
        device: str = "cuda",
    ) -> LoadedModelBundle:
        processor = processor_factory.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        model = model_factory.from_pretrained(
            self.config.model_id,
            **self.model_load_kwargs(device=device),
        )
        return LoadedModelBundle(processor=processor, model=model)

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        del image_path
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

    def load_model(self, *, device: str = "cuda"):
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **self.model_load_kwargs(device=device),
        )


class InternVLWrapper(QwenVLWrapper):
    """InternVL shares the same high-level image+text message shape."""
