"""Standardized Hugging Face model wrappers for MIND."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from mind.config import ModelConfig

from .types import parse_yes_no_answer, resolve_torch_dtype


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

    def parse_yes_no_response(self, text: str) -> int | None:
        cleaned = text.replace("<think>", " ").replace("</think>", " ").strip()
        return parse_yes_no_answer(cleaned)

    def load_processor(self):
        return AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )

    def load_model(self, *, device: str = "cuda"):
        raise NotImplementedError


@dataclass
class LoadedModelBundle:
    processor: Any
    model: Any


class QwenWrapper(BaseModelWrapper):
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
