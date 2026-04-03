"""Standardized Hugging Face model wrappers for MIND."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

from PIL import Image
from huggingface_hub import snapshot_download
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

from mind.config import ModelConfig

from .types import parse_yes_no_answer, resolve_torch_dtype


def configure_left_padding(processor: Any) -> Any:
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return processor


def _resolve_pad_value(key: str, tensor: torch.Tensor) -> int | float | bool:
    if key in {"input_ids", "image_input_idx"}:
        return -1
    if key in {"attention_mask", "image_masks", "response_mask"}:
        return 0
    if tensor.dtype == torch.bool:
        return False
    return 0


def _pad_tensor_to_shape(
    tensor: torch.Tensor,
    *,
    target_shape: tuple[int, ...],
    pad_value: int | float | bool,
) -> torch.Tensor:
    if tuple(tensor.shape) == target_shape:
        return tensor
    padded = torch.full(target_shape, pad_value, dtype=tensor.dtype)
    slices = tuple(slice(0, dimension) for dimension in tensor.shape)
    padded[slices] = tensor
    return padded


def collate_tensor_dicts(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("items must not be empty")
    collated: dict[str, torch.Tensor] = {}
    for key in items[0]:
        tensors = [item[key] for item in items]
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError(f"Expected tensors for key '{key}'")
        rank = tensors[0].ndim
        if any(tensor.ndim != rank for tensor in tensors):
            raise ValueError(f"All tensors for key '{key}' must share the same rank")
        target_shape = tuple(
            max(int(tensor.shape[dimension]) for tensor in tensors)
            for dimension in range(rank)
        )
        pad_value = _resolve_pad_value(key, tensors[0])
        collated[key] = torch.stack(
            [
                _pad_tensor_to_shape(
                    tensor,
                    target_shape=target_shape,
                    pad_value=pad_value,
                )
                for tensor in tensors
            ],
            dim=0,
        )
    return collated


def load_local_python_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_molmo_processing_modules(model_id: str) -> tuple[Any, Any, Path]:
    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "preprocessing_molmo.py",
                "image_preprocessing_molmo.py",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "merges.txt",
                "vocab.json",
            ],
        )
    )
    package_name = f"_mind_molmo_{hashlib.sha1(model_id.encode('utf-8')).hexdigest()[:8]}"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(snapshot_path)]
        sys.modules[package_name] = package
    image_module = load_local_python_module(
        f"{package_name}.image_preprocessing_molmo",
        snapshot_path / "image_preprocessing_molmo.py",
    )
    preprocessing_module = load_local_python_module(
        f"{package_name}.preprocessing_molmo",
        snapshot_path / "preprocessing_molmo.py",
    )
    return preprocessing_module, image_module, snapshot_path


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
        if isinstance(batch, dict):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch

    def generate(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        max_new_tokens: int,
    ) -> Any:
        del processor
        return model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

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


class LlavaOnevisionWrapper(QwenVLWrapper):
    """LLaVA-OneVision uses the same image-plus-text chat template contract."""


class MolmoWrapper(BaseModelWrapper):
    def load_processor(self):
        preprocessing_module, image_module, snapshot_path = load_molmo_processing_modules(
            self.config.model_id
        )
        image_processor = image_module.MolmoImageProcessor.from_pretrained(snapshot_path)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        processor = preprocessing_module.MolmoProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        return configure_left_padding(processor)

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        del image_path
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

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
        processed_inputs: list[dict[str, torch.Tensor]] = []
        for question, image_path in zip(questions, image_paths):
            if image_path is None:
                raise ValueError("MolmoWrapper requires an image path.")
            image = Image.open(image_path).convert("RGB")
            processed_inputs.append(
                processor.process(
                    images=[image],
                    text=self.format_yes_no_question(question),
                )
            )
        return self._move_batch_to_device(collate_tensor_dicts(processed_inputs), device)

    def prepare_inputs(
        self,
        processor: Any,
        *,
        question: str,
        image_path: str | None,
        device: str,
    ) -> Any:
        return self.prepare_batch_inputs(
            processor,
            questions=[question],
            image_paths=[image_path],
            device=device,
        )

    def decode_generation(
        self,
        processor: Any,
        *,
        generated_ids: Any,
        prompt_input_ids: Any,
    ) -> str:
        prompt_length = int(prompt_input_ids.shape[-1])
        continuation = generated_ids[:, prompt_length:]
        decoded = processor.tokenizer.batch_decode(
            continuation.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return str(decoded[0]).strip()

    def generate(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        max_new_tokens: int,
    ) -> Any:
        return model.generate_from_batch(
            model_inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, use_cache=True),
            tokenizer=processor.tokenizer,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

    def load_model(self, *, device: str = "cuda"):
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **self.model_load_kwargs(device=device),
        )
