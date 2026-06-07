"""Standardized Hugging Face model wrappers for MIND."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.modeling_utils import PreTrainedModel

from mind.config import ModelConfig

from .types import parse_yes_no_answer, resolve_torch_dtype


def _model_inputs_get(model_inputs: Any, key: str, default: Any = None) -> Any:
    getter = getattr(model_inputs, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return model_inputs[key]
    except (KeyError, TypeError, IndexError):
        return default


def _model_inputs_has(model_inputs: Any, key: str) -> bool:
    try:
        return key in model_inputs
    except TypeError:
        return False


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


def require_local_model_path(model_id_or_path: str) -> Path:
    local_model_path = Path(model_id_or_path).expanduser()
    if not local_model_path.is_dir():
        raise FileNotFoundError(f"local model path does not exist or is not a directory: {local_model_path}")
    return local_model_path


def load_molmo_processing_modules(model_id: str) -> tuple[Any, Any, Path]:
    snapshot_path = require_local_model_path(model_id)
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


def load_internvl_conversation_module(model_id: str) -> Any:
    local_model_path = Path(model_id).expanduser()
    if not local_model_path.exists():
        raise FileNotFoundError(f"InternVL local model path does not exist: {local_model_path}")
    conversation_path = local_model_path / "conversation.py"
    if not conversation_path.is_file():
        raise FileNotFoundError(f"InternVL conversation.py is missing: {conversation_path}")
    package_name = f"_mind_internvl_{hashlib.sha1(str(local_model_path).encode('utf-8')).hexdigest()[:8]}"
    return load_local_python_module(f"{package_name}.conversation", conversation_path)


def _model_inputs_batch_size(model_inputs: Any) -> int:
    input_ids = _model_inputs_get(model_inputs, "input_ids")
    if hasattr(input_ids, "shape") and len(input_ids.shape) > 0:
        return int(input_ids.shape[0])
    return 1


def _module_dtype(module: Any, fallback: torch.dtype = torch.float32) -> torch.dtype:
    if hasattr(module, "dtype") and isinstance(module.dtype, torch.dtype):
        return module.dtype
    try:
        return next(module.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return fallback


def _lookup_config_value(config: Any, path: Sequence[str]) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def _resolve_positive_config_int(
    config: Any,
    paths: Sequence[Sequence[str]],
    *,
    label: str,
) -> int:
    for path in paths:
        value = _lookup_config_value(config, path)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            continue
        if resolved > 0:
            return resolved
    checked = ", ".join(".".join(path) for path in paths)
    raise ValueError(f"Could not resolve {label}; checked {checked}")


def _normalize_molmo_past_key_values(
    past_key_values: Any,
) -> Any:
    if past_key_values is None:
        return None
    try:
        entries = list(past_key_values)
    except TypeError:
        return past_key_values
    if not entries:
        return None
    if all(
        entry is None
        or (
            isinstance(entry, tuple)
            and all(value is None for value in entry)
        )
        for entry in entries
    ):
        return None
    return past_key_values


def _pil_bicubic() -> Any:
    resampling = getattr(Image, "Resampling", None)
    return resampling.BICUBIC if resampling is not None else Image.BICUBIC


@dataclass
class BaseModelWrapper:
    """Base wrapper that normalizes model loading and prompt shape."""

    config: ModelConfig
    default_prompt_template_id: ClassVar[str] = "single_image_raw_question_v1"

    def model_load_kwargs(self, *, device: str = "cuda") -> dict[str, Any]:
        require_local_model_path(self.model_id_or_path())
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": resolve_torch_dtype(self.config.dtype),
            "local_files_only": True,
        }
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation
        if device.startswith("cuda"):
            kwargs["device_map"] = {"": device} if ":" in device else "auto"
        return kwargs

    def model_id_or_path(self) -> str:
        return self.config.local_path or self.config.model_id

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

    def prompt_template_id(self) -> str:
        return self.config.prompt_template_id or self.default_prompt_template_id

    def prompt_template_text(self) -> str:
        return self.config.prompt_template_text or "Single-image prompt receives the normalized question text unchanged."

    def expected_processor_class_name(self) -> str:
        return "AutoProcessor"

    def expected_model_class_name(self) -> str:
        return "generic_unknown"

    def chat_template_kwargs(self) -> dict[str, Any]:
        disable_argument = None
        if isinstance(self.config.thinking, Mapping):
            disable_argument = self.config.thinking.get("disable_argument")
        if disable_argument == "enable_thinking=false":
            return {"enable_thinking": False}
        return {}

    def deterministic_generation_kwargs(self, *, max_new_tokens: int | None = None) -> dict[str, Any]:
        configured = self.config.deterministic_generation or {}
        resolved_max_new_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else int(configured.get("max_new_tokens", 1))
        )
        return {
            "max_new_tokens": resolved_max_new_tokens,
            "do_sample": bool(configured.get("do_sample", False)),
            "temperature": configured.get("temperature", 0),
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_hidden_states": True,
        }

    def disable_thinking_kwargs(self) -> dict[str, Any]:
        return self.chat_template_kwargs()

    def prepare_asset_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement asset smoke input preparation")

    def resolve_prefill_hidden_states(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        generation_output: Any,
    ) -> Sequence[torch.Tensor]:
        hidden_state_steps = getattr(generation_output, "hidden_states", None)
        if hidden_state_steps:
            prefill_hidden_states = hidden_state_steps[0]
            if prefill_hidden_states is not None:
                return prefill_hidden_states
        return self.extract_prefill_hidden_states(
            model,
            processor,
            model_inputs=model_inputs,
        )

    def resolve_total_layers(self, model_or_config: Any) -> int:
        config = getattr(model_or_config, "config", model_or_config)
        return _resolve_positive_config_int(
            config,
            (
                ("num_hidden_layers",),
                ("text_config", "num_hidden_layers"),
                ("llm_config", "num_hidden_layers"),
                ("language_config", "num_hidden_layers"),
            ),
            label="total_layers",
        )

    def resolve_hidden_dim(self, model_or_config: Any) -> int:
        config = getattr(model_or_config, "config", model_or_config)
        return _resolve_positive_config_int(
            config,
            (
                ("hidden_size",),
                ("text_config", "hidden_size"),
                ("llm_config", "hidden_size"),
                ("language_config", "hidden_size"),
            ),
            label="hidden_dim",
        )

    def resolve_hidden_state_index_offset(self, hidden_states: Sequence[torch.Tensor] | None = None) -> int:
        del hidden_states
        configured = self.config.hidden_state_index_offset
        if configured in (0, 1, "0", "1"):
            return int(configured)
        raise ValueError("hidden_state_index_offset must be explicitly configured as 0 or 1")

    def parse_yes_no_response(self, text: str) -> int | None:
        cleaned = text.replace("<think>", " ").replace("</think>", " ").strip()
        return parse_yes_no_answer(cleaned)

    def resolve_query_token_index(
        self,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> int:
        del processor
        attention_mask = _model_inputs_get(model_inputs, "attention_mask")
        if attention_mask is not None:
            nonzero = torch.nonzero(attention_mask[batch_index], as_tuple=False).flatten()
            if len(nonzero) > 0:
                return int(nonzero[-1].item())
        if _model_inputs_has(model_inputs, "input_ids"):
            return int(model_inputs["input_ids"][batch_index].shape[-1] - 1)
        raise ValueError("Could not resolve query token index.")

    def resolve_vision_token_span(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> tuple[int, int] | None:
        del model, processor, model_inputs, batch_index
        return None

    def extract_preprojector_vision_features(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> torch.Tensor | None:
        del model, processor, model_inputs, batch_index
        return None

    def extract_prefill_hidden_states(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
    ) -> Any:
        outputs = self.extract_prefill_outputs(
            model,
            processor,
            model_inputs=model_inputs,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Forward output did not include hidden states.")
        return hidden_states

    def extract_prefill_outputs(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
    ) -> Any:
        del processor
        return model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

    def resolve_prefill_logits(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
        token_index: int,
    ) -> torch.Tensor:
        outputs = self.extract_prefill_outputs(
            model,
            processor,
            model_inputs=model_inputs,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Forward output did not include logits.")
        return logits[batch_index, token_index, :].detach().cpu()

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
            **self.deterministic_generation_kwargs(max_new_tokens=max_new_tokens),
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
        require_local_model_path(self.model_id_or_path())
        return configure_left_padding(
            AutoProcessor.from_pretrained(
                self.model_id_or_path(),
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True,
            )
        )

    def load_model(self, *, device: str = "cuda"):
        raise NotImplementedError


@dataclass
class LoadedModelBundle:
    processor: Any
    model: Any


@dataclass
class InternVLLocalProcessor:
    tokenizer: Any
    conversation_module: Any
    template_name: str
    image_size: int
    min_dynamic_patch: int
    max_dynamic_patch: int
    use_thumbnail: bool
    num_image_token: int
    eos_token_id: int
    eos_token_text: str

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)


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
                **self.chat_template_kwargs(),
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
            **self.chat_template_kwargs(),
        )
        batch = processor(text=[prompt], return_tensors="pt")
        return self._move_batch_to_device(batch, device)

    def prepare_asset_batch_inputs(
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
                                "text": question,
                            }
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs(),
            )
            for question in questions
        ]
        batch = processor(text=prompts, return_tensors="pt", padding=True)
        return self._move_batch_to_device(batch, device)

    def load_bundle(
        self,
        *,
        model_factory: Any = AutoModelForImageTextToText,
        processor_factory: Any = AutoProcessor,
        device: str = "cuda",
    ) -> LoadedModelBundle:
        require_local_model_path(self.model_id_or_path())
        processor = processor_factory.from_pretrained(
            self.model_id_or_path(),
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=True,
        )
        model = model_factory.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )
        return LoadedModelBundle(processor=processor, model=model)


class QwenVLWrapper(QwenWrapper):
    def expected_model_class_name(self) -> str:
        return "AutoModelForImageTextToText"

    def expected_processor_class_name(self) -> str:
        return "AutoProcessor"

    def resolve_vision_token_span(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> tuple[int, int] | None:
        del processor
        if not _model_inputs_has(model_inputs, "input_ids"):
            return None
        image_token_id = getattr(getattr(model, "config", None), "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if image_token_id is None:
            return None
        positions = torch.nonzero(model_inputs["input_ids"][batch_index] == int(image_token_id), as_tuple=False).flatten()
        if len(positions) == 0:
            return None
        return int(positions[0].item()), int(positions[-1].item())

    def extract_preprojector_vision_features(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> torch.Tensor | None:
        del processor, batch_index
        pixel_values = _model_inputs_get(model_inputs, "pixel_values")
        image_grid_thw = _model_inputs_get(model_inputs, "image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            return None
        if _model_inputs_batch_size(model_inputs) != 1 or int(image_grid_thw.shape[0]) != 1:
            return None

        visual = getattr(model, "visual", None)
        if visual is None and hasattr(model, "model"):
            visual = getattr(model.model, "visual", None)
        if visual is None or not hasattr(visual, "merger"):
            return None

        hidden_states = pixel_values.to(dtype=_module_dtype(visual, pixel_values.dtype))
        grid_thw = image_grid_thw.to(device=hidden_states.device)

        hidden_states = visual.patch_embed(hidden_states)
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for block in visual.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return hidden_states.detach().cpu()

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
                    **self.chat_template_kwargs(),
                )
            )
            images.append(Image.open(image_path).convert("RGB"))
        batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        return self._move_batch_to_device(batch, device)

    def prepare_asset_batch_inputs(
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
                raise ValueError(f"{type(self).__name__} requires an image path.")
            messages = self.build_messages(
                question=question,
                image_path=image_path,
            )
            prompts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs(),
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
            **self.chat_template_kwargs(),
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
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )


class Qwen25VLWrapper(QwenVLWrapper):
    def expected_model_class_name(self) -> str:
        return "Qwen2_5_VLForConditionalGeneration"

    def expected_processor_class_name(self) -> str:
        return "Qwen2_5_VLProcessor"

    def load_processor(self):
        from transformers import Qwen2_5_VLProcessor

        return configure_left_padding(
            Qwen2_5_VLProcessor.from_pretrained(
                self.model_id_or_path(),
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True,
            )
        )

    def load_model(self, *, device: str = "cuda"):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )


class Qwen35VLWrapper(QwenVLWrapper):
    def expected_model_class_name(self) -> str:
        return "Qwen3_5ForConditionalGeneration"

    def expected_processor_class_name(self) -> str:
        return "Qwen3VLProcessor"

    def load_processor(self):
        from transformers import Qwen3VLProcessor

        return configure_left_padding(
            Qwen3VLProcessor.from_pretrained(
                self.model_id_or_path(),
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True,
            )
        )

    def load_model(self, *, device: str = "cuda"):
        from transformers import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )


class Gemma3Wrapper(BaseModelWrapper):
    """Gemma3 local image-text wrapper using the asset chat template."""

    def expected_model_class_name(self) -> str:
        return "Gemma3ForConditionalGeneration"

    def expected_processor_class_name(self) -> str:
        return "Gemma3Processor"

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        if image_path is None:
            raise ValueError("Gemma3Wrapper requires an image path.")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(Path(image_path))},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def prepare_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        return self.prepare_asset_batch_inputs(
            processor,
            questions=[self.format_yes_no_question(question) for question in questions],
            image_paths=image_paths,
            device=device,
        )

    def prepare_asset_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        if len(questions) != len(image_paths):
            raise ValueError("questions and image_paths must have the same length.")
        prompts: list[str] = []
        images: list[Image.Image] = []
        for question, image_path in zip(questions, image_paths):
            if image_path is None:
                raise ValueError("Gemma3Wrapper requires an image path.")
            prompts.append(
                processor.apply_chat_template(
                    self.build_messages(question=question, image_path=image_path),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            images.append(Image.open(image_path).convert("RGB"))
        batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        return self._move_batch_to_device(batch, device)

    def load_processor(self):
        from transformers import Gemma3Processor

        local_path = require_local_model_path(self.model_id_or_path())
        return configure_left_padding(
            Gemma3Processor.from_pretrained(
                local_path,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True,
            )
        )

    def load_model(self, *, device: str = "cuda"):
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        ).eval()

    def resolve_prefill_hidden_states(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        generation_output: Any,
    ) -> Sequence[torch.Tensor]:
        del generation_output
        return self.extract_prefill_hidden_states(model, processor, model_inputs=model_inputs)

    def extract_prefill_outputs(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
    ) -> Any:
        del processor
        return model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )


class PhiImageTextWrapper(BaseModelWrapper):
    """Shared exact image-text path for local Phi remote-code assets."""

    processor_num_crops: ClassVar[int | None] = None

    def deterministic_generation_kwargs(self, *, max_new_tokens: int | None = None) -> dict[str, Any]:
        kwargs = super().deterministic_generation_kwargs(max_new_tokens=max_new_tokens)
        kwargs["use_cache"] = False
        return kwargs

    def model_load_kwargs(self, *, device: str = "cuda") -> dict[str, Any]:
        kwargs = super().model_load_kwargs(device=device)
        attention = kwargs.pop("attn_implementation", None)
        if attention is not None:
            kwargs["_attn_implementation"] = attention
        kwargs.setdefault("low_cpu_mem_usage", True)
        return kwargs

    def load_processor(self):
        require_local_model_path(self.model_id_or_path())
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": True,
        }
        if self.processor_num_crops is not None:
            kwargs["num_crops"] = self.processor_num_crops
        return configure_left_padding(
            AutoProcessor.from_pretrained(
                self.model_id_or_path(),
                **kwargs,
            )
        )

    def load_model(self, *, device: str = "cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )
        return model.eval()

    def build_prompt(self, question: str) -> str:
        raise NotImplementedError

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        if image_path is None:
            raise ValueError(f"{type(self).__name__} requires an image path.")
        return [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]

    def prepare_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        return self.prepare_asset_batch_inputs(
            processor,
            questions=[self.format_yes_no_question(question) for question in questions],
            image_paths=image_paths,
            device=device,
        )

    def prepare_asset_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        if len(questions) != len(image_paths):
            raise ValueError("questions and image_paths must have the same length.")
        processed: list[dict[str, torch.Tensor]] = []
        for question, image_path in zip(questions, image_paths):
            if image_path is None:
                raise ValueError(f"{type(self).__name__} requires an image path.")
            image = Image.open(image_path).convert("RGB")
            batch = processor(
                text=self.build_prompt(question),
                images=image,
                return_tensors="pt",
            )
            processed.append(
                {
                    key: value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
                    for key, value in dict(batch).items()
                    if isinstance(value, torch.Tensor)
                }
            )
        return self._move_batch_to_device(collate_tensor_dicts(processed), device)

    def resolve_prefill_hidden_states(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        generation_output: Any,
    ) -> Sequence[torch.Tensor]:
        del generation_output
        return self.extract_prefill_hidden_states(model, processor, model_inputs=model_inputs)

    def extract_prefill_outputs(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
    ) -> Any:
        del processor
        return model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )


class Phi35VisionWrapper(PhiImageTextWrapper):
    processor_num_crops: ClassVar[int | None] = 4

    def expected_model_class_name(self) -> str:
        return "Phi3VForCausalLM"

    def expected_processor_class_name(self) -> str:
        return "Phi3VProcessor"

    def build_prompt(self, question: str) -> str:
        return f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"


class Phi4MultimodalWrapper(PhiImageTextWrapper):
    def expected_model_class_name(self) -> str:
        return "Phi4MMForCausalLM"

    def expected_processor_class_name(self) -> str:
        return "Phi4MMProcessor"

    def build_prompt(self, question: str) -> str:
        return f"<|user|><|image_1|>{question}<|end|><|assistant|>"


class QwenTextWrapper(QwenWrapper):
    def expected_model_class_name(self) -> str:
        return "AutoModelForCausalLM"

    def expected_processor_class_name(self) -> str:
        return "AutoTokenizer"

    def load_processor(self):
        return configure_left_padding(
            AutoTokenizer.from_pretrained(
                self.model_id_or_path(),
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True,
            )
        )

    def load_bundle(
        self,
        *,
        model_factory: Any = AutoModelForCausalLM,
        processor_factory: Any = AutoTokenizer,
        device: str = "cuda",
    ) -> LoadedModelBundle:
        require_local_model_path(self.model_id_or_path())
        processor = processor_factory.from_pretrained(
            self.model_id_or_path(),
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=True,
        )
        model = model_factory.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )
        return LoadedModelBundle(processor=processor, model=model)

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        del image_path
        return [{"role": "user", "content": [{"type": "text", "text": question}]}]

    def load_model(self, *, device: str = "cuda"):
        return AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )


class InternVLWrapper(BaseModelWrapper):
    """Local InternVL chat wrapper using the asset's remote-code prompt path."""

    img_start_token: ClassVar[str] = "<img>"
    img_end_token: ClassVar[str] = "</img>"
    img_context_token: ClassVar[str] = "<IMG_CONTEXT>"
    image_placeholder: ClassVar[str] = "<image>"

    def expected_model_class_name(self) -> str:
        return "InternVLChatModel"

    def expected_processor_class_name(self) -> str:
        return "InternVLLocalProcessor"

    def build_messages(self, *, question: str, image_path: str | None = None) -> list[dict[str, Any]]:
        if image_path is None:
            raise ValueError("InternVLWrapper requires an image path.")
        return [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]

    def prepare_batch_inputs(
        self,
        processor: Any,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> Any:
        return self.prepare_asset_batch_inputs(
            processor,
            questions=[self.format_yes_no_question(question) for question in questions],
            image_paths=image_paths,
            device=device,
        )

    def prepare_asset_batch_inputs(
        self,
        processor: InternVLLocalProcessor,
        *,
        questions: list[str],
        image_paths: list[str | None],
        device: str,
    ) -> dict[str, Any]:
        if len(questions) != len(image_paths):
            raise ValueError("questions and image_paths must have the same length.")
        pixel_values: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        queries: list[str] = []
        for question, image_path in zip(questions, image_paths):
            if image_path is None:
                raise ValueError("InternVLWrapper requires an image path.")
            image_tensor = self._load_image(
                image_path,
                image_size=processor.image_size,
                min_num=processor.min_dynamic_patch,
                max_num=processor.max_dynamic_patch,
                use_thumbnail=processor.use_thumbnail,
            )
            num_patches = int(image_tensor.shape[0])
            pixel_values.append(image_tensor)
            num_patches_list.append(num_patches)
            queries.append(self._build_query(processor, question=question, num_patches=num_patches))

        processor.tokenizer.padding_side = "left"
        tokenized = processor.tokenizer(queries, return_tensors="pt", padding=True)
        dtype = resolve_torch_dtype(self.config.dtype)
        batch = {
            "input_ids": tokenized["input_ids"].to(device),
            "attention_mask": tokenized["attention_mask"].to(device),
            "pixel_values": torch.cat(pixel_values, dim=0).to(device=device, dtype=dtype),
            "image_flags": torch.ones((sum(num_patches_list), 1), dtype=torch.long, device=device),
            "num_patches_list": num_patches_list,
        }
        return batch

    def load_processor(self):
        model_path = Path(self.model_id_or_path()).expanduser()
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        preprocessor = {}
        preprocessor_path = model_path / "preprocessor_config.json"
        if preprocessor_path.is_file():
            preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        processor_config = {}
        processor_config_path = model_path / "processor_config.json"
        if processor_config_path.is_file():
            processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path(),
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        configure_left_padding(tokenizer)
        conversation_module = load_internvl_conversation_module(self.model_id_or_path())
        template_name = str(config.get("template") or "internvl2_5")
        template = conversation_module.get_conv_template(template_name)
        eos_token_text = str(template.sep).strip()
        eos_token_id = int(tokenizer.convert_tokens_to_ids(eos_token_text))
        size_config = preprocessor.get("size") if isinstance(preprocessor, Mapping) else None
        size_value = None
        if isinstance(size_config, Mapping):
            size_value = size_config.get("height") or size_config.get("width")
        image_size = int(config.get("force_image_size") or size_value or 448)
        return InternVLLocalProcessor(
            tokenizer=tokenizer,
            conversation_module=conversation_module,
            template_name=template_name,
            image_size=image_size,
            min_dynamic_patch=int(config.get("min_dynamic_patch") or preprocessor.get("min_patches") or 1),
            max_dynamic_patch=int(config.get("max_dynamic_patch") or preprocessor.get("max_patches") or 12),
            use_thumbnail=bool(config.get("use_thumbnail", True)),
            num_image_token=int(processor_config.get("image_seq_length") or 256),
            eos_token_id=eos_token_id,
            eos_token_text=eos_token_text,
        )

    def model_load_kwargs(self, *, device: str = "cuda") -> dict[str, Any]:
        kwargs = super().model_load_kwargs(device=device)
        kwargs.pop("attn_implementation", None)
        kwargs.pop("device_map", None)
        kwargs["use_flash_attn"] = False if self.config.attn_implementation == "eager" else True
        kwargs.setdefault("low_cpu_mem_usage", True)
        return kwargs

    def load_model(self, *, device: str = "cuda"):
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        model = AutoModel.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )
        if hasattr(model, "get_expanded_tied_weights_keys"):
            model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)
        if device.startswith("cuda"):
            model = model.to(device)
        return model.eval()

    def generate(
        self,
        model: Any,
        processor: InternVLLocalProcessor,
        *,
        model_inputs: Any,
        max_new_tokens: int,
    ) -> Any:
        self._set_img_context_token_id(model, processor)
        generation_kwargs = self.deterministic_generation_kwargs(max_new_tokens=max_new_tokens)
        return model.generate(
            pixel_values=model_inputs["pixel_values"],
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=generation_kwargs["max_new_tokens"],
            do_sample=generation_kwargs["do_sample"],
            temperature=generation_kwargs["temperature"],
            eos_token_id=processor.eos_token_id,
            return_dict_in_generate=generation_kwargs["return_dict_in_generate"],
            output_scores=generation_kwargs["output_scores"],
            output_hidden_states=generation_kwargs["output_hidden_states"],
        )

    def decode_generation(
        self,
        processor: InternVLLocalProcessor,
        *,
        generated_ids: Any,
        prompt_input_ids: Any,
    ) -> str:
        del prompt_input_ids
        decoded = processor.tokenizer.batch_decode(
            generated_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        text = str(decoded[0]).split(processor.eos_token_text)[0].strip()
        return text

    def resolve_prefill_hidden_states(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        generation_output: Any,
    ) -> Sequence[torch.Tensor]:
        del generation_output
        return self.extract_prefill_hidden_states(model, processor, model_inputs=model_inputs)

    def extract_prefill_outputs(
        self,
        model: Any,
        processor: InternVLLocalProcessor,
        *,
        model_inputs: Any,
    ) -> Any:
        self._set_img_context_token_id(model, processor)
        return model(
            pixel_values=model_inputs["pixel_values"],
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            image_flags=model_inputs["image_flags"],
            return_dict=True,
            output_hidden_states=True,
        )

    def resolve_total_layers(self, model_or_config: Any) -> int:
        config = getattr(model_or_config, "config", model_or_config)
        return _resolve_positive_config_int(config, (("llm_config", "num_hidden_layers"),), label="total_layers")

    def resolve_hidden_dim(self, model_or_config: Any) -> int:
        config = getattr(model_or_config, "config", model_or_config)
        return _resolve_positive_config_int(config, (("llm_config", "hidden_size"),), label="hidden_dim")

    def resolve_vision_token_span(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> tuple[int, int] | None:
        del processor
        input_ids = _model_inputs_get(model_inputs, "input_ids")
        if input_ids is None:
            return None
        img_context_token_id = getattr(model, "img_context_token_id", None)
        if img_context_token_id is None:
            return None
        positions = torch.nonzero(input_ids[batch_index] == int(img_context_token_id), as_tuple=False).flatten()
        if len(positions) == 0:
            return None
        return int(positions[0].item()), int(positions[-1].item())

    def extract_preprojector_vision_features(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> torch.Tensor | None:
        del processor
        pixel_values = _model_inputs_get(model_inputs, "pixel_values")
        if pixel_values is None:
            return None
        if _model_inputs_batch_size(model_inputs) != 1 or batch_index != 0:
            return None

        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is None and hasattr(model, "model"):
            vision_tower = getattr(model.model, "vision_tower", None)
        if vision_tower is None:
            return None

        outputs = vision_tower(pixel_values=pixel_values.to(dtype=_module_dtype(vision_tower, pixel_values.dtype)))
        vision_features = getattr(outputs, "last_hidden_state", None)
        if vision_features is None:
            return None
        if vision_features.ndim == 2:
            vision_features = vision_features.unsqueeze(0)
        if vision_features.shape[1] > 1:
            vision_features = vision_features[:, 1:, :]
        return vision_features.reshape(-1, vision_features.shape[-1]).detach().cpu()

    def _build_query(self, processor: InternVLLocalProcessor, *, question: str, num_patches: int) -> str:
        raw_question = question if self.image_placeholder in question else f"{self.image_placeholder}\n{question}"
        template = processor.conversation_module.get_conv_template(processor.template_name)
        template.append_message(template.roles[0], raw_question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = (
            self.img_start_token
            + self.img_context_token * processor.num_image_token * num_patches
            + self.img_end_token
        )
        if self.image_placeholder not in query:
            raise ValueError("InternVL prompt template did not include the image placeholder.")
        return query.replace(self.image_placeholder, image_tokens, 1)

    def _set_img_context_token_id(self, model: Any, processor: InternVLLocalProcessor) -> None:
        img_context_token_id = processor.tokenizer.convert_tokens_to_ids(self.img_context_token)
        if img_context_token_id is None or int(img_context_token_id) < 0:
            raise ValueError("InternVL tokenizer cannot resolve <IMG_CONTEXT> token id.")
        model.img_context_token_id = int(img_context_token_id)

    def _load_image(
        self,
        image_path: str,
        *,
        image_size: int,
        min_num: int,
        max_num: int,
        use_thumbnail: bool,
    ) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        images = self._dynamic_preprocess(
            image,
            image_size=image_size,
            min_num=min_num,
            max_num=max_num,
            use_thumbnail=use_thumbnail,
        )
        return torch.stack([self._internvl_image_to_tensor(item, image_size=image_size) for item in images], dim=0)

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        *,
        min_num: int,
        max_num: int,
        image_size: int,
        use_thumbnail: bool,
    ) -> list[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            },
            key=lambda ratio: ratio[0] * ratio[1],
        )
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            width=orig_width,
            height=orig_height,
            image_size=image_size,
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height), resample=_pil_bicubic())
        processed_images = []
        for index in range(blocks):
            box = (
                (index % (target_width // image_size)) * image_size,
                (index // (target_width // image_size)) * image_size,
                ((index % (target_width // image_size)) + 1) * image_size,
                ((index // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if len(processed_images) != blocks:
            raise ValueError("InternVL dynamic image preprocessing produced an unexpected tile count.")
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size), resample=_pil_bicubic()))
        return processed_images

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: Sequence[tuple[int, int]],
        *,
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = math.inf
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
        return best_ratio

    def _internvl_image_to_tensor(self, image: Image.Image, *, image_size: int) -> torch.Tensor:
        image = image.convert("RGB").resize((image_size, image_size), resample=_pil_bicubic())
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


class LlavaOnevisionWrapper(QwenVLWrapper):
    """LLaVA-OneVision uses the same image-plus-text chat template contract."""

    def extract_preprojector_vision_features(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> torch.Tensor | None:
        del processor
        pixel_values = _model_inputs_get(model_inputs, "pixel_values")
        if pixel_values is None:
            return None

        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is None and hasattr(model, "model"):
            vision_tower = getattr(model.model, "vision_tower", None)
        if vision_tower is None:
            return None

        if pixel_values.ndim != 5:
            return None
        sample_pixel_values = pixel_values[batch_index]
        outputs = vision_tower(
            sample_pixel_values.to(dtype=_module_dtype(vision_tower, sample_pixel_values.dtype)),
            output_hidden_states=False,
        )
        vision_features = getattr(outputs, "last_hidden_state", None)
        if vision_features is None:
            return None
        if vision_features.ndim == 2:
            vision_features = vision_features.unsqueeze(0)
        if vision_features.shape[1] > 1:
            vision_features = vision_features[:, 1:, :]
        return vision_features.reshape(-1, vision_features.shape[-1]).detach().cpu()


class MolmoWrapper(BaseModelWrapper):
    def model_load_kwargs(self, *, device: str = "cuda") -> dict[str, Any]:
        kwargs = super().model_load_kwargs(device=device)
        if kwargs.get("attn_implementation") == "sdpa":
            kwargs["attn_implementation"] = "eager"
        return kwargs

    def resolve_vision_token_span(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> tuple[int, int] | None:
        del model, processor
        if not _model_inputs_has(model_inputs, "image_input_idx"):
            return None
        image_input_idx = model_inputs["image_input_idx"][batch_index].reshape(-1)
        valid = image_input_idx[image_input_idx >= 0]
        if len(valid) == 0:
            return None
        return int(valid.min().item()), int(valid.max().item())

    def extract_preprojector_vision_features(
        self,
        model: Any,
        processor: Any,
        *,
        model_inputs: Any,
        batch_index: int,
    ) -> torch.Tensor | None:
        del processor
        images = _model_inputs_get(model_inputs, "images")
        if images is None:
            return None

        inner_model = getattr(model, "model", model)
        vision_backbone = getattr(inner_model, "vision_backbone", None)
        if vision_backbone is None or not hasattr(vision_backbone, "encode_image"):
            return None

        sample_images = images[batch_index : batch_index + 1].to(dtype=_module_dtype(vision_backbone, images.dtype))
        encoded = vision_backbone.encode_image(sample_images)
        if not isinstance(encoded, tuple) or not encoded:
            return None
        image_features = encoded[0]
        if image_features is None:
            return None

        flat_features = image_features.reshape(-1, image_features.shape[-1])
        image_masks = _model_inputs_get(model_inputs, "image_masks")
        if image_masks is not None:
            sample_mask = image_masks[batch_index : batch_index + 1]
            mask = sample_mask.reshape(-1) > 0
            if int(mask.sum().item()) > 0:
                flat_features = flat_features[mask]
        return flat_features.detach().cpu()

    def load_processor(self):
        preprocessing_module, image_module, snapshot_path = load_molmo_processing_modules(
            self.model_id_or_path()
        )
        image_processor = image_module.MolmoImageProcessor.from_pretrained(snapshot_path)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=True,
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
        batch = self._move_batch_to_device(collate_tensor_dicts(processed_inputs), device)
        if isinstance(batch, dict) and "images" in batch:
            batch["images"] = batch["images"].to(dtype=resolve_torch_dtype(self.config.dtype))
        return batch

    def prepare_asset_batch_inputs(
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
                    text=question,
                )
            )
        batch = self._move_batch_to_device(collate_tensor_dicts(processed_inputs), device)
        if isinstance(batch, dict) and "images" in batch:
            batch["images"] = batch["images"].to(dtype=resolve_torch_dtype(self.config.dtype))
        return batch

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
        generation_kwargs = self.deterministic_generation_kwargs(max_new_tokens=max_new_tokens)
        return model.generate_from_batch(
            model_inputs,
            GenerationConfig(
                max_new_tokens=generation_kwargs["max_new_tokens"],
                do_sample=generation_kwargs["do_sample"],
                temperature=generation_kwargs["temperature"],
                use_cache=True,
            ),
            tokenizer=processor.tokenizer,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

    def load_model(self, *, device: str = "cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path(),
            **self.model_load_kwargs(device=device),
        )
        prepare_inputs_for_generation = getattr(model, "prepare_inputs_for_generation", None)
        if callable(prepare_inputs_for_generation):
            original_prepare_inputs_for_generation = prepare_inputs_for_generation

            def patched_prepare_inputs_for_generation(
                self,
                input_ids: torch.LongTensor,
                past_key_values: Any = None,
                **kwargs,
            ):
                return original_prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=_normalize_molmo_past_key_values(past_key_values),
                    **kwargs,
                )

            model.prepare_inputs_for_generation = types.MethodType(
                patched_prepare_inputs_for_generation,
                model,
            )
        return model
