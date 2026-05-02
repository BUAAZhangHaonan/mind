"""Prefill hidden-state extraction helpers."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from mind.data import HallucinationRecord
from mind.models.types import parse_yes_no_answer


CHUNKED_CACHE_SHARD_FORMAT = "chunked_cache_shard_v1"
PREFILL_CACHE_METADATA_VERSION = 1


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def run_generation_with_prefill_request(
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    model_inputs: Any,
    max_new_tokens: int,
) -> Any:
    generate = getattr(wrapper, "generate", None)
    if callable(generate):
        return generate(
            model,
            processor,
            model_inputs=model_inputs,
            max_new_tokens=max_new_tokens,
        )
    return model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
    )


def resolve_prefill_hidden_states(
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    model_inputs: Any,
    generation_output: Any,
) -> Sequence[torch.Tensor]:
    hidden_state_steps = getattr(generation_output, "hidden_states", None)
    if hidden_state_steps:
        prefill_hidden_states = hidden_state_steps[0]
        if prefill_hidden_states is not None:
            return prefill_hidden_states

    extractor = getattr(wrapper, "extract_prefill_hidden_states", None)
    if callable(extractor):
        hidden_states = extractor(
            model,
            processor,
            model_inputs=model_inputs,
        )
        if hidden_states:
            return hidden_states
    raise ValueError("Could not resolve prefill hidden states from generation or wrapper forward pass.")


def select_layer_range(*, total_layers: int, count: int, range_name: str) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if count > total_layers:
        raise ValueError("count cannot exceed total_layers")

    normalized = range_name.strip().lower()
    if normalized == "early":
        start = 0
        end = max(0, total_layers // 2 - 1)
    elif normalized == "middle":
        start = total_layers // 4
        end = total_layers - start - 1
    elif normalized == "late":
        start = total_layers // 2
        end = total_layers - 1
    elif normalized == "all":
        start = 0
        end = total_layers - 1
    else:
        raise ValueError(f"Unsupported layer range: {range_name}")

    available = end - start + 1
    if count > available:
        raise ValueError(f"count cannot exceed available layers in the {normalized} range")
    if count == 1:
        return [(start + end) // 2]
    return [
        round(start + step * (end - start) / (count - 1))
        for step in range(count)
    ]


def select_middle_layers(*, total_layers: int, count: int) -> list[int]:
    return select_layer_range(total_layers=total_layers, count=count, range_name="middle")


def extract_prefill_vectors(
    hidden_states: Sequence[torch.Tensor],
    *,
    selected_layers: Sequence[int],
    token_index: int = -1,
    batch_index: int = 0,
) -> torch.Tensor:
    vectors = []
    for layer in selected_layers:
        state = hidden_states[layer + 1]
        vectors.append(state[batch_index, token_index, :].detach().cpu())
    return torch.stack(vectors, dim=0)


def build_prefill_cache_entry(
    *,
    record: HallucinationRecord,
    answer_text: str,
    parsed_answer: int | None,
    selected_layers: Sequence[int],
    layer_vectors: torch.Tensor,
    first_token_logits: torch.Tensor | None,
) -> dict[str, object]:
    payload = asdict(record)
    payload.update(
        {
            "answer_text": answer_text,
            "parsed_answer": parsed_answer,
            "selected_layers": list(selected_layers),
            "layer_vectors": layer_vectors.detach().cpu(),
            "first_token_logits": None
            if first_token_logits is None
            else first_token_logits.detach().cpu(),
        }
    )
    return payload


def _prepare_cache_value(
    value: object,
    *,
    dtype: torch.dtype,
    cast_floating_tensors: bool,
) -> object:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if cast_floating_tensors and tensor.is_floating_point():
            return tensor.to(dtype=dtype)
        return tensor
    if isinstance(value, dict):
        return {
            key: _prepare_cache_value(
                item,
                dtype=dtype,
                cast_floating_tensors=cast_floating_tensors,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _prepare_cache_value(
                item,
                dtype=dtype,
                cast_floating_tensors=cast_floating_tensors,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _prepare_cache_value(
                item,
                dtype=dtype,
                cast_floating_tensors=cast_floating_tensors,
            )
            for item in value
        )
    return value


def cast_prefill_cache_entries(
    entries: Sequence[dict[str, object]],
    *,
    dtype: torch.dtype = torch.float16,
    cast_all_floating_tensors: bool = False,
) -> list[dict[str, object]]:
    cast_entries: list[dict[str, object]] = []
    for entry in entries:
        cast_entry: dict[str, object] = {}
        for key, value in dict(entry).items():
            cast_entry[key] = _prepare_cache_value(
                value,
                dtype=dtype,
                cast_floating_tensors=cast_all_floating_tensors or key == "layer_vectors",
            )
        cast_entries.append(cast_entry)
    return cast_entries


def _tensor_payload_bytes(
    value: object,
    *,
    dtype: torch.dtype,
    cast_floating_tensors: bool,
) -> int:
    if isinstance(value, torch.Tensor):
        if cast_floating_tensors and value.is_floating_point():
            return int(value.numel() * torch.empty((), dtype=dtype).element_size())
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(
            _tensor_payload_bytes(
                item,
                dtype=dtype,
                cast_floating_tensors=cast_floating_tensors,
            )
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _tensor_payload_bytes(
                item,
                dtype=dtype,
                cast_floating_tensors=cast_floating_tensors,
            )
            for item in value
        )
    return 0


def estimate_prefill_cache_tensor_bytes(
    entries: Sequence[dict[str, object]],
    *,
    dtype: torch.dtype = torch.float16,
    cast_all_floating_tensors: bool = False,
) -> int:
    total = 0
    for entry in entries:
        for key, value in dict(entry).items():
            total += _tensor_payload_bytes(
                value,
                dtype=dtype,
                cast_floating_tensors=cast_all_floating_tensors or key == "layer_vectors",
            )
    return total


def _collect_tensor_metadata(value: object, *, prefix: str = "") -> list[dict[str, object]]:
    if isinstance(value, torch.Tensor):
        return [
            {
                "field": prefix,
                "shape": list(value.shape),
                "dtype": _dtype_name(value.dtype),
                "numel": int(value.numel()),
            }
        ]
    if isinstance(value, dict):
        collected: list[dict[str, object]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            field = str(key) if not prefix else f"{prefix}.{key}"
            collected.extend(_collect_tensor_metadata(item, prefix=field))
        return collected
    if isinstance(value, (list, tuple)):
        collected = []
        for index, item in enumerate(value):
            field = f"{prefix}[{index}]"
            collected.extend(_collect_tensor_metadata(item, prefix=field))
        return collected
    return []


def prefill_cache_sidecar_path(path: str | Path) -> Path:
    shard_path = Path(path)
    return shard_path.with_suffix(shard_path.suffix + ".json")


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_prefill_cache_sidecar(
    *,
    path: Path,
    entries: Sequence[dict[str, object]],
    dtype: torch.dtype,
    estimated_tensor_bytes: int,
    actual_file_bytes: int,
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    selected_layers: list[int] = []
    if entries:
        selected_layers = [int(layer) for layer in entries[0].get("selected_layers", [])]
    config = {
        "dtype": _dtype_name(dtype),
        "selected_layers": selected_layers,
        **(metadata or {}),
    }
    return {
        "metadata_version": PREFILL_CACHE_METADATA_VERSION,
        "format": "prefill_cache_shard_v1",
        "path": str(path),
        "num_entries": len(entries),
        "dtype": _dtype_name(dtype),
        "selected_layers": selected_layers,
        "estimated_tensor_bytes": int(estimated_tensor_bytes),
        "actual_file_bytes": int(actual_file_bytes),
        "content_sha256": _content_sha256(path),
        "config": config,
        "config_hash": _config_hash(config),
        "tensor_fields": _collect_tensor_metadata(entries[0]) if entries else [],
    }


def write_prefill_cache_sidecar(path: str | Path, sidecar: dict[str, object]) -> Path:
    sidecar_path = prefill_cache_sidecar_path(path)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


def cleanup_prefill_cache_shard(path: str | Path) -> None:
    shard_path = Path(path)
    prefill_cache_sidecar_path(shard_path).unlink(missing_ok=True)
    shard_path.unlink(missing_ok=True)


def extract_prefill_entry(
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    record: HallucinationRecord,
    selected_layers: Sequence[int],
    device: str,
    token_index: int = -1,
    max_new_tokens: int = 4,
) -> dict[str, object]:
    return extract_prefill_entries(
        model=model,
        processor=processor,
        wrapper=wrapper,
        records=[record],
        selected_layers=selected_layers,
        device=device,
        token_index=token_index,
        max_new_tokens=max_new_tokens,
    )[0]


def extract_prefill_entries(
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    records: Sequence[HallucinationRecord],
    selected_layers: Sequence[int],
    device: str,
    token_index: int = -1,
    max_new_tokens: int = 4,
) -> list[dict[str, object]]:
    model_inputs = wrapper.prepare_batch_inputs(
        processor,
        questions=[record.question for record in records],
        image_paths=[record.image_path for record in records],
        device=device,
    )
    generation_output = run_generation_with_prefill_request(
        model=model,
        processor=processor,
        wrapper=wrapper,
        model_inputs=model_inputs,
        max_new_tokens=max_new_tokens,
    )
    if not generation_output.scores:
        raise ValueError("Generation output did not include token scores.")
    prefill_hidden_states = resolve_prefill_hidden_states(
        model=model,
        processor=processor,
        wrapper=wrapper,
        model_inputs=model_inputs,
        generation_output=generation_output,
    )
    entries: list[dict[str, object]] = []
    for batch_index, record in enumerate(records):
        layer_vectors = extract_prefill_vectors(
            prefill_hidden_states,
            selected_layers=selected_layers,
            token_index=token_index,
            batch_index=batch_index,
        )
        answer_text = wrapper.decode_generation(
            processor,
            generated_ids=generation_output.sequences[batch_index : batch_index + 1],
            prompt_input_ids=model_inputs["input_ids"][batch_index : batch_index + 1],
        )
        entries.append(
            build_prefill_cache_entry(
                record=record,
                answer_text=answer_text,
                parsed_answer=parse_yes_no_answer(answer_text),
                selected_layers=selected_layers,
                layer_vectors=layer_vectors,
                first_token_logits=generation_output.scores[0][batch_index],
            )
        )
    return entries


def save_prefill_cache_shard(
    entries: Sequence[dict[str, object]],
    output_path: str | Path,
    *,
    dtype: torch.dtype = torch.float16,
    cast_all_floating_tensors: bool = False,
    estimated_tensor_bytes: int | None = None,
    metadata: dict[str, object] | None = None,
    verify_readback: bool = True,
) -> dict[str, object]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cast_entries = cast_prefill_cache_entries(
        entries,
        dtype=dtype,
        cast_all_floating_tensors=cast_all_floating_tensors,
    )
    estimated = (
        estimate_prefill_cache_tensor_bytes(
            entries,
            dtype=dtype,
            cast_all_floating_tensors=cast_all_floating_tensors,
        )
        if estimated_tensor_bytes is None
        else int(estimated_tensor_bytes)
    )
    torch.save(cast_entries, path)
    if verify_readback:
        restored = torch.load(path, weights_only=False)
        if not isinstance(restored, list):
            cleanup_prefill_cache_shard(path)
            raise ValueError(f"Read-back payload for {path} is not a list")
        if len(restored) != len(cast_entries):
            cleanup_prefill_cache_shard(path)
            raise ValueError(f"Read-back record count mismatch for {path}")
        if not all(isinstance(entry, dict) for entry in restored):
            cleanup_prefill_cache_shard(path)
            raise ValueError(f"Read-back payload for {path} contains non-dict records")
    sidecar = _build_prefill_cache_sidecar(
        path=path,
        entries=cast_entries,
        dtype=dtype,
        estimated_tensor_bytes=estimated,
        actual_file_bytes=path.stat().st_size,
        metadata=metadata,
    )
    sidecar_path = write_prefill_cache_sidecar(path, sidecar)
    if verify_readback:
        try:
            restored_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            cleanup_prefill_cache_shard(path)
            raise ValueError(f"Read-back sidecar for {path} is not readable: {exc}") from exc
        if restored_sidecar.get("content_sha256") != _content_sha256(path):
            cleanup_prefill_cache_shard(path)
            raise ValueError(f"Read-back sidecar content hash mismatch for {path}")
    return sidecar
