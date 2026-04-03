"""Prefill hidden-state extraction helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from mind.data import HallucinationRecord
from mind.models import parse_yes_no_answer


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


def save_prefill_cache_shard(entries: Sequence[dict[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(entries), path)
