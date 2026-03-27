"""Prefill hidden-state extraction helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from mind.data import HallucinationRecord
from mind.models import parse_yes_no_answer


def select_middle_layers(*, total_layers: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if count > total_layers:
        raise ValueError("count cannot exceed total_layers")

    start = total_layers // 4
    end = total_layers - start - 1
    if count == 1:
        return [(start + end) // 2]
    return [
        round(start + step * (end - start) / (count - 1))
        for step in range(count)
    ]


def extract_prefill_vectors(
    hidden_states: Sequence[torch.Tensor],
    *,
    selected_layers: Sequence[int],
    token_index: int = -1,
) -> torch.Tensor:
    vectors = []
    for layer in selected_layers:
        state = hidden_states[layer + 1]
        vectors.append(state[0, token_index, :].detach().cpu())
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
    model_inputs = wrapper.prepare_inputs(
        processor,
        question=record.question,
        image_path=record.image_path,
        device=device,
    )
    outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
    layer_vectors = extract_prefill_vectors(
        outputs.hidden_states,
        selected_layers=selected_layers,
        token_index=token_index,
    )
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    answer_text = wrapper.decode_generation(
        processor,
        generated_ids=generated_ids,
        prompt_input_ids=model_inputs["input_ids"],
    )
    return build_prefill_cache_entry(
        record=record,
        answer_text=answer_text,
        parsed_answer=parse_yes_no_answer(answer_text),
        selected_layers=selected_layers,
        layer_vectors=layer_vectors,
        first_token_logits=outputs.logits[0, token_index, :],
    )


def save_prefill_cache_shard(entries: Sequence[dict[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(entries), path)
