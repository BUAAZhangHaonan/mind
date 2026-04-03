"""Separate pre-generation readout extraction helpers for comparator methods."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import torch

from mind.data import HallucinationRecord
from mind.models import parse_yes_no_answer


def stack_prefill_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    *,
    batch_index: int = 0,
) -> torch.Tensor:
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must include embeddings plus decoder layers.")
    return torch.stack(
        [state[batch_index].detach().cpu() for state in hidden_states[1:]],
        dim=0,
    )


def _default_query_token_index(model_inputs: Any, *, batch_index: int) -> int:
    attention_mask = None
    if isinstance(model_inputs, dict):
        attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        nonzero = torch.nonzero(attention_mask[batch_index], as_tuple=False).flatten()
        if len(nonzero) > 0:
            return int(nonzero[-1].item())
    if isinstance(model_inputs, dict) and "input_ids" in model_inputs:
        return int(model_inputs["input_ids"][batch_index].shape[-1] - 1)
    raise ValueError("Could not resolve query token index from model inputs.")


def _resolve_query_token_index(
    wrapper: Any,
    processor: Any,
    *,
    model_inputs: Any,
    batch_index: int,
) -> int:
    resolver = getattr(wrapper, "resolve_query_token_index", None)
    if callable(resolver):
        return int(
            resolver(
                processor,
                model_inputs=model_inputs,
                batch_index=batch_index,
            )
        )
    return _default_query_token_index(model_inputs, batch_index=batch_index)


def _resolve_vision_token_span(
    wrapper: Any,
    model: Any,
    processor: Any,
    *,
    model_inputs: Any,
    batch_index: int,
) -> tuple[int, int] | None:
    resolver = getattr(wrapper, "resolve_vision_token_span", None)
    if not callable(resolver):
        return None
    span = resolver(
        model,
        processor,
        model_inputs=model_inputs,
        batch_index=batch_index,
    )
    if span is None:
        return None
    start, stop = span
    return int(start), int(stop)


def _resolve_vision_features(
    wrapper: Any,
    model: Any,
    processor: Any,
    *,
    model_inputs: Any,
    batch_index: int,
) -> torch.Tensor | None:
    extractor = getattr(wrapper, "extract_preprojector_vision_features", None)
    if not callable(extractor):
        return None
    features = extractor(
        model,
        processor,
        model_inputs=model_inputs,
        batch_index=batch_index,
    )
    if features is None:
        return None
    return torch.as_tensor(features).detach().cpu()


def build_prefill_readout_entry(
    *,
    record: HallucinationRecord,
    answer_text: str,
    parsed_answer: int | None,
    full_hidden_states: torch.Tensor,
    query_token_index: int,
    vision_token_span: tuple[int, int] | None,
    first_token_logits: torch.Tensor | None,
    vision_features: torch.Tensor | None,
) -> dict[str, object]:
    payload = asdict(record)
    payload.update(
        {
            "answer_text": answer_text,
            "parsed_answer": parsed_answer,
            "full_hidden_states": full_hidden_states.detach().cpu(),
            "query_token_index": int(query_token_index),
            "vision_token_span": None if vision_token_span is None else list(vision_token_span),
            "first_token_logits": None
            if first_token_logits is None
            else first_token_logits.detach().cpu(),
            "vision_features": None if vision_features is None else vision_features.detach().cpu(),
        }
    )
    return payload


def extract_prefill_readout_entries(
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    records: Sequence[HallucinationRecord],
    device: str,
    max_new_tokens: int = 1,
) -> list[dict[str, object]]:
    model_inputs = wrapper.prepare_batch_inputs(
        processor,
        questions=[record.question for record in records],
        image_paths=[record.image_path for record in records],
        device=device,
    )
    generate = getattr(wrapper, "generate", None)
    if callable(generate):
        generation_output = generate(
            model,
            processor,
            model_inputs=model_inputs,
            max_new_tokens=max_new_tokens,
        )
    else:
        generation_output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
    if not generation_output.hidden_states:
        raise ValueError("Generation output did not include hidden states.")
    if not generation_output.scores:
        raise ValueError("Generation output did not include token scores.")

    entries: list[dict[str, object]] = []
    for batch_index, record in enumerate(records):
        full_hidden_states = stack_prefill_hidden_states(
            generation_output.hidden_states[0],
            batch_index=batch_index,
        )
        answer_text = wrapper.decode_generation(
            processor,
            generated_ids=generation_output.sequences[batch_index : batch_index + 1],
            prompt_input_ids=model_inputs["input_ids"][batch_index : batch_index + 1],
        )
        entries.append(
            build_prefill_readout_entry(
                record=record,
                answer_text=answer_text,
                parsed_answer=parse_yes_no_answer(answer_text),
                full_hidden_states=full_hidden_states,
                query_token_index=_resolve_query_token_index(
                    wrapper,
                    processor,
                    model_inputs=model_inputs,
                    batch_index=batch_index,
                ),
                vision_token_span=_resolve_vision_token_span(
                    wrapper,
                    model,
                    processor,
                    model_inputs=model_inputs,
                    batch_index=batch_index,
                ),
                first_token_logits=generation_output.scores[0][batch_index],
                vision_features=_resolve_vision_features(
                    wrapper,
                    model,
                    processor,
                    model_inputs=model_inputs,
                    batch_index=batch_index,
                ),
            )
        )
    return entries


def extract_prefill_readout_entry(
    *,
    model: Any,
    processor: Any,
    wrapper: Any,
    record: HallucinationRecord,
    device: str,
    max_new_tokens: int = 1,
) -> dict[str, object]:
    return extract_prefill_readout_entries(
        model=model,
        processor=processor,
        wrapper=wrapper,
        records=[record],
        device=device,
        max_new_tokens=max_new_tokens,
    )[0]
