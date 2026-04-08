"""Separate pre-generation readout extraction helpers for comparator methods."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import torch

from mind.data import HallucinationRecord
from mind.models import parse_yes_no_answer

from .prefill import resolve_prefill_hidden_states, run_generation_with_prefill_request


def resolve_default_glsim_layer_indices(total_layers: int) -> list[int]:
    if total_layers < 1:
        raise ValueError("total_layers must be positive")
    return [
        0,
        total_layers // 4,
        total_layers // 2,
        (3 * total_layers) // 4,
        total_layers - 1,
    ]


def stack_prefill_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    *,
    batch_index: int = 0,
) -> torch.Tensor:
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must include embeddings plus decoder layers.")
    first_layer = hidden_states[1][batch_index].detach()
    stacked = torch.empty(
        (len(hidden_states) - 1, *first_layer.shape),
        dtype=first_layer.dtype,
        device="cpu",
    )
    stacked[0].copy_(first_layer)
    for layer_index, state in enumerate(hidden_states[2:], start=1):
        stacked[layer_index].copy_(state[batch_index].detach())
    return stacked


def _default_query_token_index(model_inputs: Any, *, batch_index: int) -> int:
    attention_mask = None
    if hasattr(model_inputs, "get"):
        attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        nonzero = torch.nonzero(attention_mask[batch_index], as_tuple=False).flatten()
        if len(nonzero) > 0:
            return int(nonzero[-1].item())
    try:
        has_input_ids = "input_ids" in model_inputs
    except TypeError:
        has_input_ids = False
    if has_input_ids:
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


def _resolve_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _build_prompt_text_for_record(
    wrapper: Any,
    processor: Any,
    record: HallucinationRecord,
) -> tuple[str | None, str]:
    formatted_question = (
        str(wrapper.format_yes_no_question(record.question))
        if hasattr(wrapper, "format_yes_no_question")
        else str(record.question)
    )
    if not hasattr(processor, "apply_chat_template") or not hasattr(wrapper, "build_messages"):
        return None, formatted_question
    try:
        prompt_text = processor.apply_chat_template(
            wrapper.build_messages(
                question=formatted_question,
                image_path=record.image_path,
            ),
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return None, formatted_question
    return str(prompt_text), formatted_question


def _find_subsequence_start(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
    if not subsequence:
        return None
    width = len(subsequence)
    for index in range(0, len(sequence) - width + 1):
        if list(sequence[index : index + width]) == list(subsequence):
            return index
    return None


def _resolve_object_token_context(
    wrapper: Any,
    processor: Any,
    record: HallucinationRecord,
    *,
    model_inputs: Any,
    batch_index: int,
) -> tuple[int, int]:
    tokenizer = _resolve_tokenizer(processor)
    input_ids = model_inputs["input_ids"][batch_index].detach().cpu().tolist()
    prompt_text, formatted_question = _build_prompt_text_for_record(wrapper, processor, record)
    if prompt_text is not None:
        question_char_start = prompt_text.lower().find(formatted_question.lower())
        search_start = 0 if question_char_start < 0 else question_char_start
        object_char_start = prompt_text.lower().find(str(record.object_name).lower(), search_start)
        if object_char_start >= 0:
            try:
                tokenized = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offset_mapping = tokenized.get("offset_mapping")
                prompt_input_ids = tokenized.get("input_ids")
            except Exception:
                offset_mapping = None
                prompt_input_ids = None
            if offset_mapping is not None and prompt_input_ids is not None:
                object_char_stop = object_char_start + len(str(record.object_name))
                matched_indices = [
                    index
                    for index, offset in enumerate(offset_mapping)
                    if int(offset[0]) < object_char_stop and int(offset[1]) > object_char_start
                ]
                if matched_indices:
                    object_token_ids = [
                        int(value)
                        for value in prompt_input_ids[matched_indices[0] : matched_indices[-1] + 1]
                    ]
                    start_index = _find_subsequence_start(input_ids, object_token_ids)
                    if start_index is not None:
                        return int(start_index), int(object_token_ids[0])

    candidate_texts = [
        str(record.object_name),
        f" {record.object_name}",
        str(record.object_name).title(),
        f" {str(record.object_name).title()}",
    ]
    for candidate_text in candidate_texts:
        try:
            object_token_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
        except TypeError:
            object_token_ids = tokenizer.encode(candidate_text)
        if not object_token_ids:
            continue
        start_index = _find_subsequence_start(input_ids, object_token_ids)
        if start_index is not None:
            return int(start_index), int(object_token_ids[0])
    raise ValueError(f"Could not locate object token span for sample {record.sample_id}")


def build_prefill_readout_entry(
    *,
    record: HallucinationRecord,
    answer_text: str,
    parsed_answer: int | None,
    full_hidden_states: torch.Tensor,
    query_token_index: int,
    vision_token_span: tuple[int, int] | None,
    object_token_index: int | None = None,
    object_token_id: int | None = None,
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
            "object_token_index": None if object_token_index is None else int(object_token_index),
            "object_token_id": None if object_token_id is None else int(object_token_id),
            "first_token_logits": None
            if first_token_logits is None
            else first_token_logits.detach().cpu(),
            "vision_features": None if vision_features is None else vision_features.detach().cpu(),
        }
    )
    return payload


def compact_prefill_readout_entry(entry: dict[str, object]) -> dict[str, object]:
    full_hidden_states = torch.as_tensor(entry["full_hidden_states"])
    query_token_index = int(entry["query_token_index"])
    vision_token_span = entry.get("vision_token_span")
    object_token_index = entry.get("object_token_index")
    if vision_token_span is None:
        raise ValueError(f"Missing vision token span for sample {entry['sample_id']}")
    if object_token_index is None:
        raise ValueError(f"Missing object token index for sample {entry['sample_id']}")

    vision_start, vision_stop = int(vision_token_span[0]), int(vision_token_span[1])
    vision_token_index = vision_stop
    total_layers = int(full_hidden_states.shape[0])
    glsim_layer_indices = resolve_default_glsim_layer_indices(total_layers)

    compact_entry = {key: value for key, value in entry.items() if key != "full_hidden_states"}
    compact_entry.update(
        {
            "readout_format": "compact_comparator_cache_v1",
            "total_layers": total_layers,
            "query_hidden_states": full_hidden_states[:, query_token_index, :].clone(),
            "vision_token_hidden_states": full_hidden_states[:, vision_token_index, :].clone(),
            "object_hidden_states": full_hidden_states[:, int(object_token_index), :].clone(),
            "glsim_layer_indices": list(glsim_layer_indices),
            "glsim_vision_hidden_states": full_hidden_states[glsim_layer_indices, vision_start : vision_stop + 1, :].clone(),
        }
    )
    return compact_entry


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
        query_token_index = _resolve_query_token_index(
            wrapper,
            processor,
            model_inputs=model_inputs,
            batch_index=batch_index,
        )
        vision_token_span = _resolve_vision_token_span(
            wrapper,
            model,
            processor,
            model_inputs=model_inputs,
            batch_index=batch_index,
        )
        object_token_index, object_token_id = _resolve_object_token_context(
            wrapper,
            processor,
            record,
            model_inputs=model_inputs,
            batch_index=batch_index,
        )
        full_hidden_states = stack_prefill_hidden_states(
            prefill_hidden_states,
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
                query_token_index=query_token_index,
                vision_token_span=vision_token_span,
                object_token_index=object_token_index,
                object_token_id=object_token_id,
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
