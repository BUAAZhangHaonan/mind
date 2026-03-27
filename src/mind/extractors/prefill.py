"""Prefill hidden-state extraction helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from mind.data import HallucinationRecord


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


def save_prefill_cache_shard(entries: Sequence[dict[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(entries), path)
