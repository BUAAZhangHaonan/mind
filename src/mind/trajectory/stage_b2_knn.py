"""Stage B2 kNN wrappers for negative-budget diagnostics."""

from __future__ import annotations

from typing import Mapping, Sequence

from .stage_b_knn import (
    ALLOWED_STAGE_B_K_VALUES,
    compute_stage_b_knn_scores,
    generate_stage_b_k_candidates,
    select_stage_b_knn_k,
)


ALLOWED_STAGE_B2_K_VALUES = ALLOWED_STAGE_B_K_VALUES


def select_stage_b2_knn_k(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select k on RePOPE cal rows and mark it frozen for Stage B2 tests."""

    selected = dict(select_stage_b_knn_k(metric_rows))
    selected["selected_on"] = "repope/cal"
    selected["frozen_for_test"] = True
    return selected


__all__ = [
    "ALLOWED_STAGE_B2_K_VALUES",
    "compute_stage_b_knn_scores",
    "generate_stage_b_k_candidates",
    "select_stage_b2_knn_k",
]
