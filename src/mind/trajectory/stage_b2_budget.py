"""Stage B2 negative-budget helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .stage_b_objectives import STAGE_B_ENCODER_FAMILY


STAGE_B2_OBJECTIVE = "proxy_anchor"
REQUIRED_STAGE_B2_RATIOS = (1.0, 0.5, 0.25, 0.10)
OPTIONAL_STAGE_B2_RATIOS = (0.05,)
REQUIRED_STAGE_B2_SEEDS = (20260506, 20260507, 20260508)


def validate_stage_b2_budget_plan(
    *,
    ratios: Sequence[float],
    seeds: Sequence[int],
    objective: str,
    encoder_family: str,
) -> dict[str, object]:
    """Validate the frozen Stage B2 budget experiment surface."""

    seed_values = [int(seed) for seed in seeds]
    if tuple(seed_values) != REQUIRED_STAGE_B2_SEEDS:
        raise ValueError(
            "Stage B2 seeds must be fixed to "
            + ", ".join(str(seed) for seed in REQUIRED_STAGE_B2_SEEDS)
        )
    if str(objective) != STAGE_B2_OBJECTIVE:
        raise ValueError("Stage B2 trains Proxy Anchor only; objective must be proxy_anchor")
    if str(encoder_family) != STAGE_B_ENCODER_FAMILY:
        raise ValueError(f"Stage B2 encoder must be {STAGE_B_ENCODER_FAMILY}")

    ratio_values = [round(float(value), 6) for value in ratios]
    allowed = {round(float(value), 6) for value in REQUIRED_STAGE_B2_RATIOS + OPTIONAL_STAGE_B2_RATIOS}
    unsupported = [value for value in ratio_values if value not in allowed]
    if unsupported:
        raise ValueError(f"Stage B2 ratio is not allowed: {unsupported}")

    required = {round(float(value), 6) for value in REQUIRED_STAGE_B2_RATIOS}
    missing = sorted(required - set(ratio_values), reverse=True)
    if missing:
        raise ValueError(f"Stage B2 required ratio(s) missing: {missing}")

    return {
        "ratios": [float(value) for value in ratio_values],
        "seeds": seed_values,
        "objective": STAGE_B2_OBJECTIVE,
        "encoder_family": STAGE_B_ENCODER_FAMILY,
        "optional_stress_ratio_included": round(0.05, 6) in ratio_values,
    }


def subsample_stage_b2_training_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    ratio: float,
    seed: int,
    min_selected_hard_negatives: int = 20,
) -> np.ndarray:
    """Use all correct rows and subsample hard hallucinations without replacement."""

    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.size == 0:
        raise ValueError("labels must not be empty")
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("Stage B2 labels must be binary 0/1")
    correct_indices = np.flatnonzero(label_array == 0)
    hard_indices = np.flatnonzero(label_array == 1)
    if correct_indices.size == 0 or hard_indices.size == 0:
        raise ValueError("Stage B2 training labels must contain correct and hard hallucination rows")

    ratio_value = float(ratio)
    if ratio_value <= 0.0 or ratio_value > 1.0:
        raise ValueError("ratio must be in (0, 1]")
    selected_hard_count = int(np.floor(hard_indices.size * ratio_value))
    selected_hard_count = max(selected_hard_count, 1)
    if hard_indices.size >= min_selected_hard_negatives and selected_hard_count < min_selected_hard_negatives:
        raise ValueError(
            f"ratio {ratio_value:g} would leave fewer than 20 hard negatives "
            f"({selected_hard_count})"
        )

    rng = np.random.default_rng(int(seed))
    selected_hard = rng.choice(hard_indices, size=selected_hard_count, replace=False)
    selected = np.concatenate([correct_indices, np.sort(selected_hard)])
    return np.sort(selected).astype(np.int64, copy=False)
