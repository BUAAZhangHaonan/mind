"""Stage C manifest, preflight, and frozen plan helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .stage_b_manifest import StageBPanelManifest, load_stage_b_panel_manifest
from .stage_b_objectives import STAGE_B_ENCODER_FAMILY


STAGE_C_OBJECTIVE = "proxy_anchor"
REQUIRED_STAGE_C_RATIO = 0.5
REQUIRED_STAGE_C_SEEDS = (20260506, 20260507, 20260508)
STAGE_C_GLM_EXCLUSION_REASON = (
    "answer format incompatible with frozen yes/no population rule"
)


def load_stage_c_panel(full_cache_root: Path | str) -> StageBPanelManifest:
    """Load the Stage C panel from the unified full-cache manifest only."""

    return load_stage_b_panel_manifest(full_cache_root)


def validate_stage_c_plan(
    *,
    ratio: float,
    seeds: Sequence[int],
    objective: str,
    encoder_family: str,
) -> dict[str, object]:
    """Validate the frozen Stage C experiment surface."""

    if str(objective) != STAGE_C_OBJECTIVE:
        raise ValueError("Stage C uses the frozen Proxy Anchor objective only")
    if str(encoder_family) != STAGE_B_ENCODER_FAMILY:
        raise ValueError(f"Stage C encoder must be {STAGE_B_ENCODER_FAMILY}")
    if round(float(ratio), 6) != REQUIRED_STAGE_C_RATIO:
        raise ValueError("Stage C negative-budget ratio must be fixed to 0.5")
    seed_values = [int(seed) for seed in seeds]
    if tuple(seed_values) != REQUIRED_STAGE_C_SEEDS:
        raise ValueError(
            "Stage C seeds must be fixed to "
            + ", ".join(str(seed) for seed in REQUIRED_STAGE_C_SEEDS)
        )
    return {
        "objective": STAGE_C_OBJECTIVE,
        "encoder_family": STAGE_B_ENCODER_FAMILY,
        "negative_budget_ratio": REQUIRED_STAGE_C_RATIO,
        "seeds": seed_values,
    }


def build_stage_c_preflight(
    panel: StageBPanelManifest,
    *,
    excluded_models: Mapping[str, object] | None = None,
    split_ready: bool,
    primary_dataset_available: bool,
) -> dict[str, object]:
    """Build a compact Stage C preflight status payload."""

    excluded = {str(model): str(reason) for model, reason in dict(excluded_models or {}).items()}
    panel_models = [str(row.get("model_alias", "")) for row in panel.models]
    missing_exclusion_rows = sorted(set(excluded) - set(panel_models))
    if missing_exclusion_rows:
        raise ValueError(
            "Stage C excluded model(s) are not in the panel: "
            + ", ".join(missing_exclusion_rows)
        )
    cache_ready = all(
        bool(row.get("cache_root") or row.get("source_cache_root"))
        for row in panel.models
    )
    return {
        "stage": "stage_c",
        "panel_manifest_path": str(panel.path),
        "manifest_source": "unified_full_cache_manifest",
        "total_panel_models": len(panel_models),
        "evaluable_models": len(panel_models) - len(excluded),
        "panel_models": panel_models,
        "excluded_models": excluded,
        "split_readiness": "ready" if bool(split_ready) else "missing",
        "primary_dataset_availability": "ready" if bool(primary_dataset_available) else "missing",
        "cache_root_readiness": "ready" if cache_ready else "missing",
        "fixed_objective": STAGE_C_OBJECTIVE,
        "fixed_encoder_family": STAGE_B_ENCODER_FAMILY,
        "fixed_negative_budget_ratio": REQUIRED_STAGE_C_RATIO,
        "fixed_seeds": list(REQUIRED_STAGE_C_SEEDS),
        "stage_d_started": False,
    }


__all__ = [
    "REQUIRED_STAGE_C_RATIO",
    "REQUIRED_STAGE_C_SEEDS",
    "STAGE_C_GLM_EXCLUSION_REASON",
    "STAGE_C_OBJECTIVE",
    "build_stage_c_preflight",
    "load_stage_c_panel",
    "validate_stage_c_plan",
]
