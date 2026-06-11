"""Stage B2 manifest and preflight helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .stage_b_manifest import StageBPanelManifest, load_stage_b_panel_manifest


def load_stage_b2_panel(full_cache_root: Path | str) -> StageBPanelManifest:
    """Load the unified full-cache panel through the Stage B manifest contract."""

    return load_stage_b_panel_manifest(full_cache_root)


def build_stage_b2_preflight(
    panel: StageBPanelManifest,
    *,
    excluded_models: Mapping[str, object] | None = None,
    split_ready: bool,
    primary_dataset_available: bool,
) -> dict[str, object]:
    """Build a compact Stage B2 preflight status payload."""

    excluded = {str(model): str(reason) for model, reason in dict(excluded_models or {}).items()}
    panel_models = [str(row.get("model_alias", "")) for row in panel.models]
    missing_exclusion_rows = sorted(set(excluded) - set(panel_models))
    if missing_exclusion_rows:
        raise ValueError(
            "Stage B2 excluded model(s) are not in the panel: "
            + ", ".join(missing_exclusion_rows)
        )

    cache_ready = all(
        bool(row.get("cache_root") or row.get("source_cache_root"))
        for row in panel.models
    )
    return {
        "stage": "stage_b2",
        "panel_manifest_path": str(panel.path),
        "total_panel_models": len(panel_models),
        "evaluable_models": len(panel_models) - len(excluded),
        "panel_models": panel_models,
        "excluded_models": excluded,
        "split_readiness": "ready" if bool(split_ready) else "missing",
        "primary_dataset_availability": "ready" if bool(primary_dataset_available) else "missing",
        "cache_root_readiness": "ready" if cache_ready else "missing",
    }
