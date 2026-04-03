"""Comparator baselines for round-two experiments."""

from .halp import (
    HALPProbeConfig,
    build_halp_probe_frames,
    evaluate_halp_nested,
    resolve_halp_layer_indices,
)

__all__ = [
    "HALPProbeConfig",
    "build_halp_probe_frames",
    "evaluate_halp_nested",
    "resolve_halp_layer_indices",
]
