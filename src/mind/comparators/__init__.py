"""Comparator baselines for round-two experiments."""

from .halp import (
    HALPProbeConfig,
    build_halp_probe_frames,
    evaluate_halp_nested,
    resolve_halp_layer_indices,
)
from .glsim_adapted import (
    build_glsim_score_frame as build_glsim_adapted_score_frame,
    build_object_token_contexts as build_glsim_adapted_object_token_contexts,
    evaluate_glsim_nested as evaluate_glsim_adapted_nested,
    find_subsequence_start as find_glsim_adapted_subsequence_start,
    resolve_glsim_layer_indices as resolve_glsim_adapted_layer_indices,
)

__all__ = [
    "HALPProbeConfig",
    "build_halp_probe_frames",
    "build_glsim_adapted_score_frame",
    "build_glsim_adapted_object_token_contexts",
    "evaluate_halp_nested",
    "evaluate_glsim_adapted_nested",
    "find_glsim_adapted_subsequence_start",
    "resolve_glsim_adapted_layer_indices",
    "resolve_halp_layer_indices",
]
