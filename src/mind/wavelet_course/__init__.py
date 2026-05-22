"""Wavelet-course data, population, split, and metrics helpers."""

from .cache_loading import (
    CacheShard,
    iter_repope_qwen_cache_shards,
    load_repope_qwen_cache_entries,
    stream_repope_qwen_cache_entries,
    validate_cache_entry,
    validate_first_token_logits,
    validate_layer_vectors,
)
from .metrics import (
    ThresholdSelection,
    binary_metrics,
    evaluate_validation_test,
    select_best_f1_threshold,
)
from .population import (
    PopulationClass,
    WaveletPopulation,
    build_grouped_split_assignments,
    build_population_audit_rows,
    build_wavelet_population,
    classify_entry,
    load_or_build_split_assignments,
    primary_label,
)

__all__ = [
    "CacheShard",
    "PopulationClass",
    "ThresholdSelection",
    "WaveletPopulation",
    "binary_metrics",
    "build_grouped_split_assignments",
    "build_population_audit_rows",
    "build_wavelet_population",
    "classify_entry",
    "evaluate_validation_test",
    "iter_repope_qwen_cache_shards",
    "load_or_build_split_assignments",
    "load_repope_qwen_cache_entries",
    "primary_label",
    "select_best_f1_threshold",
    "stream_repope_qwen_cache_entries",
    "validate_cache_entry",
    "validate_first_token_logits",
    "validate_layer_vectors",
]
