"""Local manifold construction for MIND."""

from .local_pca import (
    LocalPCAManifold,
    SHARED_BANK_KEY,
    build_reference_bank,
    clean_reference_entries,
    compute_reference_bank_stats,
    fit_local_pca_manifold,
    normalized_normal_residual,
    resolve_reference_scope_key,
)

__all__ = [
    "LocalPCAManifold",
    "SHARED_BANK_KEY",
    "build_reference_bank",
    "clean_reference_entries",
    "compute_reference_bank_stats",
    "fit_local_pca_manifold",
    "normalized_normal_residual",
    "resolve_reference_scope_key",
]
