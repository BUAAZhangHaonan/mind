from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_TRACKED_PATH_PREFIXES = (
    "configs/experiments/",
    "src/mind/drift/",
    "src/mind/manifolds/",
    "src/mind/wavelets/",
)

FORBIDDEN_TRACKED_FILES = {
    "scripts/build_manifolds.py",
    "scripts/compute_drift.py",
    "src/mind/evaluation/baselines.py",
}

FORBIDDEN_CODE_TOKENS = (
    "outputs/round2_2026_04",
    "round2_2026_04",
    "mind.drift",
    "mind.manifolds",
    "mind.wavelets",
    "scripts/build_manifolds.py",
    "scripts/compute_drift.py",
    "PyWavelets",
    "pywt",
)

SCANNED_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_ALLOWLIST_PREFIXES = (
    "docs/_archive/",
    "docs/plans/",
    "legacy/",
)

CONTENT_ALLOWLIST_FILES = {
    "tests/stage0/test_master_cleanup.py",
}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def test_master_does_not_track_v1_drift_manifold_wavelet_paths() -> None:
    tracked_files = _tracked_files()

    forbidden = [
        path
        for path in tracked_files
        if path in FORBIDDEN_TRACKED_FILES
        or any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PATH_PREFIXES)
    ]

    assert forbidden == []


def test_master_text_files_do_not_reference_v1_code_paths_outside_legacy_docs() -> None:
    offenders: list[str] = []
    for tracked_path in _tracked_files():
        if tracked_path in CONTENT_ALLOWLIST_FILES:
            continue
        if tracked_path.startswith(CONTENT_ALLOWLIST_PREFIXES):
            continue
        path = REPO_ROOT / tracked_path
        if path.suffix not in SCANNED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        matches = [token for token in FORBIDDEN_CODE_TOKENS if token in text]
        if matches:
            offenders.append(f"{tracked_path}: {', '.join(matches)}")

    assert offenders == []
