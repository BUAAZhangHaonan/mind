# Cleanup Inventory

This inventory was taken before any destructive move or delete in the structural migration pass.

## Source Commands

- `git status --short --ignored`
- `git ls-files --others --exclude-standard`
- `find . -maxdepth 3 -type d | sort`

## Tracked Current Stage 0 Paths To Move

| Path | Proposed action |
| --- | --- |
| old versioned script directory `stage0_audit_data.py` | Move to `scripts/stage0_audit_data.py`. |
| old versioned script directory `stage0_build_splits.py` | Move to `scripts/stage0_build_splits.py`. |
| old versioned script directory `stage0_extract_full_layer_cache.py` | Move to `scripts/stage0_extract_full_layer_cache.py`. |
| old versioned script directory `stage0_run.py` | Move to `scripts/stage0_run.py`. |
| old versioned script directory `stage0_validate_cache.py` | Move to `scripts/stage0_validate_cache.py`. |
| old versioned Stage 0 config directory | Move tracked configs to `configs/stage0/`. |
| old versioned Stage 0 document | Move to `docs/STAGE0.md`. |
| old versioned design notes document | Move to `docs/DESIGN_NOTES.md`. |
| old versioned test directory | Move tracked tests to `tests/stage0/`. |
| old versioned Stage A config placeholder | Remove. No Stage A config is created in this pass. |

## Tracked Legacy Runtime Residue

| Path | Proposed action |
| --- | --- |
| `legacy/v1/README.md` | Remove after confirming refs exist. The user already confirmed the remote branch and freeze tag. Remove `legacy/` if empty. |

## Untracked Source-Like Files

None. `git ls-files --others --exclude-standard` returned no paths.

## Untracked Generated Outputs

None outside ignored paths. `git ls-files --others --exclude-standard` returned no paths.

## Ignored Output Directories

| Path | Proposed action |
| --- | --- |
| old versioned Stage 0 output root | Move to `outputs/stage0/` if `outputs/stage0/` is absent or empty. Update path strings and stage labels in JSON metadata and sidecars only. Preserve tensor payloads and scientific metadata. Remove the old directory after a successful move. |
| `outputs/round2_2026_04/` | Keep. This is the normalized input used by current Stage 0 configs. |

## Local V1 Artifacts

No ignored or untracked local V1 artifact directory was found at max depth 3. Only the tracked `legacy/v1/README.md` residue is present.

## Local Current Stage 0 Artifacts

| Path | Proposed action |
| --- | --- |
| old versioned Stage 0 audit output directory | Move under `outputs/stage0/audit/` as part of output migration. |
| old versioned Stage 0 cache output directory | Move under `outputs/stage0/cache/` as part of output migration. |
| old versioned Stage 0 log output directory | Move under `outputs/stage0/logs/` as part of output migration. |
| old versioned Stage 0 manifest output directory | Move under `outputs/stage0/manifests/` as part of output migration. Update listed manifest JSON path strings. |

## Ignored Temporary Caches

| Path | Proposed action |
| --- | --- |
| `.pytest_cache/` | Delete after inventory. |
| `scripts/__pycache__/` | Delete after inventory. |
| old versioned script cache directory | Delete after inventory. |
| `src/mind/**/__pycache__/` | Delete after inventory. |
| `tests/unit/__pycache__/` | Delete after inventory. |
| old versioned test cache directory | Delete after inventory. |

## Ignored Local Non-Output Directories

| Path | Proposed action |
| --- | --- |
| `.vscode/` | Keep. Local editor state. |
| `data/` | Keep. Local datasets and source data. |
