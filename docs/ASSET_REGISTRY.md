# Asset Registry

Experiment 1 uses `configs/assets/model_assets.yaml` as the canonical local model asset registry.

The canonical output root is `outputs/assets`. The task does not write to `outputs/stage0`; it only checks that the required normalized smoke files exist before model loading.

## Contract

Each asset has one of these statuses:

- `verified`
- `blocked`
- `unsupported_by_policy`
- `unsupported_by_wrapper`
- `failed_validation`

The final Experiment 1 status is `passed` only when all 15 requested assets are verified. If any asset has another status, the final status is `blocked`.

## Policy Checks

The audit checks local metadata only by default. It rejects MoE indicators, unresolved thinking controls, non-single-image VLM families, non-deterministic generation settings, unsupported wrappers, unresolved layer counts, unresolved hidden dimensions, and unknown hidden-state offsets.

The smoke prompt uses each normalized record question unchanged. If a model needs a chat template, the registry records the prompt template ID and how the question is inserted.

## Outputs

The audit writes:

- `outputs/assets/asset_inventory.csv`
- `outputs/assets/model_capability_matrix.csv`
- `outputs/assets/model_asset_manifest.json`
- `outputs/assets/unsupported_models.json`
- `outputs/assets/blocked_models.json`

The smoke and validation steps write:

- `outputs/assets/smoke_extraction_report.csv`
- `outputs/assets/hidden_state_validation_report.csv`
- `outputs/assets/validation_checksums.json`
- `outputs/assets/asset_completion_summary.json`
- `outputs/assets/ASSET_COMPLETION_REPORT.md`

## Current Blocker

The required smoke inputs are:

- `outputs/stage0/normalized/pope/popular.jsonl`
- `outputs/stage0/normalized/repope/popular.jsonl`
- `outputs/stage0/normalized/dash-b/all.jsonl`

If any file is missing, `scripts/asset_smoke_extract.py` writes blocked rows for every model/dataset pair and exits before model loading.
