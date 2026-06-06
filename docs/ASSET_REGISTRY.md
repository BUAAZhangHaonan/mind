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
- `not_attempted_due_to_dependency`

The final Experiment 1 status is `passed` only when all 15 requested assets are verified. If any asset has another status, the final status is `blocked`.

## Policy Checks

The audit checks local metadata only by default. It rejects MoE indicators, unresolved thinking controls, non-single-image VLM families, non-deterministic generation settings, unsupported wrappers, unresolved layer counts, unresolved hidden dimensions, and unknown hidden-state offsets.

The smoke prompt uses each normalized record question unchanged. If a model needs a chat template, the registry records the prompt template ID and how the question is inserted.

## Canonical Stage 0 Normalized Records

Experiment 1 smoke extraction reads canonical source records from `outputs/stage0/normalized`. These files are materialized from existing Stage 0 cache metadata with `scripts/asset_materialize_stage0_normalized.py`.

The materialization step reads the `qwen3-vl-8b` Stage 0 cache as the source and checks it against the `internvl3.5-8b` Stage 0 cache before writing records. For each shared `sample_id`, `image_id`, `image_path`, `question`, `label`, `object_name`, `source_dataset`, and `subset` must match exactly. If either model has an extra sample or any field differs, the script writes a mismatch report and does not choose one side silently.

The normalized files contain only source-record metadata. Tensor fields, logits, generated answers, parsed answers, selected layers, model names, hidden dimensions, and layer counts are excluded. These files are source records for smoke or later extraction only; they are not scientific result files.

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
