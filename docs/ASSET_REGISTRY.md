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

The final Experiment 1 status is `passed` only when all 16 registered assets are verified. If any asset has another status, the final status is `blocked`.

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

## Wrapper Batch Status

The normalized smoke source files now exist under `outputs/stage0/normalized`. The required smoke inputs for the scoped asset smoke runs are:

- `outputs/stage0/normalized/pope/popular.jsonl`
- `outputs/stage0/normalized/repope/popular.jsonl`
- `outputs/stage0/normalized/dash-b/all.jsonl`

If any file is missing, `scripts/asset_smoke_extract.py` writes blocked rows for every model/dataset pair and exits before model loading.

Batch 1 verified these assets for local smoke extraction and hidden-state validation:

- `qwen2.5-vl-7b`
- `qwen3.5-4b`
- `qwen3.5-9b`
- `internvl3.5-8b`
- `qwen3-vl-8b`
- `llava-onevision-qwen2-7b-ov-hf`

Batch 2 verified these assets for local smoke extraction and hidden-state validation:

- `gemma-3-4b-it`
- `gemma-3-12b-it`
- `phi-3.5-vision-instruct`

Batch 3 verified these assets for local smoke extraction and hidden-state validation:

- `glm-4.6v-flash`
- `minicpm-v-2_6`
- `minicpm-v-4_5`

Before the final closure batch, these 12 assets were verified:

- `glm-4.6v-flash`
- `minicpm-v-2_6`
- `minicpm-v-4_5`
- `qwen2.5-vl-7b`
- `qwen3-vl-8b`
- `qwen3.5-4b`
- `qwen3.5-9b`
- `internvl3.5-8b`
- `llava-onevision-qwen2-7b-ov-hf`
- `gemma-3-4b-it`
- `gemma-3-12b-it`
- `phi-3.5-vision-instruct`

The final closure batch targets these existing blockers and the new Gemma 4 asset:

- `gemma-4-12b-it`
- `phi-4-multimodal-instruct`
- `molmo-7b-d-0924`
- `llava-v1.5-7b`

After the latest temporary repair run, the 12 previously verified assets remained verified in the main `mind-py311` asset pipeline. The remaining unresolved assets are handled only by their current explicit statuses:

- `gemma-4-12b-it`
- `phi-4-multimodal-instruct`
- `molmo-7b-d-0924`
- `llava-v1.5-7b`

Asset verification only checks local asset metadata, deterministic smoke extraction, and hidden-state extraction contracts. It is not scientific validation of a model or of MIND. The final Experiment 1 status remains `blocked` until all 16 registered assets are verified.

## Gemma 4 Asset Notes

`gemma-4-12b-it` is registered as a separate `gemma4` family. It is not silently handled by the Gemma3 wrapper. The registry records `google/gemma-4-12B-it` as the Hugging Face model ID and `/home/team/lvshuyang/Models/gemma-4-12B-it` as the desired local path.

The local Gemma 4 path now exists. The core `model.safetensors` file was moved into `/home/team/lvshuyang/Models/gemma-4-12B-it`, and the temporary repair script reports the local asset as `already_present`. The local config is `model_type=gemma4_unified` with `architectures=["Gemma4UnifiedForConditionalGeneration"]`, `processor_class=Gemma4UnifiedProcessor`, and nested `Gemma4UnifiedImageProcessor` metadata.

The main production asset pipeline still does not verify Gemma 4. It currently marks the asset `unsupported_by_policy` because the production MoE detector flags `text_config.num_experts` even when the local value is null, and the installed Transformers path does not provide exact `gemma4_unified` wrapper support. That production path was not changed in the temporary repair task.

## Remaining Blockers

- `gemma-4-12b-it`: local asset is complete enough for file integrity checks, but the main pipeline remains `unsupported_by_policy` until production validation and wrapper support handle `gemma4_unified` exactly.
- `phi-4-multimodal-instruct`: Python package blockers have been repaired. Normal smoke extraction now blocks on `RuntimeError: Tensor.item() cannot be called on meta tensors`, which is a loading/runtime issue, not a missing package issue.
- `molmo-7b-d-0924`: the main `mind-py311` pipeline remains blocked because the local class lacks `_extract_generation_mode_kwargs` under that Transformers generation API. A separate `mind-molmo-py311` environment with Transformers 4.57.1 verified Molmo smoke extraction and hidden-state validation under `outputs/assets_molmo_tf457`.
- `llava-v1.5-7b`: the registry now points to the complete HF 7B local asset at `/home/team/lvshuyang/Models/llava-1.5-7b-hf`. The 13B HF asset is also structurally complete but is not registered. The main pipeline remains `unsupported_by_wrapper` because no Experiment 1 LLaVA-v1.5 wrapper is implemented.

## Temporary Repair Attempt for Remaining Blockers

Experiment 1.7 keeps all temporary repair code under `tmp/asset_repair` so it is separate from the production wrapper path. This task did not modify `src/mind/models/wrappers.py`, `src/mind/models/factory.py`, `src/mind/models/asset_validation.py`, or `src/mind/models/registry.py`.

The temporary repair scripts write diagnostic reports under `outputs/assets/repair`, but those reports are not the verification authority. The normal `scripts/asset_audit.py`, `scripts/asset_smoke_extract.py`, and `scripts/asset_validate_hidden_states.py` pipeline remains the only path that can mark a model verified.

Repair outcomes:

- `gemma-4-12b-it`: local asset is present after moving the uploaded `model.safetensors`; temporary file checks pass.
- `phi-4-multimodal-instruct`: `peft` and the later missing package blockers are installed; the remaining blocker is the meta-tensor runtime error.
- `molmo-7b-d-0924`: verified in the separate `mind-molmo-py311` environment through `tmp/asset_repair/run_molmo_tf457_asset_pipeline.py`. The main `mind-py311` pipeline remains blocked.
- `llava-v1.5-7b`: local HF 7B asset is complete and is now the registered path; no metadata was copied from LLaVA-OneVision. The remaining blocker is wrapper coverage.

Candidates to remove from the main model panel if no further repair is approved:

- `gemma-4-12b-it`
- `phi-4-multimodal-instruct`
- `llava-v1.5-7b`

`molmo-7b-d-0924` does not need removal if the project accepts separate-environment extraction for this asset. The produced hidden-state tensors and sidecars are normal saved outputs and are not tied to the Transformers runtime after extraction.
