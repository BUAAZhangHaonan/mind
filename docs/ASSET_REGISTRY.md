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

`gemma-4-12b-it` is registered as a separate `gemma4_unified` candidate family. It is not silently handled by the Gemma3 wrapper. The registry records `google/gemma-4-12B-it` as the Hugging Face model ID and `/home/team/lvshuyang/Models/gemma-4-12B-it` as the desired local path.

The local Gemma 4 path now exists. The core `model.safetensors` file was moved into `/home/team/lvshuyang/Models/gemma-4-12B-it`, and the temporary repair script reports the local asset as `already_present`. The local config is `model_type=gemma4_unified` with `architectures=["Gemma4UnifiedForConditionalGeneration"]`, `processor_class=Gemma4UnifiedProcessor`, and nested `Gemma4UnifiedImageProcessor` metadata.

Gemma 4 Unified has no independent vision tower, so its image wiring check must use the image-sensitivity canary, deterministic repeat check, and full-layer pre-generation hidden-state validation. Thinking must be explicitly disabled with `enable_thinking=False` or an equivalent recorded setting. The main production asset pipeline still does not verify Gemma 4 because exact `gemma4_unified` wrapper support is not in the production path.

## Remaining Blockers

- `gemma-4-12b-it`: local asset is complete enough for file integrity checks, but the main pipeline remains `unsupported_by_policy` until production validation and wrapper support handle `gemma4_unified` exactly.
- `phi-4-multimodal-instruct`: Python package blockers have been repaired. Normal smoke extraction now blocks on `RuntimeError: Tensor.item() cannot be called on meta tensors`, which is a loading/runtime issue, not a missing package issue.
- `molmo-7b-d-0924`: the main `mind-py311` pipeline remains blocked because the local class lacks `_extract_generation_mode_kwargs` under that Transformers generation API. A separate `mind-molmo-py311` environment with Transformers 4.57.1 verified Molmo smoke extraction and hidden-state validation under `outputs/assets_molmo_tf457`.
- `llava-v1.5-7b`: the registry now points to the complete HF 7B local asset at `/home/team/lvshuyang/Models/llava-1.5-7b-hf`. The 13B HF asset is also structurally complete but is not registered. The main pipeline remains `unsupported_by_wrapper` because no Experiment 1 LLaVA-v1.5 wrapper is implemented.

## Final Temporary Repair Attempt

Experiment 1.7 keeps all temporary repair code under `tmp/asset_repair` so it is separate from the production wrapper path. This task did not modify `src/mind/models/wrappers.py`, `src/mind/models/factory.py`, `src/mind/models/asset_validation.py`, or `src/mind/models/registry.py`.

The temporary repair scripts write diagnostic reports under `outputs/assets/repair`, but those reports are not the verification authority. The normal `scripts/asset_audit.py`, `scripts/asset_smoke_extract.py`, and `scripts/asset_validate_hidden_states.py` pipeline remains the only path that can mark a model verified.

Final repair outcomes and panel decisions:

- `gemma-4-12b-it`: local Gemma 4 Unified asset is present, `model_type=gemma4_unified` is recorded, and file integrity checks pass. It remains a high-value candidate, but it is `blocked_manual_future_work` until exact production `gemma4_unified` support records `enable_thinking=False` and passes the normal smoke and validation pipeline.
- `phi-4-multimodal-instruct`: package blockers were repaired, but safe load diagnostics still hit the Phi4 Flash Attention 2 / meta-tensor loading path. The decision is `blocked_remove_from_panel` unless a later environment-level load fix is approved.
- `molmo-7b-d-0924`: accepted as `verified_separate_env` from the `mind-molmo-py311` output under `outputs/assets_molmo_tf457`. This is not main-env verification; it means the separate environment produced smoke and hidden-state validation artifacts that satisfy the same contract.
- `llava-v1.5-7b`: the registered local path is the complete HF 7B asset at `/home/team/lvshuyang/Models/llava-1.5-7b-hf`, with no metadata copied from LLaVA-OneVision. It remains `blocked_remove_from_panel` until an exact production LLaVA-v1.5 wrapper passes the normal pipeline.

Models that remain in the candidate set:

- the 12 main-env verified assets listed above
- `gemma-4-12b-it` as manual future work for the separate `gemma4_unified` family
- `molmo-7b-d-0924` only if the panel allows a `verified_separate_env` label

Removal candidates from the main model panel are `phi-4-multimodal-instruct` and `llava-v1.5-7b`. These decisions are asset and extraction decisions only. They do not make scientific validation claims about any model or about MIND.
