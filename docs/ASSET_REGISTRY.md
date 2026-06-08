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

Gemma 4 Unified has no independent vision tower, so its image wiring check must use the image-sensitivity canary, deterministic repeat check, and full-layer pre-generation hidden-state validation. Thinking must be explicitly disabled with `enable_thinking=False` or an equivalent recorded setting. The production path now has an explicit `gemma4_unified` wrapper, but the current `mind-py311` Transformers runtime does not expose the required Gemma4 Unified classes, so Gemma4 remains blocked in the main pipeline.

## Final Production Integration Status

Experiment 1.8 moved the workable temporary paths into the production asset pipeline where the current `mind-py311` environment can support them.

Current target outcomes:

- `phi-4-multimodal-instruct`: verified in the main asset pipeline. The production Phi4 path uses eager attention, not Flash Attention 2. It sets `low_cpu_mem_usage=False`, avoids `device_map="auto"`, removes user-site Python packages before smoke extraction, confirms no meta tensors after load, and uses a forward-only one-token smoke path because the local remote class does not safely support the standard `generate` path in this environment.
- `llava-v1.5-7b`: verified in the main asset pipeline with the complete HF local path `/home/team/lvshuyang/Models/llava-1.5-7b-hf`. It uses its own `llava_v15` wrapper and does not copy metadata or prompt logic from LLaVA-OneVision.
- `molmo-7b-d-0924`: accepted as `verified_separate_env` from `outputs/assets_molmo_tf457`. This is distinct from main-env `verified`. It means the separate environment produced smoke and hidden-state validation artifacts that satisfy the same asset contract.
- `gemma-4-12b-it`: still blocked in the main production pipeline. The local asset is registered as `gemma4_unified`, not Gemma3, and it has no separate vision encoder. Thinking must be disabled with `enable_thinking=False`. The current `mind-py311` production environment has the explicit wrapper path, but it is missing the concrete Gemma4 Unified Transformers classes needed for local loading: `Gemma4UnifiedProcessor` and `Gemma4UnifiedForConditionalGeneration`.

Gemma4 remains a candidate for future integration, but it needs a production environment with explicit Gemma4 Unified class support before the normal smoke and hidden-state validation pipeline can verify it. The Gemma4 wrapper must rely on deterministic repeat checks, image sensitivity canary, and full-layer hidden-state validation rather than a vision-tower check.

The final Experiment 1.8 production summary has 14 main-env verified models, 1 separate-env verified model, and 1 blocked model. `verified_separate_env` is not the same as `verified`; downstream panels should label Molmo explicitly if they include it.

These are asset and extraction checks only. They do not make scientific validation claims about any model or about MIND.
