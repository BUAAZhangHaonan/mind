# Temporary Asset Repair Scripts

These scripts are for Experiment 1.7 only. They inspect or repair the four remaining blocked local assets without changing the production wrapper path.

They write diagnostic reports under `outputs/assets/repair/`. Those reports do not decide final model status. The normal asset audit, smoke extraction, and hidden-state validation scripts remain the only verification authority.

## Scripts

- `repair_gemma4_download.py`: checks or downloads only `google/gemma-4-12B-it` to `/home/team/lvshuyang/Models/gemma-4-12B-it`.
- `repair_phi4_peft.py`: checks or installs only `peft`, and only when `--execute --allow-install-peft` is used.
- `repair_molmo_asset.py`: inspects Molmo local files and remote-code compatibility. It can redownload only `allenai/Molmo-7B-D-0924` if the local asset is incomplete.
- `repair_llava_v15_asset.py`: inspects LLaVA-v1.5 metadata and dependencies. It does not guess an exact model id and does not create fake processor metadata.
- `run_remaining_asset_repairs.py`: runs the four scripts together.
- `run_molmo_tf457_asset_pipeline.py`: runs the normal smoke and validation scripts for Molmo inside the separate `mind-molmo-py311` environment. The only compatibility change is a process-local import alias for `AutoModelForMultimodalLM`; production wrappers are not edited.

## Safety

Default mode is dry-run. Use `--execute` for any asset or environment change.

The scripts must not edit:

- `src/mind/models/wrappers.py`
- `src/mind/models/factory.py`
- `src/mind/models/asset_validation.py`
- `src/mind/models/registry.py`

## Commands

Dry-run all checks:

```bash
conda run --no-capture-output -n mind-py311 python tmp/asset_repair/run_remaining_asset_repairs.py --dry-run --output-root outputs/assets/repair
```

Explicit repair attempt:

```bash
conda run --no-capture-output -n mind-py311 python tmp/asset_repair/run_remaining_asset_repairs.py --execute --repair-gemma4 --repair-phi4 --repair-molmo --repair-llava-v15 --allow-install-peft --output-root outputs/assets/repair
```

Do not use `--allow-install-tokenizer-deps` unless the exact LLaVA-v1.5 dependency need and model id are known.

Molmo separate-env smoke and validation:

```bash
conda run --no-capture-output -n mind-molmo-py311 python tmp/asset_repair/run_molmo_tf457_asset_pipeline.py --execute --pipeline-output-root outputs/assets_molmo_tf457 --output-root outputs/assets/repair
```

Molmo outputs from this runner are separate from the main `outputs/assets` root. The extracted tensors are ordinary saved tensors and sidecars; they are not tied to the runtime Transformers version once produced.

## Current Repair Results

- `gemma-4-12b-it`: local files are present. The uploaded `model.safetensors` was moved into `/home/team/lvshuyang/Models/gemma-4-12B-it`, and the temporary repair check reports `already_present`.
- `phi-4-multimodal-instruct`: package blockers are repaired. The main smoke pipeline now blocks on `RuntimeError: Tensor.item() cannot be called on meta tensors`.
- `molmo-7b-d-0924`: the main `mind-py311` pipeline remains blocked by the local generation API, but the separate `mind-molmo-py311` runner verifies Molmo under `outputs/assets_molmo_tf457`.
- `llava-v1.5-7b`: the registered local path now points to the complete HF 7B asset at `/home/team/lvshuyang/Models/llava-1.5-7b-hf`; the remaining blocker is missing production wrapper support.
