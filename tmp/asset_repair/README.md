# Temporary Asset Repair Scripts

These scripts are for Experiment 1.7 only. They inspect or repair the four remaining blocked local assets without changing the production wrapper path.

They write diagnostic reports under `outputs/assets/repair/`. Those reports do not decide final model status. The normal asset audit, smoke extraction, and hidden-state validation scripts remain the only verification authority.

## Scripts

- `repair_gemma4_download.py`: checks or downloads only `google/gemma-4-12B-it` to `/home/team/lvshuyang/Models/gemma-4-12B-it`.
- `repair_phi4_peft.py`: checks or installs only `peft`, and only when `--execute --allow-install-peft` is used.
- `repair_molmo_asset.py`: inspects Molmo local files and remote-code compatibility. It can redownload only `allenai/Molmo-7B-D-0924` if the local asset is incomplete.
- `repair_llava_v15_asset.py`: inspects LLaVA-v1.5 metadata and dependencies. It does not guess an exact model id and does not create fake processor metadata.
- `run_remaining_asset_repairs.py`: runs the four scripts together.

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
