# Full-Cache Asset Surface

Experiment 2 builds the reusable full-cache asset surface for the final model panel. This is an extraction and validation artifact layer only. It is not Stage A, it is not training, and it does not make scientific validation claims.

## Closeout Artifacts

The authoritative full-cache manifest and ledger are generated under:

- `outputs/full_cache/manifests/unified_full_cache_manifest.json`
- `outputs/full_cache/manifests/unified_full_cache_manifest.csv`
- `outputs/full_cache/manifests/extraction_ledger.csv`

The human-readable closeout report is:

- `outputs/full_cache/reports/FULL_CACHE_SUMMARY.md`

These files are ignored by git through the repository-wide `outputs/` rule, so this document records their canonical paths.

## Final Coverage

The closeout run validated all 16 panel models. Each model covers:

- POPE: `popular`, `random`, `adversarial`
- RePOPE: `popular`, `random`, `adversarial`
- DASH-B: `all`

Final generated summary:

- total panel models: 16
- total full-cache records: 317,872
- per-model records: 19,867
- per-model shards: 158

Route status counts:

- `accepted_existing_stage0`: 2
- `accepted_existing_separate_env`: 1
- `extracted_main_env`: 11
- `extracted_separate_env`: 2

Gemma4, Molmo, and Qwen3.5-4B use separate-environment routes where required by local Transformers compatibility. The unified manifest records the physical cache roots, so downstream code should read the manifest instead of guessing cache paths.
