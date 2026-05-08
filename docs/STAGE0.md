# MIND Stage 0

Stage 0 is the data audit, split, and cache entry point for the MIND pipeline. It prepares audited records, deterministic grouped splits, full-layer hidden-state cache shards, manifests, and one run log under `outputs/stage0`. Stage 0 is complete only when POPE, RePOPE, and DASH-B are cached for both primary models.

## What Stage 0 Does

- Loads normalized extraction records for cache work.
- Audits discovered POPE, RePOPE, and DASH-B records, using normalized files when present and raw local files for audit-only coverage when normalized files are missing.
- Audits dataset fields, object names, label balance, and sample overlap before cache work starts.
- Creates deterministic grouped splits by `image_id`.
- Extracts pre-generation full-layer hidden states for each configured model and record.
- Writes audit CSVs, cache shards, cache sidecars, manifests, and a run log under `outputs/stage0`.

## What Stage 0 Does Not Do

- Stage A is not started. Stage 0 does not implement Stage A, Stage B, Stage C, Stage D, or Stage E training.
- It does not use drift, manifold, or wavelet features as the main path.
- It does not sample 16 layers or choose a layer subset.
- It does not depend on the old MIND path for the main path.

## Branch Migration Status

The old work is frozen outside `master`. The current branch is the Stage 0 line and keeps no old runtime modules.

## Output Contract

Each Stage 0 run writes exactly this output surface:

```text
outputs/stage0/
  audit/
    dataset_audit.csv
    object_name_audit.csv
    label_balance.csv
    sample_overlap_audit.csv
  cache/
    <model_name>/
      <dataset_name>/
        <split_or_subset>/
          shard-00000.pt
          shard-00000.pt.json
  manifests/
    cache_manifest.json
    split_manifest_<dataset_name>_<subset>.json
    split_manifest.json
    stage0_summary.json
  logs/
    stage0_run.log
```

The cache shard stores full-layer hidden states. The `.pt.json` sidecar stores shard metadata that can be checked without loading tensor payloads. `cache_manifest.json` records shard locations and row counts. Per-dataset `split_manifest_<dataset_name>_<subset>.json` files record grouped split membership. Top-level `split_manifest.json` is an index/summary of those split manifests. `stage0_summary.json` records the run config, dataset counts, model counts, audit status, and output paths.

## Dataset Audit Contract

The audit must run before cache extraction. RePOPE and DASH-B are not optional for Stage 0 completion. The complete run must fail closed when any requested POPE, RePOPE, or DASH-B extraction dataset is not ready.

Required record fields:

- `sample_id`
- `image_id`
- `image_path`
- `question`
- `label`
- `object_name`
- `source_dataset`
- `split`
- `subset`

The audit CSVs must cover row count, label counts, unique image count, duplicate sample IDs, missing image paths, null required fields, object-name coverage, object-name frequency, label balance by dataset and subset, sample overlap across configured datasets, and the source record path for each dataset.

Full extraction does not accept raw POPE files. If a requested full-run subset only exists under `data/pope`, normalize it first with `scripts/prepare_data.py`.

## Cache Extraction Contract

Stage 0 extracts pre-generation hidden states with:

- `dtype: float16`
- `max_new_tokens: 1`
- `token_index: -1`
- all model layers retained

Each cache row must preserve dataset identity, source row identity, image group, prompt text, answer label, object name, split group, and model identity. Stage A will consume the full-layer hidden states as its primary input, but Stage A work has not started.

## Split Contract

The split is deterministic and grouped.

- `seed: 20260506`
- `group_key: image_id`
- `groups: [encoder_train, bank, cal, test]`
- `ratios: [0.50, 0.20, 0.10, 0.20]`

The ratio order is `encoder_train`, `bank`, `cal`, `test`. All rows with the same `image_id` must stay in the same split group. Split assignment is based on sorted unique groups plus the configured seed.

## Smoke Run Command

This is the smoke command:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage0_run.py \
  --output-root outputs/stage0 \
  --models qwen3-vl-8b internvl3.5-8b \
  --datasets pope \
  --subsets popular \
  --smoke-limit 8 \
  --device cuda:0 \
  --dtype float16
```

## Full Run Command

This is the config-driven full-run command:

```bash
HF_HUB_DISABLE_XET=1 conda run --no-capture-output -n mind-py311 python scripts/stage0_run.py \
  --config configs/stage0/stage0_complete.yaml \
  --device cuda:0
```

The complete config must include both primary models, POPE popular/random/adversarial, RePOPE popular/random/adversarial, and DASH-B all. DASH-B and RePOPE are required, not optional. For long runs, mount the command in `tmux` and keep output under `outputs/stage0/logs`.

## Stage 0 Completion Gate

- The smoke CLI completes on POPE popular with both primary models and `smoke_limit: 8`. This is a wiring check, not completion.
- The config-driven full run completes for both primary models and every required dataset: POPE popular/random/adversarial, RePOPE popular/random/adversarial, and DASH-B all.
- Dataset audit CSVs have no missing required fields.
- Splits have no `image_id` leakage across split groups.
- Cache manifests match audited row counts for each model and dataset.
- Cache tensors retain full-layer hidden states with no 16-layer sampling.
- Stage 0 output lives only under `outputs/stage0`.
- Stage A remains not started until this gate is satisfied.
