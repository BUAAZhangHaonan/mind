# MIND v2 Stage 0

Stage 0 is the data audit, split, and cache entry point for the MIND v2 pipeline. It prepares audited records, deterministic grouped splits, full-layer hidden-state cache shards, manifests, and one run log under `outputs/v2_stage0`.

## What Stage 0 Does

- Loads normalized dataset records from configured JSONL paths.
- Audits dataset fields, object names, label balance, and sample overlap before cache work starts.
- Creates deterministic grouped splits by `image_id`.
- Extracts pre-generation full-layer hidden states for each configured model and record.
- Writes audit CSVs, cache shards, cache sidecars, manifests, and a run log under `outputs/v2_stage0`.

## What Stage 0 Does Not Do

- It does not implement Stage A, Stage B, Stage C, Stage D, or Stage E training.
- It does not use v1 drift, manifold, or wavelet features as the v2 main path.
- It does not sample 16 layers or choose a layer subset.
- It does not depend on the v1 MIND path for the v2 main path.

## Branch Migration Status

The v1 work is frozen on the local branch `v1` and tag `v1-freeze-before-v2` at commit `81e3444`. The remote push is deferred until v2 checks pass. The v2 path starts from the new `configs/v2`, `docs/v2`, and `src/mind/trajectory` surface.

## Output Contract

Each Stage 0 run writes exactly this output surface:

```text
outputs/v2_stage0/
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
    split_manifest.json
    stage0_summary.json
  logs/
    stage0_run.log
```

The cache shard stores full-layer hidden states. The `.pt.json` sidecar stores shard metadata that can be checked without loading tensor payloads. `cache_manifest.json` records shard locations and row counts. `split_manifest.json` records grouped split membership. `stage0_summary.json` records the run config, dataset counts, model counts, audit status, and output paths.

## Dataset Audit Contract

The audit must run before cache extraction. It must fail closed when a required field is missing.

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

## Cache Extraction Contract

Stage 0 extracts pre-generation hidden states with:

- `dtype: float16`
- `max_new_tokens: 1`
- `token_index: -1`
- all model layers retained

Each cache row must preserve dataset identity, source row identity, image group, prompt text, answer label, object name, split group, and model identity. Stage A consumes the full-layer hidden states as its primary input.

## Split Contract

The split is deterministic and grouped.

- `seed: 20260506`
- `group_key: image_id`
- `groups: [encoder_train, bank, cal, test]`
- `ratios: [0.50, 0.20, 0.10, 0.20]`

The ratio order is `encoder_train`, `bank`, `cal`, `test`. All rows with the same `image_id` must stay in the same split group. Split assignment is based on sorted unique groups plus the configured seed.

## Smoke Run Command

This is the required smoke command surface once the Stage 0 runner exists:

```bash
python scripts/v2/stage0_run.py \
  --config configs/v2/stage0/qwen_pope_popular_smoke.yaml
```

Use the same shape for the InternVL smoke run:

```bash
python scripts/v2/stage0_run.py \
  --config configs/v2/stage0/internvl_pope_popular_smoke.yaml
```

## Full Run Command

This is the required full-run command surface once the Stage 0 runner exists:

```bash
python scripts/v2/stage0_run.py \
  --config configs/v2/stage0/qwen_internvl_stage0_full.yaml \
  --full-run
```

The full config must include POPE subsets in this order: popular, random, adversarial.

## Gate Criteria For Stage A

- Both smoke configs load and complete on POPE popular with `smoke_limit: 8`.
- The full config loads with both models and POPE popular, random, and adversarial.
- Dataset audit CSVs have no missing required fields.
- Splits have no `image_id` leakage across split groups.
- Cache manifests match audited row counts for each model and dataset.
- Cache tensors retain full-layer hidden states with no 16-layer sampling.
- Stage 0 output lives only under `outputs/v2_stage0`.
