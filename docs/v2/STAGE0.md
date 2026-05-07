# MIND v2 Stage 0

Stage 0 is the data and cache entry point for the MIND v2 exploration pipeline. This milestone only creates the contract, configs, and package marker. It does not add a runnable Stage 0 implementation yet.

## What Stage 0 Does

- Loads normalized dataset records from the configured JSONL paths.
- Audits each dataset before any cache work starts.
- Creates deterministic grouped splits by image.
- Extracts full-layer hidden states for each configured model and record.
- Writes a manifest, audit files, split files, cache shards, and cache indexes under `outputs/v2_stage0`.

## What Stage 0 Does Not Do

- It does not train a detector.
- It does not build drift, manifold, or wavelet features.
- It does not sample 16 layers or choose a layer subset.
- It does not run Stage A, B, C, D, or E logic.
- It does not depend on the v1 MIND path.

## Branch Migration Status

The v1 work is frozen on the local branch `v1` and tag `v1-freeze-before-v2` at HEAD `81e3444`. The v2 path starts from the new `configs/v2`, `docs/v2`, and `src/mind/trajectory` surface. Legacy packages remain available for v1 reproduction, but they are not the v2 main path.

## Output Contract

Each Stage 0 run writes to:

```text
outputs/v2_stage0/<run-name>/
  manifest.yaml
  dataset_audit.json
  splits.jsonl
  cache_index.jsonl
  cache/<model-name>/<dataset-name>/shard-00000.pt
```

The manifest records the config path, model config paths, dataset record paths, generation settings, split settings, and output paths. The cache index maps each dataset row to its cache shard, model, dataset, split, group key, and row offset.

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

The audit report records row count, label counts, unique image count, duplicate sample IDs, missing image paths, null required fields, and the source record path for each dataset.

## Cache Extraction Contract

Stage 0 extracts pre-generation hidden states with:

- `dtype: float16`
- `max_new_tokens: 1`
- `token_index: -1`
- all model layers retained

Each cache row must preserve the dataset identity, source row identity, image group, prompt text, answer label, object name, and model identity. Stage A consumes the full-layer hidden states as its primary input.

## Split Contract

The split is deterministic and grouped.

- `seed: 20260506`
- `group_key: image_id`
- `ratios: [0.50, 0.20, 0.10, 0.20]`

The ratio order is `bank`, `train`, `validation`, `test`. All rows with the same `image_id` must stay in the same split. Split assignment is based on sorted unique groups plus the configured seed.

## Smoke Run Command

This is the intended command surface once the Stage 0 runner exists:

```bash
conda run --no-capture-output -n mind-py311 python -m mind.trajectory.stage0 \
  --config configs/v2/stage0/qwen_pope_popular_smoke.yaml
```

Use `configs/v2/stage0/internvl_pope_popular_smoke.yaml` for the InternVL smoke run.

## Full Run Command

This is the intended full-run command surface once the Stage 0 runner exists:

```bash
conda run --no-capture-output -n mind-py311 python -m mind.trajectory.stage0 \
  --config configs/v2/stage0/qwen_internvl_stage0_full.yaml
```

## Gate Criteria For Stage A

- Both smoke configs load and complete on POPE popular with `smoke_limit: 8`.
- The full config loads with both models and all three POPE subsets.
- Dataset audit reports have no missing required fields.
- Splits have no `image_id` leakage across split names.
- Cache indexes match audited row counts for each model and dataset.
- Cache tensors retain full-layer hidden states with no 16-layer sampling.
- Stage 0 output lives only under `outputs/v2_stage0`.
