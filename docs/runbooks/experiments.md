# Experiment Runbook

This runbook follows the current staged plan for MIND on the available machine.

## Stage 1: Environment Check

```bash
make env
make verify-env
make test
```

Expected result:

- imports succeed
- 4 GPUs are visible
- the unit suite passes

## Stage 2: Prepare Benchmarks

Normalize each POPE subset into a canonical JSONL file.

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/prepare_data.py \
  normalize-pope \
  --source data/pope/popular.jsonl \
  --output outputs/normalized/pope/popular.jsonl \
  --subset popular \
  --split val
```

For RePOPE, keep the raw override file and apply it at evaluation time through `scripts/evaluate.py --label-overrides ...`.

For H-POPE, only continue if the benchmark assets are publicly available in `data/hpope`.

## Stage 3: Reference Candidates

Build reference image candidates from MSCOCO train annotations.

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/prepare_data.py \
  build-reference \
  --instances-json data/coco/annotations/instances_train2017.json \
  --output outputs/reference_candidates/coco_train_candidates.json \
  --allowed-object dog \
  --allowed-object bus
```

This step only builds the candidate list. The grounded subset still has to be filtered by model-correct samples once extraction runs.

## Stage 4: Hidden-State Extraction

Run pre-generation extraction on normalized records.

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/extract_eval_states.py \
  --records outputs/normalized/pope/popular.jsonl \
  --model-config configs/models/qwen3_vl_8b.yaml \
  --output-root outputs/cache \
  --dataset-name pope \
  --split popular \
  --device cuda \
  --selected-layers 16 \
  --shard-size 64
```

Run this once for reference records and once for evaluation records.

## Stage 5: Manifold Artifacts

The current repo already fixes the output path convention.

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/build_manifolds.py \
  --output-root outputs/reference_banks \
  --model-name qwen3-vl-8b \
  --object-name dog \
  --layer-index 8
```

The actual manifold fit runs in Python through `mind.manifolds`.

## Stage 6: Detector Features

The detector feature path is:

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/train_detector.py \
  --output-root outputs/features \
  --experiment-name smoke-qwen3-vl \
  --split popular
```

The actual feature content should include:

- raw drift values
- approximation energy
- detail energy by level
- max drift
- mean drift
- peak layer index

## Stage 7: Evaluation

Evaluate raw predictions or feature-derived detector outputs.

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/evaluate.py \
  --input-path outputs/reports/raw/predictions.parquet \
  --output-root outputs/reports \
  --experiment-name smoke-qwen3-vl
```

For RePOPE relabeling:

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/evaluate.py \
  --input-path outputs/reports/raw/predictions.parquet \
  --output-root outputs/reports \
  --experiment-name smoke-qwen3-vl-repope \
  --label-overrides data/repope/relabels.jsonl
```

## Stage 8: Plotting

Resolve the canonical output paths and write plots into `outputs/plots/<experiment>/`.

```bash
/home/d7049/miniconda3/envs/mind-py311/bin/python scripts/plot_results.py \
  --output-root outputs/plots \
  --experiment-name smoke-qwen3-vl
```

The plot module currently supports:

- drift curve comparison
- wavelet heatmap
- ROC curve
- ablation bars

## Recommended Execution Order

1. Smoke run with `configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml`
2. Medium run with `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`
3. Main POPE run with `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`
4. Ablations with `configs/experiments/ablations/qwen3_vl_8b_popular.yaml`
5. RePOPE relabel evaluation on the same prediction table
6. H-POPE only after the public assets are confirmed locally

## Known Blocker

Hugging Face access was unavailable during the current session. The repo is ready for model loading, but real end-to-end experiments require either:

- restored access to `huggingface.co`
- or `HF_ENDPOINT=https://hf-mirror.com` with the same model ids
