# Experiment Runbook

This runbook matches the repo state that has actually been verified on the recovered `3 x RTX 3090 24GB` machine in this session.

## 1. Environment

```bash
make env
make verify-env
make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
make test
```

Verified result in this session:

- `scripts/verify_env.py` currently sees the 3 usable RTX 3090 GPUs on this machine
- Hugging Face config and processor loading worked through `HF_ENDPOINT=https://hf-mirror.com`
- the full unit and integration suite passed
- the project now uses one canonical environment only: `mind-py311`
- the answer prompt was tightened to one-word yes/no output and the extraction scripts now run with `--max-new-tokens 1`

## 2. Normalize POPE and RePOPE

POPE:

```bash
for subset in random popular adversarial; do
  conda run --no-capture-output -n mind-py311 python scripts/prepare_data.py \
    normalize-pope \
    --source data/pope/${subset}.jsonl \
    --output outputs/normalized/pope/${subset}.jsonl \
    --subset ${subset} \
    --split val
done
```

RePOPE:

```bash
for subset in random popular adversarial; do
  conda run --no-capture-output -n mind-py311 python scripts/prepare_data.py \
    normalize-pope \
    --source data/repope/${subset}.jsonl \
    --output outputs/normalized/repope/${subset}.jsonl \
    --subset ${subset} \
    --split val \
    --source-dataset repope
done
```

These normalized files already exist locally.

## 3. Build Reference Candidates

This step needs MSCOCO train annotations.

```bash
conda run --no-capture-output -n mind-py311 python scripts/prepare_data.py \
  build-reference \
  --instances-json data/coco/annotations/instances_train2017.json \
  --output outputs/reference_candidates/coco_train_candidates.json \
  --allowed-objects-from outputs/normalized/pope/popular.jsonl \
  --max-images-per-object 64
```

Notes:

- use only object names that appear in the target POPE subset
- the current repo keeps reference candidates separate from evaluation records
- the final grounded reference bank is built from cached hidden states, not directly from this JSON

## 4. Plan Experiment Commands

Preview the commands for a preset without running them:

```bash
conda run --no-capture-output -n mind-py311 python scripts/run_experiment.py \
  --config configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml \
  --stages prepare,cache_reference,extract_eval,build_manifolds,compute_drift,train_detector,evaluate,plot
```

Use `--execute` only after the required assets are in place.

## 5. Manual Stage Commands

### Reference cache

```bash
conda run --no-capture-output -n mind-py311 python scripts/cache_reference_states.py \
  --references outputs/reference_candidates/coco_train_candidates.json \
  --image-root data/coco/train2017 \
  --model-config configs/models/qwen3_vl_8b.yaml \
  --output-root outputs/cache \
  --dataset-name pope-reference-64 \
  --split train \
  --device cuda \
  --selected-layers 16 \
  --layer-range middle \
  --shard-size 8 \
  --batch-size 8 \
  --max-new-tokens 1
```

### Evaluation cache

```bash
conda run --no-capture-output -n mind-py311 python scripts/extract_eval_states.py \
  --records outputs/normalized/pope/popular.jsonl \
  --model-config configs/models/qwen3_vl_8b.yaml \
  --output-root outputs/cache \
  --dataset-name pope \
  --split popular \
  --image-root data/coco/val2014 \
  --device cuda \
  --selected-layers 16 \
  --layer-range middle \
  --shard-size 8 \
  --batch-size 8 \
  --max-new-tokens 1
```

### Build manifolds

```bash
conda run --no-capture-output -n mind-py311 python scripts/build_manifolds.py \
  --reference-cache outputs/cache/qwen3-vl-8b/pope-reference-64/train \
  --output-root outputs/reference_banks \
  --model-name qwen3-vl-8b
```

The reference cache input can be a single shard or a shard directory.

### Compute drift and wavelet features

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_drift.py \
  --cache-path outputs/cache/qwen3-vl-8b/pope/popular \
  --reference-root outputs/reference_banks \
  --model-name qwen3-vl-8b \
  --output-root outputs/features \
  --experiment-name medium-qwen3-vl-8b-popular \
  --split popular
```

The cache input can be a single shard or a shard directory.

### Train detector

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/features/medium-qwen3-vl-8b-popular/popular.parquet \
  --output-root outputs/reports \
  --experiment-name medium-qwen3-vl-8b-popular
```

This command performs its own reproducible stratified split when `--eval-path` is omitted.

### Evaluate

```bash
conda run --no-capture-output -n mind-py311 python scripts/evaluate.py \
  --input-path outputs/reports/medium-qwen3-vl-8b-popular/results.csv \
  --output-root outputs/reports \
  --experiment-name medium-qwen3-vl-8b-popular
```

### RePOPE relabel pass

```bash
conda run --no-capture-output -n mind-py311 python scripts/evaluate.py \
  --input-path outputs/reports/medium-qwen3-vl-8b-popular/results.csv \
  --label-overrides outputs/normalized/repope/popular.jsonl \
  --output-root outputs/reports \
  --experiment-name medium-qwen3-vl-8b-popular-repope
```

### Plotting

```bash
conda run --no-capture-output -n mind-py311 python scripts/plot_results.py \
  --features-path outputs/features/medium-qwen3-vl-8b-popular/popular.parquet \
  --results-path outputs/reports/medium-qwen3-vl-8b-popular/results.csv \
  --ablation-path outputs/reports/medium-qwen3-vl-8b-popular/ablations.csv \
  --output-root outputs/plots \
  --experiment-name medium-qwen3-vl-8b-popular
```

### Baselines and ablations

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py \
  --features-path outputs/features/medium-qwen3-vl-8b-popular/popular.parquet \
  --cache-path outputs/cache/qwen3-vl-8b/pope/popular \
  --reference-root outputs/reference_banks \
  --model-name qwen3-vl-8b \
  --output-root outputs/reports \
  --experiment-name medium-qwen3-vl-8b-popular
```

### Cross-family InternVL popular run

The stable setting on the current machine is:

- `CUDA_VISIBLE_DEVICES=0,1,2`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `HF_ENDPOINT=https://hf-mirror.com`
- `--batch-size 8`
- `--shard-size 8`
- `--max-new-tokens 1`

Use the same stage commands as above, but swap the model config to `configs/models/internvl3_5_8b.yaml` and the model name to `internvl3.5-8b`.

Completed output roots from this session:

- `outputs/reports/cross-internvl3.5-8b-popular/`
- `outputs/reports/cross-internvl3.5-8b-popular-repope/`
- `outputs/plots/cross-internvl3.5-8b-popular/`

## 6. Intended Stage Order

1. Smoke: `configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml`
2. Medium: `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`
3. Main: `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`
4. Ablations: `configs/experiments/ablations/qwen3_vl_8b_popular.yaml`
5. RePOPE relabel evaluation on the same saved predictions
6. InternVL cross-family comparison
7. H-POPE only if the public files become available

## 7. Current External Blockers

- the public COCO assets are present locally
- `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`, and `OpenGVLab/InternVL3_5-8B-HF` config or processor checks were verified through `HF_ENDPOINT=https://hf-mirror.com`
- the fourth GPU remains faulty and unavailable, but the sequential `3 x RTX 3090` path is working
- H-POPE public assets were not found in a directly usable release package
