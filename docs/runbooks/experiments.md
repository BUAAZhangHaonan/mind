# Experiment Runbook

This runbook matches the repo state that has actually been verified on the current machine state in this session.

## 1. Environment

```bash
make env
make verify-env
make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
make test
```

Verified result in this session:

- `scripts/verify_env.py` and direct PyTorch checks did see `4 x RTX 3090 24GB` after the earlier recovery
- Hugging Face config and processor loading worked through `HF_ENDPOINT=https://hf-mirror.com`
- the full unit and integration suite passed
- the project now uses one canonical environment only: `mind-py311`
- the answer prompt was tightened to one-word yes/no output and the extraction scripts now run with `--max-new-tokens 1`
- if you switch branches or worktrees, run `make install` again so the editable package points at the active checkout

Current machine note for the closeout phase:

- the fresh InternVL adversarial rerun hit a recurring machine-level failure on `2026-03-31`
- observed state after the failed retry:
  - `nvidia-smi`: `Unable to determine the device handle for GPU1`
  - fresh `mind-py311` PyTorch: `torch.cuda.device_count() == 0`
- that means:
  - the CPU-only shared-bank closeout stages can still complete
  - the fresh InternVL adversarial extraction cannot complete until the GPU state is recovered again
  - the final paper-package export still waits on that missing InternVL adversarial report

## 2. Correction-Phase Reruns From Existing Popular Caches

These are the correction-phase commands that were actually run on 2026-03-30. They do not re-extract model states. They reuse the existing popular caches and rebuild only the corrected signal path and corrected evaluation protocol.

Output root used in this phase:

- `outputs/correction_phase/`

### Rebuild cleaned reference banks

Qwen:

```bash
conda run --no-capture-output -n mind-py311 python scripts/build_manifolds.py \
  --reference-cache outputs/cache/qwen3-vl-8b/pope-reference-64/train \
  --output-root outputs/correction_phase/reference_banks \
  --model-name qwen3-vl-8b \
  --k-neighbors 32
```

InternVL:

```bash
conda run --no-capture-output -n mind-py311 python scripts/build_manifolds.py \
  --reference-cache outputs/cache/internvl3.5-8b/pope-reference-64/train \
  --output-root outputs/correction_phase/reference_banks \
  --model-name internvl3.5-8b \
  --k-neighbors 32
```

### Recompute corrected features

Qwen:

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_drift.py \
  --cache-path outputs/cache/qwen3-vl-8b/pope/popular \
  --reference-root outputs/correction_phase/reference_banks \
  --model-name qwen3-vl-8b \
  --output-root outputs/correction_phase/features \
  --experiment-name correction-qwen3-vl-8b-popular \
  --split popular
```

InternVL:

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_drift.py \
  --cache-path outputs/cache/internvl3.5-8b/pope/popular \
  --reference-root outputs/correction_phase/reference_banks \
  --model-name internvl3.5-8b \
  --output-root outputs/correction_phase/features \
  --experiment-name correction-internvl3.5-8b-popular \
  --split popular
```

### Primary protocol: `image_grouped`

Qwen:

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/correction_phase/features/correction-qwen3-vl-8b-popular/popular.parquet \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-qwen3-vl-8b-popular \
  --split-strategy image_grouped \
  --num-folds 5
```

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py \
  --features-path outputs/correction_phase/features/correction-qwen3-vl-8b-popular/popular.parquet \
  --cache-path outputs/cache/qwen3-vl-8b/pope/popular \
  --reference-root outputs/correction_phase/reference_banks \
  --model-name qwen3-vl-8b \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-qwen3-vl-8b-popular \
  --split-strategy image_grouped \
  --num-folds 5
```

InternVL:

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/correction_phase/features/correction-internvl3.5-8b-popular/popular.parquet \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-internvl3.5-8b-popular \
  --split-strategy image_grouped \
  --num-folds 5
```

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py \
  --features-path outputs/correction_phase/features/correction-internvl3.5-8b-popular/popular.parquet \
  --cache-path outputs/cache/internvl3.5-8b/pope/popular \
  --reference-root outputs/correction_phase/reference_banks \
  --model-name internvl3.5-8b \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-internvl3.5-8b-popular \
  --split-strategy image_grouped \
  --num-folds 5
```

### Legacy comparison: `row`

Use the same commands with:

- `--split-strategy row`
- experiment names:
  - `correction-qwen3-vl-8b-popular-row`
  - `correction-internvl3.5-8b-popular-row`

### Secondary protocol: `object_heldout`

Use the same commands with:

- `--split-strategy object_heldout`
- `--num-folds 2`

`2` folds were used because it was the largest shared valid setting across both model families that preserved both classes in every held-out fold.

### Correction-phase summary outputs

- `outputs/correction_phase/reports/correction_summary.csv`
- `outputs/correction_phase/plots/correction_summary_protocols.png`

## 2A. Paper-Closeout Follow-up

These are the additional closeout commands layered on top of the correction phase.

### RePOPE relabel on the corrected popular predictions

Qwen:

```bash
conda run --no-capture-output -n mind-py311 python scripts/evaluate.py \
  --input-path outputs/correction_phase/reports/correction-qwen3-vl-8b-popular/results.csv \
  --label-overrides outputs/normalized/repope/popular.jsonl \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-qwen3-vl-8b-popular-repope
```

InternVL:

```bash
conda run --no-capture-output -n mind-py311 python scripts/evaluate.py \
  --input-path outputs/correction_phase/reports/correction-internvl3.5-8b-popular/results.csv \
  --label-overrides outputs/normalized/repope/popular.jsonl \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-internvl3.5-8b-popular-repope
```

### Shared-bank control

Build shared banks:

```bash
conda run --no-capture-output -n mind-py311 python scripts/build_manifolds.py \
  --reference-cache outputs/cache/qwen3-vl-8b/pope-reference-64/train \
  --output-root outputs/correction_phase/reference_banks_shared \
  --model-name qwen3-vl-8b \
  --bank-scope shared
```

```bash
conda run --no-capture-output -n mind-py311 python scripts/build_manifolds.py \
  --reference-cache outputs/cache/internvl3.5-8b/pope-reference-64/train \
  --output-root outputs/correction_phase/reference_banks_shared \
  --model-name internvl3.5-8b \
  --bank-scope shared
```

Compute shared-bank popular features:

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_drift.py \
  --cache-path outputs/cache/qwen3-vl-8b/pope/popular \
  --reference-root outputs/correction_phase/reference_banks_shared \
  --model-name qwen3-vl-8b \
  --output-root outputs/correction_phase/features \
  --experiment-name correction-qwen3-vl-8b-popular-shared \
  --split popular \
  --bank-scope shared
```

Completed closeout outputs now present:

- `outputs/correction_phase/reports/correction-qwen3-vl-8b-popular-shared/metrics.json`
- `outputs/correction_phase/reports/correction-qwen3-vl-8b-popular-shared-object-heldout/metrics.json`
- `outputs/correction_phase/reports/correction-internvl3.5-8b-popular-shared/metrics.json`
- `outputs/correction_phase/reports/correction-internvl3.5-8b-popular-shared-object-heldout/metrics.json`

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_drift.py \
  --cache-path outputs/cache/internvl3.5-8b/pope/popular \
  --reference-root outputs/correction_phase/reference_banks_shared \
  --model-name internvl3.5-8b \
  --output-root outputs/correction_phase/features \
  --experiment-name correction-internvl3.5-8b-popular-shared \
  --split popular \
  --bank-scope shared
```

Popular `image_grouped` detector runs:

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/correction_phase/features/correction-qwen3-vl-8b-popular-shared/popular.parquet \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-qwen3-vl-8b-popular-shared \
  --split-strategy image_grouped \
  --num-folds 5
```

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/correction_phase/features/correction-internvl3.5-8b-popular-shared/popular.parquet \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-internvl3.5-8b-popular-shared \
  --split-strategy image_grouped \
  --num-folds 5
```

Popular `object_heldout` detector runs:

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/correction_phase/features/correction-qwen3-vl-8b-popular-shared/popular.parquet \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-qwen3-vl-8b-popular-shared-object-heldout \
  --split-strategy object_heldout \
  --num-folds 2
```

```bash
conda run --no-capture-output -n mind-py311 python scripts/train_detector.py \
  --train-path outputs/correction_phase/features/correction-internvl3.5-8b-popular-shared/popular.parquet \
  --output-root outputs/correction_phase/reports \
  --experiment-name correction-internvl3.5-8b-popular-shared-object-heldout \
  --split-strategy object_heldout \
  --num-folds 2
```

### Adversarial closeout reruns

Qwen can reuse the existing adversarial cache:

```bash
conda run --no-capture-output -n mind-py311 python scripts/compute_drift.py \
  --cache-path outputs/cache/qwen3-vl-8b/pope/adversarial \
  --reference-root outputs/correction_phase/reference_banks \
  --model-name qwen3-vl-8b \
  --output-root outputs/correction_phase/features \
  --experiment-name correction-qwen3-vl-8b-adversarial \
  --split adversarial
```

InternVL needs a fresh adversarial extraction first, and that is the step currently blocked by the machine CUDA state described above.

### Export the paper package

```bash
conda run --no-capture-output -n mind-py311 python scripts/export_paper_package.py \
  --reports-root outputs/correction_phase/reports \
  --output-root artifacts/paper_closeout
```

This final export is still blocked until:

- `outputs/correction_phase/reports/correction-internvl3.5-8b-adversarial/metrics.json`

## 3. Normalize POPE and RePOPE

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

## 4. Build Reference Candidates

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

## 5. Plan Experiment Commands

Preview the commands for a preset without running them:

```bash
conda run --no-capture-output -n mind-py311 python scripts/run_experiment.py \
  --config configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml \
  --stages prepare,cache_reference,extract_eval,build_manifolds,compute_drift,train_detector,evaluate,plot
```

Use `--execute` only after the required assets are in place.

## 6. Manual Stage Commands

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
  --experiment-name medium-qwen3-vl-8b-popular \
  --split-strategy image_grouped \
  --num-folds 5
```

This command now supports:

- `row`
- `image_grouped`
- `object_heldout`

Use `image_grouped` as the primary protocol and keep `row` only for historical comparison.

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
  --experiment-name medium-qwen3-vl-8b-popular \
  --split-strategy image_grouped \
  --num-folds 5
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

## 7. Intended Stage Order

1. Smoke: `configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml`
2. Medium: `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`
3. Main: `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`
4. Ablations: `configs/experiments/ablations/qwen3_vl_8b_popular.yaml`
5. RePOPE relabel evaluation on the same saved predictions
6. InternVL cross-family comparison
7. H-POPE only if the public files become available

## 8. Current External Blockers

- the public COCO assets are present locally
- `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`, and `OpenGVLab/InternVL3_5-8B-HF` config or processor checks were verified through `HF_ENDPOINT=https://hf-mirror.com`
- H-POPE public assets were not found in a directly usable release package
- the machine state changed again on `2026-03-31`:
  - `nvidia-smi` reports `Unable to determine the device handle for GPU1: 0000:3B:00.0: Unknown Error`
  - fresh `mind-py311` PyTorch processes report `torch.cuda.is_available() == False`
  - fresh `mind-py311` PyTorch processes report `torch.cuda.device_count() == 0`
- this blocks:
  - the fresh InternVL adversarial extraction
  - the pooled shared-bank closeout control, because the exact pooled reference-bank stats are not practical on CPU alone at the current per-layer support size
