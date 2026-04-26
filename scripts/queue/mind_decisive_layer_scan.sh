#!/usr/bin/env bash
set -euo pipefail

# Decisive-round layer-count sensitivity queue.
# This is a serial script meant to be launched under tmux.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROJECT_ROOT="/home/team/zhanghaonan/mind"
ROUND2="$PROJECT_ROOT/outputs/round2_2026_04"
OUT="$PROJECT_ROOT/outputs/decisive_round_2026_04/layer_scan"
CONDA_ENV="${CONDA_ENV:-mind-py311}"
GPU_ID="${GPU_ID:-1}"

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT/job_logs"

run_py() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

ensure_pope_reference_candidates() {
  local candidates="$OUT/reference_candidates/pope_popular_coco_train_candidates.json"

  if [[ -f "$candidates" ]]; then
    return
  fi

  run_py scripts/prepare_data.py \
    build-reference \
    --instances-json "$PROJECT_ROOT/data/coco/annotations/instances_train2017.json" \
    --output "$candidates" \
    --allowed-objects-from "$ROUND2/normalized/pope/popular.jsonl" \
    --max-images-per-object 64
}

cache_reference_states() {
  local model_config="$1"
  local layer_count="$2"
  local setting_key="$3"
  local dataset_name="$4"
  local batch_size="$5"
  local references
  local image_root

  if [[ "$setting_key" == "pope" ]]; then
    ensure_pope_reference_candidates
    references="$OUT/reference_candidates/pope_popular_coco_train_candidates.json"
    image_root="$PROJECT_ROOT/data/coco/train2017"
  elif [[ "$setting_key" == "dash-b" ]]; then
    references="$ROUND2/reference_candidates/dash_b_positive_reference_candidates.json"
    image_root="$PROJECT_ROOT/data/dash_b"
  else
    echo "Unknown setting_key for reference cache: $setting_key" >&2
    exit 1
  fi

  run_py scripts/cache_reference_states.py \
    --references "$references" \
    --image-root "$image_root" \
    --model-config "$model_config" \
    --output-root "$OUT/reference_cache/lc${layer_count}/${setting_key}" \
    --dataset-name "$dataset_name" \
    --split train \
    --device cuda \
    --shard-size 128 \
    --batch-size "$batch_size" \
    --selected-layers "$layer_count" \
    --layer-range middle \
    --max-new-tokens 1
}

build_layer_reference_bank() {
  local model="$1"
  local model_config="$2"
  local layer_count="$3"
  local setting_key="$4"
  local dataset_name="$5"
  local batch_size="$6"
  local cache_root="$OUT/reference_cache/lc${layer_count}/${setting_key}/$model/$dataset_name/train"
  local bank_root="$OUT/reference_banks/lc${layer_count}/${setting_key}/object"

  cache_reference_states "$model_config" "$layer_count" "$setting_key" "$dataset_name" "$batch_size"

  run_py scripts/build_manifolds.py \
    --reference-cache "$cache_root" \
    --output-root "$bank_root" \
    --model-name "$model" \
    --bank-scope object
}

extract_eval_cache() {
  local model="$1"
  local model_config="$2"
  local layer_count="$3"
  local dataset_name="$4"
  local split="$5"
  local records="$6"
  local image_root="$7"
  local batch_size="$8"

  run_py scripts/extract_eval_states.py \
    --records "$records" \
    --model-config "$model_config" \
    --output-root "$OUT/cache/lc${layer_count}" \
    --dataset-name "$dataset_name" \
    --split "$split" \
    --image-root "$image_root" \
    --device cuda \
    --shard-size 128 \
    --batch-size "$batch_size" \
    --selected-layers "$layer_count" \
    --layer-range middle \
    --max-new-tokens 1
}

compute_features() {
  local model="$1"
  local layer_count="$2"
  local dataset_name="$3"
  local split="$4"
  local reference_root="$5"
  local experiment="$6"

  run_py scripts/compute_drift.py \
    --cache-path "$OUT/cache/lc${layer_count}/$model/$dataset_name/$split" \
    --reference-root "$reference_root" \
    --model-name "$model" \
    --output-root "$OUT/features" \
    --experiment-name "$experiment" \
    --split "$split" \
    --bank-scope object
}

compute_baselines() {
  local model="$1"
  local layer_count="$2"
  local dataset_name="$3"
  local split="$4"
  local reference_root="$5"
  local experiment="$6"
  local split_strategy="$7"
  local num_folds="$8"

  run_py scripts/compute_baselines.py \
    --features-path "$OUT/features/$experiment/$split.parquet" \
    --cache-path "$OUT/cache/lc${layer_count}/$model/$dataset_name/$split" \
    --reference-root "$reference_root" \
    --model-name "$model" \
    --output-root "$OUT/reports" \
    --experiment-name "$experiment" \
    --split-strategy "$split_strategy" \
    --num-folds "$num_folds" \
    --bank-scope object \
    --full-variant raw_plus_calibrated_full_curve \
    --variants full,linear_probe,no_manifold
}

run_setting() {
  local model="$1"
  local model_config="$2"
  local layer_count="$3"
  local dataset_name="$4"
  local split="$5"
  local setting="$6"
  local records="$7"
  local image_root="$8"
  local setting_key="$9"
  local reference_dataset_name="${10}"
  local split_strategy="${11}"
  local num_folds="${12}"
  local batch_size="${13}"
  local experiment="layer-scan-${model}-${setting}-lc${layer_count}-object"
  local reference_root="$OUT/reference_banks/lc${layer_count}/${setting_key}/object"

  build_layer_reference_bank "$model" "$model_config" "$layer_count" "$setting_key" "$reference_dataset_name" "$batch_size"
  extract_eval_cache "$model" "$model_config" "$layer_count" "$dataset_name" "$split" "$records" "$image_root" "$batch_size"
  compute_features "$model" "$layer_count" "$dataset_name" "$split" "$reference_root" "$experiment"
  compute_baselines "$model" "$layer_count" "$dataset_name" "$split" "$reference_root" "$experiment" "$split_strategy" "$num_folds"
}

run_model() {
  local model="$1"
  local model_config="$2"
  local batch_size="$3"
  shift 3
  local layer_count

  for layer_count in "$@"; do
    run_setting \
      "$model" \
      "$model_config" \
      "$layer_count" \
      pope \
      popular \
      popular-object-heldout \
      "$ROUND2/normalized/pope/popular.jsonl" \
      "$PROJECT_ROOT/data/coco/val2014" \
      pope \
      pope-reference-64 \
      object_heldout \
      2 \
      "$batch_size"

    run_setting \
      "$model" \
      "$model_config" \
      "$layer_count" \
      dash-b \
      main \
      dash-b \
      "$ROUND2/normalized/dash-b/main.jsonl" \
      "$PROJECT_ROOT/data/dash_b" \
      dash-b \
      dash-b-reference-64 \
      image_grouped \
      5 \
      "$batch_size"
  done
}

run_model qwen3-vl-8b configs/models/qwen3_vl_8b_local.yaml 8 8 12 16
run_model molmo-7b-d-0924 configs/models/molmo_7b_d_0924_local.yaml 4 8 12
