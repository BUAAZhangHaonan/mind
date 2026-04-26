#!/usr/bin/env bash
set -euo pipefail

# Decisive-round bank identity queue.
# This is a serial script meant to be launched under tmux.

WORKTREE="/home/team/zhanghaonan/mind/.worktrees/decisive-round-202604"
PROJECT_ROOT="/home/team/zhanghaonan/mind"
ROUND2="$PROJECT_ROOT/outputs/round2_2026_04"
OUT="$PROJECT_ROOT/outputs/decisive_round_2026_04"
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

cd "$WORKTREE"
mkdir -p "$OUT/job_logs"

run_py() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

build_banks() {
  local model="$1"
  local source_root="$2"
  local shared_root="$3"
  local shuffled_root="$4"

  run_py scripts/build_manifolds.py \
    --reference-bank-root "$source_root" \
    --output-root "$shared_root" \
    --model-name "$model" \
    --bank-scope shared

  run_py scripts/build_manifolds.py \
    --reference-bank-root "$source_root" \
    --output-root "$shuffled_root" \
    --model-name "$model" \
    --bank-scope shuffled_object
}

extract_dash_cache() {
  local model="$1"
  local model_config="$2"
  local selected_layers="$3"

  run_py scripts/extract_eval_states.py \
    --records "$ROUND2/normalized/dash-b/main.jsonl" \
    --model-config "$model_config" \
    --output-root "$OUT/cache" \
    --dataset-name dash-b \
    --split main \
    --image-root "$PROJECT_ROOT/data/dash_b" \
    --device cuda \
    --selected-layers "$selected_layers" \
    --layer-range middle
}

compute_bank_features() {
  local model="$1"
  local benchmark="$2"
  local split="$3"
  local cache_path="$4"
  local bank_scope="$5"
  local reference_root="$6"
  local experiment="$7"

  run_py scripts/compute_drift.py \
    --cache-path "$cache_path" \
    --reference-root "$reference_root" \
    --model-name "$model" \
    --output-root "$OUT/features" \
    --experiment-name "$experiment" \
    --split "$split" \
    --bank-scope "$bank_scope" \
    --batch-size 32
}

compute_bank_baseline() {
  local model="$1"
  local split_strategy="$2"
  local num_folds="$3"
  local cache_path="$4"
  local bank_scope="$5"
  local reference_root="$6"
  local experiment="$7"
  local split="$8"

  run_py scripts/compute_baselines.py \
    --features-path "$OUT/features/$experiment/$split.parquet" \
    --cache-path "$cache_path" \
    --reference-root "$reference_root" \
    --model-name "$model" \
    --output-root "$OUT/reports" \
    --experiment-name "$experiment" \
    --split-strategy "$split_strategy" \
    --num-folds "$num_folds" \
    --bank-scope "$bank_scope" \
    --full-variant raw_plus_calibrated_full_curve \
    --variants full
}

run_model() {
  local model="$1"
  local model_config="$2"
  local dash_selected_layers="$3"

  local pope_cache="$ROUND2/cache/$model/pope/popular"
  local dash_cache="$OUT/cache/$model/dash-b/main"

  build_banks "$model" "$ROUND2/reference_banks" "$OUT/reference_banks_shared" "$OUT/reference_banks_shuffled"
  build_banks "$model" "$ROUND2/reference_banks_dash_b" "$OUT/reference_banks_dash_b_shared" "$OUT/reference_banks_dash_b_shuffled"
  extract_dash_cache "$model" "$model_config" "$dash_selected_layers"

  for bank_scope in object shared shuffled_object; do
    local pope_reference="$ROUND2/reference_banks"
    local dash_reference="$ROUND2/reference_banks_dash_b"
    if [[ "$bank_scope" == "shared" ]]; then
      pope_reference="$OUT/reference_banks_shared"
      dash_reference="$OUT/reference_banks_dash_b_shared"
    elif [[ "$bank_scope" == "shuffled_object" ]]; then
      pope_reference="$OUT/reference_banks_shuffled"
      dash_reference="$OUT/reference_banks_dash_b_shuffled"
    fi

    local pope_exp="bankid-$model-popular-$bank_scope-object-heldout"
    compute_bank_features "$model" pope popular "$pope_cache" "$bank_scope" "$pope_reference" "$pope_exp"
    compute_bank_baseline "$model" object_heldout 2 "$pope_cache" "$bank_scope" "$pope_reference" "$pope_exp" popular

    local dash_exp="bankid-$model-dash-b-$bank_scope"
    compute_bank_features "$model" dash-b main "$dash_cache" "$bank_scope" "$dash_reference" "$dash_exp"
    compute_bank_baseline "$model" image_grouped 5 "$dash_cache" "$bank_scope" "$dash_reference" "$dash_exp" main
  done
}

run_model qwen3-vl-8b configs/models/qwen3_vl_8b_local.yaml 16
run_model molmo-7b-d-0924 configs/models/molmo_7b_d_0924_local.yaml 14
