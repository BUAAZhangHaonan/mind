#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-mind-py311}"
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

QUEUE_LOG="${QUEUE_LOG:-outputs/round2_2026_04/job_logs/mind_gpu1_serial_queue_20260407.log}"
mkdir -p "$(dirname "$QUEUE_LOG")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$QUEUE_LOG"
}

run_logged() {
  local name="$1"
  shift
  log "START $name"
  set +e
  "$@" 2>&1 | tee -a "$QUEUE_LOG"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAIL $name (exit=$status)"
    exit "$status"
  fi
  log "DONE $name"
}

jsonl_rows() {
  grep -cve '^[[:space:]]*$' "$1"
}

expected_shards() {
  local records_path="$1"
  local shard_size="$2"
  local rows
  rows="$(jsonl_rows "$records_path")"
  echo $(((rows + shard_size - 1) / shard_size))
}

shard_count() {
  local shard_dir="$1"
  if [[ ! -d "$shard_dir" ]]; then
    echo 0
    return
  fi
  find "$shard_dir" -maxdepth 1 -name 'shard-*.pt' | wc -l | tr -d ' '
}

archive_partial_dir() {
  local target_dir="$1"
  if [[ ! -d "$target_dir" ]]; then
    return
  fi
  local timestamp_suffix
  timestamp_suffix="$(date '+%Y%m%d_%H%M%S')"
  local backup_dir="${target_dir}.partial_${timestamp_suffix}"
  log "ARCHIVE $target_dir -> $backup_dir"
  mv "$target_dir" "$backup_dir"
}

ensure_readouts() {
  local step_name="$1"
  local model_config="$2"
  local model_name="$3"
  local dataset_name="$4"
  local split_name="$5"
  local records_path="$6"
  local image_root="$7"
  local batch_size="${8:-8}"

  local shard_dir="outputs/round2_2026_04/readouts/${model_name}/${dataset_name}/${split_name}"
  local expected
  expected="$(expected_shards "$records_path" 64)"
  local existing
  existing="$(shard_count "$shard_dir")"

  if [[ "$existing" -eq "$expected" && "$expected" -gt 0 ]]; then
    log "SKIP $step_name complete ($existing/$expected shards)"
    return
  fi
  if [[ "$existing" -gt 0 ]]; then
    archive_partial_dir "$shard_dir"
  fi

  run_logged "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/extract_readout_states.py \
        --records "$records_path" \
        --model-config "$model_config" \
        --output-root outputs/round2_2026_04/readouts \
        --dataset-name "$dataset_name" \
        --split "$split_name" \
        --image-root "$image_root" \
        --device cuda \
        --shard-size 64 \
        --batch-size "$batch_size" \
        --max-new-tokens 1

  existing="$(shard_count "$shard_dir")"
  if [[ "$existing" -ne "$expected" ]]; then
    log "FAIL $step_name wrote $existing/$expected shards"
    exit 1
  fi
}

ensure_glsim() {
  local step_name="$1"
  local readout_path="$2"
  local model_config="$3"
  local experiment_name="$4"
  local split_strategy="$5"
  local num_folds="$6"

  local report_root="outputs/round2_2026_04/reports/${experiment_name}"
  if [[ -f "${report_root}/glsim.json" && -f "${report_root}/glsim_results.csv" && -f "${report_root}/glsim_selection.csv" ]]; then
    log "SKIP $step_name complete"
    return
  fi

  run_logged "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/run_glsim.py \
        --readout-path "$readout_path" \
        --model-config "$model_config" \
        --output-root outputs/round2_2026_04/reports \
        --experiment-name "$experiment_name" \
        --device cuda \
        --split-strategy "$split_strategy" \
        --num-folds "$num_folds" \
        --bootstrap-resamples 1000
}

ensure_halp() {
  local step_name="$1"
  local readout_path="$2"
  local experiment_name="$3"
  local split_strategy="$4"
  local num_folds="$5"

  local report_root="outputs/round2_2026_04/reports/${experiment_name}"
  if [[ -f "${report_root}/halp.json" && -f "${report_root}/halp_results.csv" && -f "${report_root}/halp_selection.csv" ]]; then
    log "SKIP $step_name complete"
    return
  fi

  run_logged "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/run_halp.py \
        --readout-path "$readout_path" \
        --output-root outputs/round2_2026_04/reports \
        --experiment-name "$experiment_name" \
        --split-strategy "$split_strategy" \
        --num-folds "$num_folds" \
        --bootstrap-resamples 1000
}

ensure_eval_cache() {
  local step_name="$1"
  local model_config="$2"
  local model_name="$3"
  local records_path="$4"
  local split_name="$5"
  local image_root="$6"
  local batch_size="${7:-8}"

  local shard_dir="outputs/round2_2026_04/cache/${model_name}/pope/${split_name}"
  local expected
  expected="$(expected_shards "$records_path" 128)"
  local existing
  existing="$(shard_count "$shard_dir")"

  if [[ "$existing" -eq "$expected" && "$expected" -gt 0 ]]; then
    log "SKIP $step_name complete ($existing/$expected shards)"
    return
  fi
  if [[ "$existing" -gt 0 ]]; then
    archive_partial_dir "$shard_dir"
  fi

  run_logged "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/extract_eval_states.py \
        --records "$records_path" \
        --model-config "$model_config" \
        --output-root outputs/round2_2026_04/cache \
        --dataset-name pope \
        --split "$split_name" \
        --image-root "$image_root" \
        --device cuda \
        --shard-size 128 \
        --batch-size "$batch_size" \
        --selected-layers 16 \
        --layer-range middle \
        --max-new-tokens 1

  existing="$(shard_count "$shard_dir")"
  if [[ "$existing" -ne "$expected" ]]; then
    log "FAIL $step_name wrote $existing/$expected shards"
    exit 1
  fi
}

queue_model_gpu_work() {
  local model_name="$1"
  local model_config="$2"
  local popular_report="$3"
  local dash_report="$4"
  local needs_dash_repair="$5"

  local popular_readout="outputs/round2_2026_04/readouts/${model_name}/pope/popular"
  local dash_readout="outputs/round2_2026_04/readouts/${model_name}/dash-b/main"

  ensure_readouts \
    "${model_name} pope popular readouts" \
    "$model_config" \
    "$model_name" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014"

  if [[ "$needs_dash_repair" == "yes" ]]; then
    ensure_readouts \
      "${model_name} dash-b readouts" \
      "$model_config" \
      "$model_name" \
      "dash-b" \
      "main" \
      "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
      "data/dash_b"
  else
    log "SKIP ${model_name} dash-b readouts already complete"
  fi

  ensure_glsim \
    "${model_name} pope popular GLSim image_grouped" \
    "$popular_readout" \
    "$model_config" \
    "$popular_report" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "${model_name} pope popular GLSim object_heldout" \
    "$popular_readout" \
    "$model_config" \
    "${popular_report}-object-heldout" \
    "object_heldout" \
    "2"

  ensure_glsim \
    "${model_name} dash-b GLSim image_grouped" \
    "$dash_readout" \
    "$model_config" \
    "$dash_report" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "${model_name} dash-b GLSim object_heldout" \
    "$dash_readout" \
    "$model_config" \
    "${dash_report}-object-heldout" \
    "object_heldout" \
    "2"
}

queue_model_halp() {
  local model_name="$1"
  local popular_report="$2"
  local dash_report="$3"

  local popular_readout="outputs/round2_2026_04/readouts/${model_name}/pope/popular"
  local dash_readout="outputs/round2_2026_04/readouts/${model_name}/dash-b/main"

  ensure_halp \
    "${model_name} pope popular HALP image_grouped" \
    "$popular_readout" \
    "$popular_report" \
    "image_grouped" \
    "5"
  ensure_halp \
    "${model_name} pope popular HALP object_heldout" \
    "$popular_readout" \
    "${popular_report}-object-heldout" \
    "object_heldout" \
    "2"

  ensure_halp \
    "${model_name} dash-b HALP image_grouped" \
    "$dash_readout" \
    "$dash_report" \
    "image_grouped" \
    "5"
  ensure_halp \
    "${model_name} dash-b HALP object_heldout" \
    "$dash_readout" \
    "${dash_report}-object-heldout" \
    "object_heldout" \
    "2"
}

main() {
  log "GPU1 queue starting"
  log "root=$ROOT_DIR"
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  log "conda_env=$CONDA_ENV"

  queue_model_gpu_work \
    "qwen3-vl-8b" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "round2-qwen3-vl-8b-popular-final" \
    "round2-qwen3-vl-8b-dash-b" \
    "no"
  queue_model_gpu_work \
    "internvl3.5-8b" \
    "configs/models/internvl3_5_8b.yaml" \
    "round2-internvl3.5-8b-popular" \
    "round2-internvl3.5-8b-dash-b" \
    "yes"
  queue_model_gpu_work \
    "llava-onevision-7b" \
    "configs/models/llava_onevision_7b.yaml" \
    "round2-llava-onevision-7b-popular" \
    "round2-llava-onevision-7b-dash-b" \
    "yes"
  queue_model_gpu_work \
    "molmo-7b-d-0924" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "round2-molmo-7b-d-0924-popular" \
    "round2-molmo-7b-d-0924-dash-b" \
    "no"

  ensure_eval_cache \
    "qwen3-vl-8b pope adversarial eval cache" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "outputs/round2_2026_04/normalized/pope/adversarial.jsonl" \
    "adversarial" \
    "data/coco/val2014" \
    "8"
  ensure_eval_cache \
    "internvl3.5-8b pope adversarial eval cache" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "outputs/round2_2026_04/normalized/pope/adversarial.jsonl" \
    "adversarial" \
    "data/coco/val2014" \
    "4"

  queue_model_halp \
    "qwen3-vl-8b" \
    "round2-qwen3-vl-8b-popular-final" \
    "round2-qwen3-vl-8b-dash-b"
  queue_model_halp \
    "internvl3.5-8b" \
    "round2-internvl3.5-8b-popular" \
    "round2-internvl3.5-8b-dash-b"
  queue_model_halp \
    "llava-onevision-7b" \
    "round2-llava-onevision-7b-popular" \
    "round2-llava-onevision-7b-dash-b"
  queue_model_halp \
    "molmo-7b-d-0924" \
    "round2-molmo-7b-d-0924-popular" \
    "round2-molmo-7b-d-0924-dash-b"

  log "GPU1 queue completed"
}

main "$@"
