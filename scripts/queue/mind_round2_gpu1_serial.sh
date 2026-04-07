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

run_with_retries() {
  local name="$1"
  shift
  local attempt
  local status
  for attempt in 1 2 3; do
    log "START $name attempt=$attempt"
    set +e
    "$@" 2>&1 | tee -a "$QUEUE_LOG"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -eq 0 ]]; then
      log "DONE $name attempt=$attempt"
      return 0
    fi
    if [[ "$attempt" -lt 3 ]]; then
      log "RETRY $name attempt=$attempt exit=$status sleep=60"
      sleep 60
      continue
    fi
    log "FAIL $name (exit=$status attempts=$attempt)"
    exit "$status"
  done
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

  run_with_retries "$step_name" \
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

  run_with_retries "$step_name" \
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

main() {
  log "GPU1 extraction queue starting"
  log "root=$ROOT_DIR"
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  log "conda_env=$CONDA_ENV"

  ensure_readouts \
    "internvl3.5-8b pope popular readouts" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "4"

  ensure_readouts \
    "llava-onevision-7b pope popular readouts" \
    "configs/models/llava_onevision_7b.yaml" \
    "llava-onevision-7b" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "4"

  ensure_readouts \
    "molmo-7b-d-0924 pope popular readouts" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "4"

  ensure_readouts \
    "internvl3.5-8b dash-b readouts" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "dash-b" \
    "main" \
    "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
    "data/dash_b" \
    "4"

  ensure_readouts \
    "llava-onevision-7b dash-b readouts" \
    "configs/models/llava_onevision_7b.yaml" \
    "llava-onevision-7b" \
    "dash-b" \
    "main" \
    "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
    "data/dash_b" \
    "4"

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

  log "GPU1 extraction queue completed"
}

main "$@"
