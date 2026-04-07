#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-mind-py311}"
QUEUE_LOG="${QUEUE_LOG:-outputs/round2_2026_04/job_logs/mind_glsim_cpu_queue_20260407.log}"
mkdir -p "$(dirname "$QUEUE_LOG")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$QUEUE_LOG"
}

command_string() {
  printf '%q ' "$@"
}

run_with_retry() {
  local step_name="$1"
  shift
  local -a cmd=("$@")
  local attempt=1
  local max_attempts=3
  local delay_seconds=60

  while true; do
    log "START $step_name (attempt ${attempt}/${max_attempts})"
    log "CMD $(command_string "${cmd[@]}")"
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$QUEUE_LOG"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -eq 0 ]]; then
      log "DONE $step_name"
      return 0
    fi
    if (( attempt >= max_attempts )); then
      log "FAIL $step_name (exit=$status, attempts=$attempt)"
      return "$status"
    fi
    log "RETRY $step_name in ${delay_seconds}s (exit=$status, next_attempt=$((attempt + 1))/${max_attempts})"
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done
}

shard_count() {
  local shard_dir="$1"
  if [[ ! -d "$shard_dir" ]]; then
    echo 0
    return
  fi
  find "$shard_dir" -maxdepth 1 -name 'shard-*.pt' | wc -l | tr -d ' '
}

wait_for_complete_readouts() {
  local step_name="$1"
  local shard_dir="$2"
  local expected_shards="$3"

  while true; do
    local existing
    existing="$(shard_count "$shard_dir")"
    if [[ "$existing" -eq "$expected_shards" && "$expected_shards" -gt 0 ]]; then
      log "READY $step_name readouts ($existing/$expected_shards shards)"
      return
    fi
    if [[ "$existing" -gt "$expected_shards" ]]; then
      log "FAIL $step_name readouts exceed expected count ($existing/$expected_shards shards)"
      exit 1
    fi
    log "WAIT $step_name readouts ($existing/$expected_shards shards)"
    sleep 60
  done
}

ensure_glsim() {
  local step_name="$1"
  local readout_path="$2"
  local expected_shards="$3"
  local model_config="$4"
  local model_name="$5"
  local reference_root="$6"
  local experiment_name="$7"
  local split_strategy="$8"
  local num_folds="$9"

  wait_for_complete_readouts "$step_name" "$readout_path" "$expected_shards"

  local report_root="outputs/round2_2026_04/reports/${experiment_name}"
  if [[ -f "${report_root}/glsim.json" && -f "${report_root}/glsim_results.csv" && -f "${report_root}/glsim_selection.csv" ]]; then
    log "SKIP $step_name complete"
    return
  fi

  run_with_retry "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/run_glsim.py \
        --readout-path "$readout_path" \
        --model-config "$model_config" \
        --output-root outputs/round2_2026_04/reports \
        --experiment-name "$experiment_name" \
        --device cpu \
        --split-strategy "$split_strategy" \
        --num-folds "$num_folds" \
        --bootstrap-resamples 1000 \
        --reference-root "$reference_root" \
        --model-name "$model_name" \
        --bank-scope object
}

main() {
  log "GLSim CPU queue starting"
  log "root=$ROOT_DIR"
  log "conda_env=$CONDA_ENV"

  ensure_glsim \
    "qwen3-vl-8b pope popular GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/qwen3-vl-8b/pope/popular" \
    "47" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-qwen3-vl-8b-popular-final" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "qwen3-vl-8b pope popular GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/qwen3-vl-8b/pope/popular" \
    "47" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-qwen3-vl-8b-popular-final-object-heldout" \
    "object_heldout" \
    "2"
  ensure_glsim \
    "qwen3-vl-8b dash-b GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/qwen3-vl-8b/dash-b/main" \
    "42" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-qwen3-vl-8b-dash-b" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "qwen3-vl-8b dash-b GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/qwen3-vl-8b/dash-b/main" \
    "42" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-qwen3-vl-8b-dash-b-object-heldout" \
    "object_heldout" \
    "2"

  ensure_glsim \
    "molmo-7b-d-0924 dash-b GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/molmo-7b-d-0924/dash-b/main" \
    "42" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-molmo-7b-d-0924-dash-b" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "molmo-7b-d-0924 dash-b GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/molmo-7b-d-0924/dash-b/main" \
    "42" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-molmo-7b-d-0924-dash-b-object-heldout" \
    "object_heldout" \
    "2"

  ensure_glsim \
    "internvl3.5-8b pope popular GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/internvl3.5-8b/pope/popular" \
    "47" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-internvl3.5-8b-popular" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "internvl3.5-8b pope popular GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/internvl3.5-8b/pope/popular" \
    "47" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-internvl3.5-8b-popular-object-heldout" \
    "object_heldout" \
    "2"
  ensure_glsim \
    "llava-onevision-7b pope popular GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/llava-onevision-7b/pope/popular" \
    "47" \
    "configs/models/llava_onevision_7b.yaml" \
    "llava-onevision-7b" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-llava-onevision-7b-popular" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "llava-onevision-7b pope popular GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/llava-onevision-7b/pope/popular" \
    "47" \
    "configs/models/llava_onevision_7b.yaml" \
    "llava-onevision-7b" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-llava-onevision-7b-popular-object-heldout" \
    "object_heldout" \
    "2"
  ensure_glsim \
    "molmo-7b-d-0924 pope popular GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/molmo-7b-d-0924/pope/popular" \
    "47" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-molmo-7b-d-0924-popular" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "molmo-7b-d-0924 pope popular GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/molmo-7b-d-0924/pope/popular" \
    "47" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "outputs/round2_2026_04/reference_banks" \
    "round2-molmo-7b-d-0924-popular-object-heldout" \
    "object_heldout" \
    "2"

  ensure_glsim \
    "internvl3.5-8b dash-b GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/internvl3.5-8b/dash-b/main" \
    "42" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-internvl3.5-8b-dash-b" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "internvl3.5-8b dash-b GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/internvl3.5-8b/dash-b/main" \
    "42" \
    "configs/models/internvl3_5_8b.yaml" \
    "internvl3.5-8b" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-internvl3.5-8b-dash-b-object-heldout" \
    "object_heldout" \
    "2"
  ensure_glsim \
    "llava-onevision-7b dash-b GLSim image_grouped" \
    "outputs/round2_2026_04/readouts/llava-onevision-7b/dash-b/main" \
    "42" \
    "configs/models/llava_onevision_7b.yaml" \
    "llava-onevision-7b" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-llava-onevision-7b-dash-b" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "llava-onevision-7b dash-b GLSim object_heldout" \
    "outputs/round2_2026_04/readouts/llava-onevision-7b/dash-b/main" \
    "42" \
    "configs/models/llava_onevision_7b.yaml" \
    "llava-onevision-7b" \
    "outputs/round2_2026_04/reference_banks_dash_b" \
    "round2-llava-onevision-7b-dash-b-object-heldout" \
    "object_heldout" \
    "2"

  log "GLSim CPU queue completed"
}

main "$@"
