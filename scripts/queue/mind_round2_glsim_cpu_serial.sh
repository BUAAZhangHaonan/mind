#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-mind-py311}"
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"
POLL_SECONDS="${POLL_SECONDS:-60}"
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

run_with_retries() {
  local step_name="$1"
  shift
  local -a cmd=("$@")
  local attempt=1
  local status=0
  local cmd_text
  cmd_text="$(command_string "${cmd[@]}")"
  while (( attempt <= RETRY_ATTEMPTS )); do
    log "START $step_name (attempt ${attempt}/${RETRY_ATTEMPTS})"
    log "CMD $cmd_text"
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$QUEUE_LOG"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -eq 0 ]]; then
      log "DONE $step_name"
      return 0
    fi
    if (( attempt == RETRY_ATTEMPTS )); then
      log "FAIL $step_name (exit=$status)"
      return "$status"
    fi
    log "RETRY $step_name after exit=$status; sleeping ${RETRY_DELAY_SECONDS}s"
    sleep "$RETRY_DELAY_SECONDS"
    attempt=$((attempt + 1))
  done
  return 1
}

shard_count() {
  local shard_dir="$1"
  if [[ ! -d "$shard_dir" ]]; then
    echo 0
    return
  fi
  find "$shard_dir" -maxdepth 1 -name 'shard-*.pt' | wc -l | tr -d ' '
}

wait_for_readouts() {
  local step_name="$1"
  local readout_path="$2"
  local expected_shards="$3"
  while true; do
    local existing
    existing="$(shard_count "$readout_path")"
    if [[ "$existing" -eq "$expected_shards" ]]; then
      log "READY $step_name readouts ($existing/$expected_shards shards)"
      return
    fi
    if [[ "$existing" -gt "$expected_shards" ]]; then
      log "FAIL $step_name readouts show $existing shards but expected $expected_shards"
      exit 1
    fi
    log "WAIT $step_name readouts ($existing/$expected_shards shards)"
    sleep "$POLL_SECONDS"
  done
}

ensure_glsim() {
  local step_name="$1"
  local readout_path="$2"
  local expected_shards="$3"
  local model_config="$4"
  local experiment_name="$5"
  local split_strategy="$6"
  local num_folds="$7"

  local report_root="outputs/round2_2026_04/reports/${experiment_name}"
  if [[ -f "${report_root}/glsim.json" && -f "${report_root}/glsim_results.csv" && -f "${report_root}/glsim_selection.csv" ]]; then
    log "SKIP $step_name complete"
    return
  fi

  wait_for_readouts "$step_name" "$readout_path" "$expected_shards"

  run_with_retries "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/run_glsim.py \
        --readout-path "$readout_path" \
        --model-config "$model_config" \
        --output-root outputs/round2_2026_04/reports \
        --experiment-name "$experiment_name" \
        --device cpu \
        --split-strategy "$split_strategy" \
        --num-folds "$num_folds" \
        --bootstrap-resamples 1000
}

queue_model_glsim() {
  local model_name="$1"
  local model_config="$2"
  local popular_report="$3"
  local dash_report="$4"

  local popular_readout="outputs/round2_2026_04/readouts/${model_name}/pope/popular"
  local dash_readout="outputs/round2_2026_04/readouts/${model_name}/dash-b/main"

  ensure_glsim \
    "${model_name} pope popular GLSim image_grouped" \
    "$popular_readout" \
    "47" \
    "$model_config" \
    "$popular_report" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "${model_name} pope popular GLSim object_heldout" \
    "$popular_readout" \
    "47" \
    "$model_config" \
    "${popular_report}-object-heldout" \
    "object_heldout" \
    "2"
  ensure_glsim \
    "${model_name} dash-b GLSim image_grouped" \
    "$dash_readout" \
    "42" \
    "$model_config" \
    "$dash_report" \
    "image_grouped" \
    "5"
  ensure_glsim \
    "${model_name} dash-b GLSim object_heldout" \
    "$dash_readout" \
    "42" \
    "$model_config" \
    "${dash_report}-object-heldout" \
    "object_heldout" \
    "2"
}

main() {
  log "GLSim CPU queue starting"
  log "root=$ROOT_DIR"
  log "conda_env=$CONDA_ENV"

  queue_model_glsim \
    "qwen3-vl-8b" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "round2-qwen3-vl-8b-popular-final" \
    "round2-qwen3-vl-8b-dash-b"
  queue_model_glsim \
    "internvl3.5-8b" \
    "configs/models/internvl3_5_8b.yaml" \
    "round2-internvl3.5-8b-popular" \
    "round2-internvl3.5-8b-dash-b"
  queue_model_glsim \
    "llava-onevision-7b" \
    "configs/models/llava_onevision_7b.yaml" \
    "round2-llava-onevision-7b-popular" \
    "round2-llava-onevision-7b-dash-b"
  queue_model_glsim \
    "molmo-7b-d-0924" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "round2-molmo-7b-d-0924-popular" \
    "round2-molmo-7b-d-0924-dash-b"

  log "GLSim CPU queue completed"
}

main "$@"
