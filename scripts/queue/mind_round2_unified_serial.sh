#!/usr/bin/env bash
set -euo pipefail

# MIND round-two unified serial queue.
#
# Scheduling policy:
# - Run exactly one MIND task at a time.
# - Readouts are temporary working files, not durable artifacts. Extract one
#   model/benchmark unit, run the corrected official HALP row-split baseline on
#   that same unit, then delete the readout cache before moving on.
# - The old grouped HALP path and the readout-based GLSim adaptation are not in
#   this queue. They are not paper-safe official baselines for the current
#   round-two POPE/DASH-B lane.
# - Memory-heavy comparator work must never run concurrently.
# - The queue logs system memory and GPU state before and after every step, and
#   retries a step once if it dies with exit code 137.
# - The serial scheduler and the memory gate are the real host-safety controls.
#   A hard `ulimit -v` cap is optional because HALP can exceed a low virtual
#   address ceiling even when the machine still has plenty of free RAM.
# - GPU 0 is the only GPU allowed for current MIND work in this recovery pass.
#   GPU 1 is reserved for other project traffic.
#
# This queue replaces the older split queue scripts. It will refuse to start if
# another MIND extraction queue is still active, because parallel queues are the
# root scheduling failure that caused the recent crash.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-mind-py311}"
GPU_ID="${GPU_ID:-0}"
READOUT_BATCH_SIZE_DEFAULT="${READOUT_BATCH_SIZE_DEFAULT:-1}"
if [[ "$GPU_ID" != "0" ]]; then
  echo "MIND is restricted to GPU 0 only. Refusing GPU_ID=$GPU_ID." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

QUEUE_LOG="${QUEUE_LOG:-outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_disk_bounded.log}"
SAFE_AVAILABLE_GB="${SAFE_AVAILABLE_GB:-10}"
MEMORY_CHECK_SLEEP_SECONDS="${MEMORY_CHECK_SLEEP_SECONDS:-60}"
MEMORY_CHECK_ATTEMPTS="${MEMORY_CHECK_ATTEMPTS:-3}"
OOM_RETRY_SLEEP_SECONDS="${OOM_RETRY_SLEEP_SECONDS:-120}"
VMEM_LIMIT_PERCENT="${VMEM_LIMIT_PERCENT:-0}"

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

memory_available_gb() {
  awk '/^MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo
}

total_memory_gb() {
  awk '/^MemTotal:/ {print int($2 / 1024 / 1024)}' /proc/meminfo
}

total_memory_kb() {
  awk '/^MemTotal:/ {print $2}' /proc/meminfo
}

log_resource_state() {
  local label="$1"
  log "RESOURCE ${label}"
  {
    echo "--- free -h ---"
    free -h
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  } | tee -a "$QUEUE_LOG"
}

path_size_human() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "0"
    return
  fi
  du -sh "$path" 2>/dev/null | awk '{print $1}'
}

set_virtual_memory_limit() {
  local total_gb total_kb limit_gb limit_kb
  total_gb="$(total_memory_gb)"
  total_kb="$(total_memory_kb)"
  if [[ -z "$total_gb" || "$total_gb" -le 0 ]]; then
    log "FAIL could not resolve total system RAM from /proc/meminfo"
    exit 1
  fi
  if [[ "$VMEM_LIMIT_PERCENT" -le 0 ]]; then
    log "ulimit -v disabled; relying on serial scheduling and the memory gate"
    return
  fi
  if [[ "$VMEM_LIMIT_PERCENT" -ge 100 ]]; then
    log "FAIL VMEM_LIMIT_PERCENT must be between 1 and 99, got ${VMEM_LIMIT_PERCENT}"
    exit 1
  fi
  limit_kb=$(( total_kb * VMEM_LIMIT_PERCENT / 100 ))
  limit_gb=$(( limit_kb / 1024 / 1024 ))
  if [[ "$limit_kb" -lt 1048576 ]]; then
    log "FAIL computed virtual memory limit is too small: ${limit_kb} KB"
    exit 1
  fi
  ulimit -v "$limit_kb"
  log "ulimit -v set to ${limit_kb} KB (${limit_gb}G of ${total_gb}G total RAM; ${VMEM_LIMIT_PERCENT}%)"
}

wait_for_safe_memory() {
  local attempt available_gb
  for attempt in $(seq 1 "$MEMORY_CHECK_ATTEMPTS"); do
    available_gb="$(memory_available_gb)"
    log "MEMORY-GATE attempt=${attempt}/${MEMORY_CHECK_ATTEMPTS} available=${available_gb}G threshold=${SAFE_AVAILABLE_GB}G"
    if [[ "$available_gb" -ge "$SAFE_AVAILABLE_GB" ]]; then
      return 0
    fi
    log_resource_state "memory gate low-memory attempt ${attempt}"
    if [[ "$attempt" -lt "$MEMORY_CHECK_ATTEMPTS" ]]; then
      log "WAIT low available memory; sleeping ${MEMORY_CHECK_SLEEP_SECONDS}s"
      sleep "$MEMORY_CHECK_SLEEP_SECONDS"
    fi
  done
  log "FAIL insufficient memory: available RAM stayed below ${SAFE_AVAILABLE_GB}G after ${MEMORY_CHECK_ATTEMPTS} checks"
  return 1
}

kill_lingering_cpu_comparators() {
  local pids
  pids="$(ps -eo pid=,cmd= | awk '/python scripts\/run_(halp|glsim|glsim_adapted)\.py/ {print $1}')"
  if [[ -z "$pids" ]]; then
    log "No lingering HALP/GLSim-adapted CPU processes found"
    return
  fi
  log "KILL lingering CPU comparator processes: ${pids//$'\n'/ }"
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
  done <<< "$pids"
  sleep 5
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      log "KILL -9 lingering comparator PID=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$pids"
}

cleanup_partial_comparator_outputs() {
  local kind report_root metrics results selection present_count
  for report_root in outputs/round2_2026_04/reports/*; do
    [[ -d "$report_root" ]] || continue
    for kind in halp glsim glsim_adapted; do
      metrics="${report_root}/${kind}.json"
      results="${report_root}/${kind}_results.csv"
      selection="${report_root}/${kind}_selection.csv"
      present_count=0
      [[ -f "$metrics" ]] && present_count=$((present_count + 1))
      [[ -f "$results" ]] && present_count=$((present_count + 1))
      [[ -f "$selection" ]] && present_count=$((present_count + 1))
      if [[ "$present_count" -gt 0 && "$present_count" -lt 3 ]]; then
        log "CLEAN partial ${kind} output under ${report_root}"
        rm -f "$metrics" "$results" "$selection"
      fi
    done
  done
}

require_no_active_mind_extraction_queue() {
  local active
  active="$(ps -eo pid=,cmd= | awk '/python scripts\/extract_(readout_states|eval_states)\.py/ {print $1 ":" substr($0, index($0,$2))}')"
  if [[ -n "$active" ]]; then
    log "DEFER unified queue: active MIND extraction already running"
    while read -r line; do
      [[ -z "$line" ]] && continue
      log "ACTIVE ${line}"
    done <<< "$active"
    log "Let the current GPU extraction queue finish before starting the unified queue."
    return 1
  fi
  return 0
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
  find "$shard_dir" \
    -maxdepth 1 \
    -type f \
    -name 'shard-*.pt' \
    ! -name 'shard-*.part-*.pt' \
    | wc -l | tr -d ' '
}

readout_cache_has_required_halp_fields() {
  local readout_dir="$1"
  conda run --no-capture-output -n "$CONDA_ENV" python - "$readout_dir" <<'PY'
from pathlib import Path
import re
import sys

import torch

from mind.extractors.prefill import CHUNKED_CACHE_SHARD_FORMAT

root = Path(sys.argv[1])
part_pattern = re.compile(r"\.part-\d{5}\.pt$")
shard_paths = sorted(
    shard_path
    for shard_path in root.glob("shard-*.pt")
    if not part_pattern.search(shard_path.name)
)
if not shard_paths:
    raise SystemExit(2)

def iter_entries():
    for shard_path in shard_paths:
        payload = torch.load(shard_path, weights_only=False)
        if isinstance(payload, dict) and payload.get("format") == CHUNKED_CACHE_SHARD_FORMAT:
            for part_name in payload.get("parts", []):
                part_entries = torch.load(shard_path.parent / str(part_name), weights_only=False)
                for entry in part_entries:
                    yield entry
        elif isinstance(payload, list):
            for entry in payload:
                yield entry

for entry in iter_entries():
    if entry.get("vision_features") is None:
        sample_id = entry.get("sample_id", "<unknown>")
        print(f"missing vision_features for sample {sample_id}", file=sys.stderr)
        raise SystemExit(1)
    if entry.get("query_hidden_states") is None:
        sample_id = entry.get("sample_id", "<unknown>")
        print(f"missing query_hidden_states for sample {sample_id}", file=sys.stderr)
        raise SystemExit(1)
    if entry.get("vision_token_hidden_states") is None:
        sample_id = entry.get("sample_id", "<unknown>")
        print(f"missing vision_token_hidden_states for sample {sample_id}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

raise SystemExit(2)
PY
}

report_complete() {
  local report_root="$1"
  shift
  local required_path
  for required_path in "$@"; do
    [[ -f "${report_root}/${required_path}" ]] || return 1
  done
  return 0
}

require_path() {
  local path="$1"
  local reason="$2"
  if [[ ! -e "$path" ]]; then
    log "FAIL missing prerequisite: ${path} (${reason})"
    exit 1
  fi
}

run_serial_step() {
  local step_name="$1"
  shift
  local -a cmd=("$@")
  local cmd_text status
  cmd_text="$(command_string "${cmd[@]}")"
  log_resource_state "before ${step_name}"
  wait_for_safe_memory || exit 1
  log "START ${step_name}"
  log "CMD ${cmd_text}"
  set +e
  "${cmd[@]}" 2>&1 | tee -a "$QUEUE_LOG"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -eq 0 ]]; then
    log "DONE ${step_name}"
    log_resource_state "after ${step_name}"
    return 0
  fi
  if [[ "$status" -eq 137 ]]; then
    log "OOM ${step_name} exit=137; sleeping ${OOM_RETRY_SLEEP_SECONDS}s before retry"
    sleep "$OOM_RETRY_SLEEP_SECONDS"
    log_resource_state "oom recovery before retry ${step_name}"
    wait_for_safe_memory || exit 137
    log "RETRY ${step_name}"
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$QUEUE_LOG"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "$status" -eq 0 ]]; then
      log "DONE ${step_name} after retry"
      log_resource_state "after retry ${step_name}"
      return 0
    fi
    if [[ "$status" -eq 137 ]]; then
      log "FAIL ${step_name}: exit=137 even after retry"
      exit 137
    fi
  fi
  log "FAIL ${step_name}: exit=${status}"
  exit "$status"
}

ensure_readouts() {
  local step_name="$1"
  local model_config="$2"
  local model_name="$3"
  local dataset_name="$4"
  local split_name="$5"
  local records_path="$6"
  local image_root="$7"
  local batch_size="$8"
  local shard_dir expected existing

  shard_dir="outputs/round2_2026_04/readouts/${model_name}/${dataset_name}/${split_name}"
  expected="$(expected_shards "$records_path" 64)"
  existing="$(shard_count "$shard_dir")"
  if [[ "$existing" -eq "$expected" && "$expected" -gt 0 ]]; then
    if readout_cache_has_required_halp_fields "$shard_dir"; then
      log "SKIP ${step_name} complete (${existing}/${expected} shards)"
      return
    fi
    log "REBUILD ${step_name}: existing readout cache is not HALP-ready"
    rm -rf "$shard_dir"
    existing=0
  fi

  run_serial_step "$step_name" \
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
    log "FAIL ${step_name}: wrote ${existing}/${expected} shards"
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
  local batch_size="$7"
  local shard_dir expected existing

  shard_dir="outputs/round2_2026_04/cache/${model_name}/pope/${split_name}"
  expected="$(expected_shards "$records_path" 128)"
  existing="$(shard_count "$shard_dir")"
  if [[ "$existing" -eq "$expected" && "$expected" -gt 0 ]]; then
    log "SKIP ${step_name} complete (${existing}/${expected} shards)"
    return
  fi

  run_serial_step "$step_name" \
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
    log "FAIL ${step_name}: wrote ${existing}/${expected} shards"
    exit 1
  fi
}

cleanup_readout_path_if_results_complete() {
  local step_name="$1"
  local readout_path="$2"
  local halp_report_root="$3"

  if ! report_complete "$halp_report_root" "halp.json" "halp_results.csv" "halp_selection.csv"; then
    return
  fi
  if [[ -d "$readout_path" ]]; then
    local before_size after_size
    before_size="$(path_size_human "$readout_path")"
    log "CLEAN ${step_name}: deleting transient readout cache ${readout_path} size=${before_size}"
    rm -rf "$readout_path"
    after_size="$(path_size_human "$readout_path")"
    log "CLEAN ${step_name}: removed ${readout_path} size_after=${after_size}"
  fi
}

ensure_halp() {
  local step_name="$1"
  local readout_path="$2"
  local expected_shards="$3"
  local experiment_name="$4"
  local report_root="outputs/round2_2026_04/reports/${experiment_name}"

  if report_complete "$report_root" "halp.json" "halp_results.csv" "halp_selection.csv"; then
    log "SKIP ${step_name} complete"
    return
  fi
  require_path "$readout_path" "${step_name} requires readout cache"
  if [[ "$(shard_count "$readout_path")" -ne "$expected_shards" ]]; then
    log "FAIL ${step_name}: expected ${expected_shards} readout shards under ${readout_path}"
    exit 1
  fi

  run_serial_step "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/run_halp.py \
        --readout-path "$readout_path" \
        --output-root outputs/round2_2026_04/reports \
        --experiment-name "$experiment_name" \
        --device cuda \
        --split-strategy row \
        --test-size 0.2 \
        --bootstrap-resamples 1000
}

ensure_features() {
  local step_name="$1"
  local cache_path="$2"
  local reference_root="$3"
  local model_name="$4"
  local experiment_name="$5"
  local split_name="$6"
  local bank_scope="$7"
  local features_path="outputs/round2_2026_04/features/${experiment_name}/${split_name}.parquet"

  if [[ -f "$features_path" ]]; then
    log "SKIP ${step_name} complete"
    return
  fi
  require_path "$cache_path" "${step_name} requires eval cache"
  require_path "$reference_root/${model_name}/reference_counts.csv" "${step_name} requires reference bank stats"

  run_serial_step "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/compute_drift.py \
        --cache-path "$cache_path" \
        --reference-root "$reference_root" \
        --model-name "$model_name" \
        --output-root outputs/round2_2026_04/features \
        --experiment-name "$experiment_name" \
        --split "$split_name" \
        --bank-scope "$bank_scope"
}

ensure_baseline_report() {
  local step_name="$1"
  local features_path="$2"
  local cache_path="$3"
  local reference_root="$4"
  local model_name="$5"
  local experiment_name="$6"
  local split_strategy="$7"
  local num_folds="$8"
  local bank_scope="$9"
  shift 9
  local variants_csv=("$@")
  local report_root="outputs/round2_2026_04/reports/${experiment_name}"
  local variant_csv

  if report_complete "$report_root" "baselines.json" "ablations.csv" "split_sensitivity.csv"; then
    local complete="1"
    for variant_csv in "${variants_csv[@]}"; do
      [[ -f "${report_root}/variant_results/${variant_csv}.csv" ]] || complete="0"
    done
    if [[ "$complete" == "1" ]]; then
      log "SKIP ${step_name} complete"
      return
    fi
  fi

  require_path "$features_path" "${step_name} requires features"
  require_path "$cache_path" "${step_name} requires eval cache"
  require_path "$reference_root/${model_name}/reference_counts.csv" "${step_name} requires reference bank stats"

  run_serial_step "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/compute_baselines.py \
        --features-path "$features_path" \
        --cache-path "$cache_path" \
        --reference-root "$reference_root" \
        --model-name "$model_name" \
        --output-root outputs/round2_2026_04/reports \
        --experiment-name "$experiment_name" \
        --split-strategy "$split_strategy" \
        --num-folds "$num_folds" \
        --bank-scope "$bank_scope" \
        --full-variant raw_plus_calibrated_simple
}

ensure_reference_bank() {
  local step_name="$1"
  local reference_cache="$2"
  local output_root="$3"
  local model_name="$4"
  local bank_scope="$5"

  if [[ -f "${output_root}/${model_name}/reference_counts.csv" ]]; then
    log "SKIP ${step_name} complete"
    return
  fi
  require_path "$reference_cache" "${step_name} requires reference cache"

  run_serial_step "$step_name" \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python scripts/build_manifolds.py \
        --reference-cache "$reference_cache" \
        --output-root "$output_root" \
        --model-name "$model_name" \
        --bank-scope "$bank_scope"
}

run_comparator_unit() {
  local unit_name="$1"
  local model_config="$2"
  local model_name="$3"
  local dataset_name="$4"
  local split_name="$5"
  local records_path="$6"
  local image_root="$7"
  local batch_size="$8"
  local halp_experiment="${9}"
  local expected_shards_count="${10}"
  local readout_path="outputs/round2_2026_04/readouts/${model_name}/${dataset_name}/${split_name}"
  local halp_report_root="outputs/round2_2026_04/reports/${halp_experiment}"

  if report_complete "$halp_report_root" "halp.json" "halp_results.csv" "halp_selection.csv"; then
    log "SKIP ${unit_name} comparator unit complete"
    cleanup_readout_path_if_results_complete \
      "$unit_name" \
      "$readout_path" \
      "$halp_report_root"
    return
  fi

  ensure_readouts \
    "${unit_name} readouts" \
    "$model_config" \
    "$model_name" \
    "$dataset_name" \
    "$split_name" \
    "$records_path" \
    "$image_root" \
    "$batch_size"

  ensure_halp \
    "${unit_name} HALP row" \
    "$readout_path" \
    "$expected_shards_count" \
    "$halp_experiment"

  cleanup_readout_path_if_results_complete \
    "$unit_name" \
    "$readout_path" \
    "$halp_report_root"
}

phase_main_comparators() {
  log "PHASE main comparator units"
  run_comparator_unit \
    "qwen3-vl-8b pope popular" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-qwen3-vl-8b-popular-halp-row" \
    "47"

  run_comparator_unit \
    "internvl3.5-8b pope popular" \
    "configs/models/internvl3_5_8b_local.yaml" \
    "internvl3.5-8b" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-internvl3.5-8b-popular-halp-row" \
    "47"

  run_comparator_unit \
    "llava-onevision-7b pope popular" \
    "configs/models/llava_onevision_7b_local.yaml" \
    "llava-onevision-7b" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-llava-onevision-7b-popular-halp-row" \
    "47"

  run_comparator_unit \
    "molmo-7b-d-0924 pope popular" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "pope" \
    "popular" \
    "outputs/round2_2026_04/normalized/pope/popular.jsonl" \
    "data/coco/val2014" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-molmo-7b-d-0924-popular-halp-row" \
    "47"

  run_comparator_unit \
    "qwen3-vl-8b dash-b" \
    "configs/models/qwen3_vl_8b_local.yaml" \
    "qwen3-vl-8b" \
    "dash-b" \
    "main" \
    "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
    "data/dash_b" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-qwen3-vl-8b-dash-b-halp-row" \
    "42"

  run_comparator_unit \
    "internvl3.5-8b dash-b" \
    "configs/models/internvl3_5_8b_local.yaml" \
    "internvl3.5-8b" \
    "dash-b" \
    "main" \
    "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
    "data/dash_b" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-internvl3.5-8b-dash-b-halp-row" \
    "42"

  run_comparator_unit \
    "llava-onevision-7b dash-b" \
    "configs/models/llava_onevision_7b_local.yaml" \
    "llava-onevision-7b" \
    "dash-b" \
    "main" \
    "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
    "data/dash_b" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-llava-onevision-7b-dash-b-halp-row" \
    "42"

  run_comparator_unit \
    "molmo-7b-d-0924 dash-b" \
    "configs/models/molmo_7b_d_0924.yaml" \
    "molmo-7b-d-0924" \
    "dash-b" \
    "main" \
    "outputs/round2_2026_04/normalized/dash-b/main.jsonl" \
    "data/dash_b" \
    "$READOUT_BATCH_SIZE_DEFAULT" \
    "round2-molmo-7b-d-0924-dash-b-halp-row" \
    "42"

  ensure_eval_cache "qwen3-vl-8b pope adversarial eval cache" "configs/models/qwen3_vl_8b_local.yaml" "qwen3-vl-8b" "outputs/round2_2026_04/normalized/pope/adversarial.jsonl" "adversarial" "data/coco/val2014" "8"
  ensure_eval_cache "internvl3.5-8b pope adversarial eval cache" "configs/models/internvl3_5_8b_local.yaml" "internvl3.5-8b" "outputs/round2_2026_04/normalized/pope/adversarial.jsonl" "adversarial" "data/coco/val2014" "4"
}

phase_adversarial_cpu() {
  log "PHASE adversarial cpu pipeline"
  ensure_features "qwen3-vl-8b pope adversarial features" "outputs/round2_2026_04/cache/qwen3-vl-8b/pope/adversarial" "outputs/round2_2026_04/reference_banks" "qwen3-vl-8b" "round2-qwen3-vl-8b-adversarial" "adversarial" "object"
  ensure_baseline_report "qwen3-vl-8b pope adversarial baselines" "outputs/round2_2026_04/features/round2-qwen3-vl-8b-adversarial/adversarial.parquet" "outputs/round2_2026_04/cache/qwen3-vl-8b/pope/adversarial" "outputs/round2_2026_04/reference_banks" "qwen3-vl-8b" "round2-qwen3-vl-8b-adversarial" "image_grouped" "5" "object" "full" "drift_only" "no_manifold" "linear_probe" "output_p_yes" "output_logit_margin" "output_chosen_answer_confidence"

  ensure_features "internvl3.5-8b pope adversarial features" "outputs/round2_2026_04/cache/internvl3.5-8b/pope/adversarial" "outputs/round2_2026_04/reference_banks" "internvl3.5-8b" "round2-internvl3.5-8b-adversarial" "adversarial" "object"
  ensure_baseline_report "internvl3.5-8b pope adversarial baselines" "outputs/round2_2026_04/features/round2-internvl3.5-8b-adversarial/adversarial.parquet" "outputs/round2_2026_04/cache/internvl3.5-8b/pope/adversarial" "outputs/round2_2026_04/reference_banks" "internvl3.5-8b" "round2-internvl3.5-8b-adversarial" "image_grouped" "5" "object" "full" "drift_only" "no_manifold" "linear_probe" "output_p_yes" "output_logit_margin" "output_chosen_answer_confidence"

  ensure_features "llava-onevision-7b pope adversarial features" "outputs/round2_2026_04/cache/llava-onevision-7b/pope/adversarial" "outputs/round2_2026_04/reference_banks" "llava-onevision-7b" "round2-llava-onevision-7b-adversarial" "adversarial" "object"
  ensure_baseline_report "llava-onevision-7b pope adversarial baselines" "outputs/round2_2026_04/features/round2-llava-onevision-7b-adversarial/adversarial.parquet" "outputs/round2_2026_04/cache/llava-onevision-7b/pope/adversarial" "outputs/round2_2026_04/reference_banks" "llava-onevision-7b" "round2-llava-onevision-7b-adversarial" "image_grouped" "5" "object" "full" "drift_only" "no_manifold" "linear_probe" "output_p_yes" "output_logit_margin" "output_chosen_answer_confidence"

  ensure_features "molmo-7b-d-0924 pope adversarial features" "outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/adversarial" "outputs/round2_2026_04/reference_banks" "molmo-7b-d-0924" "round2-molmo-7b-d-0924-adversarial" "adversarial" "object"
  ensure_baseline_report "molmo-7b-d-0924 pope adversarial baselines" "outputs/round2_2026_04/features/round2-molmo-7b-d-0924-adversarial/adversarial.parquet" "outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/adversarial" "outputs/round2_2026_04/reference_banks" "molmo-7b-d-0924" "round2-molmo-7b-d-0924-adversarial" "image_grouped" "5" "object" "full" "drift_only" "no_manifold" "linear_probe" "output_p_yes" "output_logit_margin" "output_chosen_answer_confidence"
}

phase_late_cpu_work_pending() {
  log "PHASE late cpu pipeline pending"
  log "The remaining late CPU stages still need extra round-two prerequisites:"
  log "- qwen3-vl-8b POPE popular eval cache is missing under outputs/round2_2026_04/cache/qwen3-vl-8b/pope/popular"
  log "- internvl3.5-8b POPE popular eval cache is missing under outputs/round2_2026_04/cache/internvl3.5-8b/pope/popular"
  log "- qwen3-vl-8b and internvl3.5-8b POPE reference caches are missing, so shared/shuffled controls and bank-size ablations still cannot be rebuilt faithfully"
  log "- layer-count ablation still needs a dedicated runner that derives drift features from comparator-layer subsets"
}

main() {
  log "Unified MIND round-two serial queue starting"
  log "root=${ROOT_DIR}"
  log "gpu_id=${GPU_ID}"
  log "readout_batch_size_default=${READOUT_BATCH_SIZE_DEFAULT}"
  log "conda_env=${CONDA_ENV}"
  set_virtual_memory_limit
  kill_lingering_cpu_comparators
  cleanup_partial_comparator_outputs
  if ! require_no_active_mind_extraction_queue; then
    exit 0
  fi
  phase_main_comparators
  phase_adversarial_cpu
  phase_late_cpu_work_pending
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
