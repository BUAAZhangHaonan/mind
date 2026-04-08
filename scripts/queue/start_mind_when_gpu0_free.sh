#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
if [[ "$GPU_ID" != "0" ]]; then
  echo "MIND is restricted to GPU 0 only. Refusing GPU_ID=$GPU_ID." >&2
  exit 1
fi

WAIT_LOG="${WAIT_LOG:-outputs/round2_2026_04/job_logs/mind_wait_for_gpu0_20260408.log}"
QUEUE_LOG="${QUEUE_LOG:-outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_disk_bounded.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "$WAIT_LOG")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$WAIT_LOG"
}

gpu_uuid_for_index() {
  local index="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F', ' -v target="$index" '$1 == target {print $2}'
}

gpu_busy() {
  local gpu_uuid="$1"
  if [[ -z "$gpu_uuid" ]]; then
    return 1
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | awk -F', ' -v target="$gpu_uuid" '$1 == target {found=1} END {exit(found ? 0 : 1)}'
}

main() {
  local gpu_uuid
  gpu_uuid="$(gpu_uuid_for_index "$GPU_ID")"
  if [[ -z "$gpu_uuid" ]]; then
    log "FAIL could not resolve GPU ${GPU_ID} uuid"
    exit 1
  fi

  log "Waiting for GPU ${GPU_ID} (${gpu_uuid}) to become free for MIND"
  while gpu_busy "$gpu_uuid"; do
    log "GPU ${GPU_ID} still busy; sleeping ${POLL_SECONDS}s"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | tee -a "$WAIT_LOG"
    sleep "$POLL_SECONDS"
  done

  while [[ -e outputs/round2_2026_04/readouts ]]; do
    log "Transient readout tree still exists; waiting for cleanup to finish"
    sleep "$POLL_SECONDS"
  done

  log "GPU ${GPU_ID} is free; starting unified MIND queue"
  export GPU_ID
  export QUEUE_LOG
  exec bash scripts/queue/mind_round2_unified_serial.sh
}

main "$@"
