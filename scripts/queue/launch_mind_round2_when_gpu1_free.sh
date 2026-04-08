#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WAIT_LOG="${WAIT_LOG:-outputs/round2_2026_04/job_logs/mind_round2_unified_wait_20260408.log}"
QUEUE_LOG="${QUEUE_LOG:-outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_resume_wait.log}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

mkdir -p "$(dirname "$WAIT_LOG")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$WAIT_LOG"
}

gpu_uuid_for_index() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1 {print $2}'
}

gpu1_busy() {
  local gpu1_uuid="$1"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader \
    | grep -q "^${gpu1_uuid},"
}

main() {
  local gpu1_uuid
  gpu1_uuid="$(gpu_uuid_for_index)"
  if [[ -z "$gpu1_uuid" ]]; then
    log "FAIL could not resolve GPU 1 UUID"
    exit 1
  fi

  log "Waiting for GPU 1 to become free for MIND (uuid=${gpu1_uuid})"
  while gpu1_busy "$gpu1_uuid"; do
    log "GPU 1 still busy; sleeping ${CHECK_INTERVAL_SECONDS}s"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader \
      | grep "^${gpu1_uuid}," | tee -a "$WAIT_LOG"
    sleep "$CHECK_INTERVAL_SECONDS"
  done

  log "GPU 1 is free; starting unified MIND queue"
  QUEUE_LOG="$QUEUE_LOG" bash scripts/queue/mind_round2_unified_serial.sh 2>&1 | tee -a "$WAIT_LOG"
}

main "$@"
