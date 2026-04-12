#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMUX_SESSION="${TMUX_SESSION:-mind_round2_unified_queue}"
GPU_ID="${GPU_ID:-1}"
WAIT_LOG="${WAIT_LOG:-outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260412_gpu1_wait.log}"
QUEUE_LOG="${QUEUE_LOG:-outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260412_gpu1_tmux.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ "$GPU_ID" != "1" ]]; then
  echo "MIND is restricted to GPU 1 only. Refusing GPU_ID=$GPU_ID." >&2
  exit 1
fi

mkdir -p "$(dirname "$WAIT_LOG")" "$(dirname "$QUEUE_LOG")"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session $TMUX_SESSION already exists."
  echo "Attach with: tmux attach -t $TMUX_SESSION"
  echo "Logs: queue=$QUEUE_LOG wait=$WAIT_LOG"
  exit 0
fi

build_tmux_command() {
  cat <<EOF
set -euo pipefail
ROOT_DIR=$(printf '%q' "$ROOT_DIR")
WAIT_LOG=$(printf '%q' "$WAIT_LOG")
QUEUE_LOG=$(printf '%q' "$QUEUE_LOG")
POLL_SECONDS=$(printf '%q' "$POLL_SECONDS")
cd "$ROOT_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[\$(timestamp)] \$*" | tee -a "$WAIT_LOG"
}

gpu_uuid_for_index() {
  local index="\$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F', ' -v target="\$index" '\$1 == target {print \$2}'
}

gpu_busy() {
  local gpu_uuid="\$1"
  if [[ -z "\$gpu_uuid" ]]; then
    return 1
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | awk -F', ' -v target="\$gpu_uuid" '\$1 == target {found=1} END {exit(found ? 0 : 1)}'
}

main() {
  local gpu_uuid
  gpu_uuid="\$(gpu_uuid_for_index 1)"
  if [[ -z "\$gpu_uuid" ]]; then
    log "FAIL could not resolve GPU 1 uuid"
    exit 1
  fi

  log "Waiting for GPU 1 (\$gpu_uuid) to become free for MIND"
  while gpu_busy "\$gpu_uuid"; do
    log "GPU 1 still busy; sleeping \${POLL_SECONDS}s"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | tee -a "$WAIT_LOG"
    sleep "\$POLL_SECONDS"
  done

  log "GPU 1 is free; starting unified MIND queue"
  export GPU_ID=1
  export QUEUE_LOG="$QUEUE_LOG"
  exec bash scripts/queue/mind_round2_unified_serial.sh
}

main "\$@"
EOF
}

tmux_command="$(build_tmux_command)"
tmux new-session -d -s "$TMUX_SESSION" bash -lc "$tmux_command"

echo "Started tmux session $TMUX_SESSION for MIND on GPU 1."
echo "Attach with: tmux attach -t $TMUX_SESSION"
echo "Logs: queue=$QUEUE_LOG wait=$WAIT_LOG"
