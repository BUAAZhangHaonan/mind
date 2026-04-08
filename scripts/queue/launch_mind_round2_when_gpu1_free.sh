#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "launch_mind_round2_when_gpu1_free.sh is deprecated. MIND now runs on GPU 0 only." >&2
exec bash scripts/queue/start_mind_when_gpu0_free.sh "$@"
