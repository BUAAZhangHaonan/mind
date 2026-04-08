# GPU0 Queue Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the interrupted MIND round-two queue from the stale GPU1 wait loop to a real `GPU 0` tmux-mounted queue that resumes the unfinished work safely.

**Architecture:** Keep one unified serial queue. Update the queue guard and launcher so both target `GPU 0`, replace the stale tmux waiter, and record the live state in the tracked audit note.

**Tech Stack:** Bash queue scripts, tmux, git, existing round-two logs and output roots.

---

### Task 1: Confirm the stale queue state

**Files:**
- Inspect: `scripts/queue/mind_round2_unified_serial.sh`
- Inspect: `scripts/queue/start_mind_when_gpu0_free.sh`
- Inspect: `scripts/queue/launch_mind_round2_when_gpu1_free.sh`
- Inspect: `docs/review/2026-04-round4-status-audit.md`

**Steps:**
1. Check `tmux ls`, `nvidia-smi`, the wait log, and `ps` to confirm the current MIND session is only a GPU1 waiting loop.
2. Confirm `GPU 0` is free and `GPU 1` is occupied by MagFormer.
3. Confirm the unfinished queue state from the readout tree and the last queue failure log.

**Validation:**
- The evidence shows no active MIND training process.
- The evidence shows the current live tmux session is not doing useful work.

### Task 2: Align the queue scripts to GPU 0

**Files:**
- Modify: `scripts/queue/mind_round2_unified_serial.sh`
- Modify: `scripts/queue/launch_mind_round2_when_gpu1_free.sh`
- Modify: `scripts/queue/start_mind_when_gpu0_free.sh`

**Steps:**
1. Change the unified queue default GPU to `0`.
2. Change the guard so it refuses anything except `GPU 0`.
3. Update the header comments so they match the new policy.
4. Turn the old GPU1 launcher into a compatibility wrapper that hands control to the GPU0 launcher instead of waiting on the wrong device.
5. Keep the dedicated GPU0 launcher as the live entry point for tmux.

**Validation:**
- `bash -n scripts/queue/mind_round2_unified_serial.sh`
- `bash -n scripts/queue/start_mind_when_gpu0_free.sh`
- `bash -n scripts/queue/launch_mind_round2_when_gpu1_free.sh`

### Task 3: Remount the interrupted queue in tmux

**Files:**
- Runtime only

**Steps:**
1. Stop the stale `mind_round2_unified_queue` tmux session if it still points at the GPU1 waiter.
2. Start a fresh `mind_round2_unified_queue` session that runs `scripts/queue/start_mind_when_gpu0_free.sh`.
3. Verify the session starts immediately if `GPU 0` is free, or waits only on `GPU 0` if not.

**Validation:**
- `tmux ls`
- `ps` shows the GPU0 launcher or unified queue command.
- `nvidia-smi` shows MIND on `GPU 0` only when extraction starts.

### Task 4: Record the live state and publish it

**Files:**
- Modify: `docs/review/2026-04-round4-status-audit.md`

**Steps:**
1. Replace the stale “running on GPU 1 wait loop” wording with the current GPU0 recovery state.
2. Record whether the queue resumed immediately or is waiting.
3. Commit the queue-script changes and the status-note update.
4. Push `master`.

**Validation:**
- `git status --short` is clean after the push.
- `git show --stat --oneline HEAD`
