# GPU0 Training Status Check And Recovery Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the current round-two GPU0 training queue finished cleanly or stopped on an error, then either report the full completed results or remount the unfinished queue safely in `tmux` or `nohup` on `GPU 0`.

**Architecture:** Use the repo's unified GPU0 queue as the single source of truth. First gather live runtime evidence and the latest queue logs, then compare that against the expected round-two output tree. If the queue is incomplete, resume the existing launcher in a persistent session without changing the scheduling policy.

**Tech Stack:** Bash queue scripts, `tmux`, `nohup`, `nvidia-smi`, repo job logs, round-two output/report directories.

---

### Task 1: Pin down the active runtime state

**Files:**
- Inspect: `scripts/queue/mind_round2_unified_serial.sh`
- Inspect: `scripts/queue/start_mind_when_gpu0_free.sh`
- Inspect: `docs/review/2026-04-round4-status-audit.md`

**Steps:**
1. Check `nvidia-smi`, `tmux ls`, and `ps` for any live MIND queue or worker.
2. Confirm whether `GPU 0` is free, waiting, or running a live MIND job.
3. Record whether the active queue is mounted in a persistent session already.

**Validation:**
- Live process evidence identifies the current MIND owner state on `GPU 0`.
- The persistent-session status is explicit, not inferred.

### Task 2: Find the last successful queue checkpoint

**Files:**
- Inspect: `outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260409_resume6.log`
- Inspect: `outputs/round2_2026_04/job_logs/mind_wait_for_gpu0_20260409_resume6.log`
- Inspect: `outputs/round2_2026_04/reports/`
- Inspect: `outputs/round2_2026_04/readouts/`

**Steps:**
1. Read the tail of the latest queue log and wait log.
2. Identify the last completed unit and any terminal error or normal completion marker.
3. Cross-check that with saved report directories and remaining compact readout directories.

**Validation:**
- The queue end state is classified as either clean completion or specific failure/interruption.
- The last completed and first missing units are known exactly.

### Task 3: Branch on the outcome

**Files:**
- Runtime only if recovery is needed
- Inspect: `outputs/round2_2026_04/reports/` if reporting is needed

**Steps:**
1. If the queue finished, collect the final result set that the run actually produced.
2. If the queue failed, identify the exact resume command from the existing launcher path.
3. Use the existing GPU0 launcher policy; do not create a parallel queue or move work to `GPU 1`.

**Validation:**
- The chosen branch follows directly from the evidence.
- No alternate queue path is introduced.

### Task 4: Resume safely if the queue is incomplete

**Files:**
- Runtime only

**Steps:**
1. Stop any stale non-working session only if it is no longer doing useful work.
2. Remount `scripts/queue/start_mind_when_gpu0_free.sh` in a persistent `tmux` session, or `nohup` only if `tmux` is unavailable.
3. Verify the resumed session points to `GPU 0` and writes to fresh logs.

**Validation:**
- `tmux ls` or `ps` shows the persistent launcher.
- `nvidia-smi` and the new log tail show the queue is waiting on or using `GPU 0` only.

### Task 5: Produce the final audit

**Files:**
- Inspect: `outputs/round2_2026_04/reports/`
- Inspect: `docs/tables/round2/`
- Update if needed: `docs/review/2026-04-round4-status-audit.md`

**Steps:**
1. Summarize what completed, what remains, and whether training is live again.
2. If the run is complete, report the saved round-two results that exist now.
3. If the run is resumed, report the exact session/log handles needed to monitor it.

**Validation:**
- Every checklist item is checked against the user request.
- Final claims are backed by fresh command output and file evidence.
