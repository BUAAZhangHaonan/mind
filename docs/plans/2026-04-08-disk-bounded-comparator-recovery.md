# Disk-Bounded Comparator Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover the round-two comparator pipeline so it fits on the current disk, preserves the benchmark logic, and deletes transient readout caches once results are saved.

**Architecture:** Replace the current full-hidden-state readout cache with a compact comparator cache that stores only the tensors HALP and GLSim actually consume. Then switch the serial queue from "extract everything first" to "extract one unit, run HALP, run GLSim, delete that cache, move on" so disk use stays bounded.

**Tech Stack:** Python, PyTorch, pandas, bash queue runner, tmux

---

### Task 1: Add compact comparator cache support

**Files:**
- Modify: `src/mind/extractors/readouts.py`
- Modify: `scripts/extract_readout_states.py`
- Modify: `src/mind/comparators/halp.py`
- Modify: `src/mind/comparators/glsim.py`
- Test: `tests/unit/test_pipeline_scripts.py`

**Step 1: Write tests for compact readout entries**

Add tiny-entry tests that omit `full_hidden_states` and instead provide:
- `query_hidden_states`
- `vision_token_hidden_states`
- `object_hidden_states`
- `glsim_layer_indices`
- `glsim_vision_hidden_states`
- `vision_features`

Verify:
- `run_halp.py` still writes metrics and results
- `run_glsim.py` still writes metrics and results

**Step 2: Run the new compact-cache tests and watch them fail**

Run:

```bash
conda run --no-capture-output -n mind-py311 pytest tests/unit/test_pipeline_scripts.py -q -k 'compact'
```

**Step 3: Implement compact readout entry generation**

In `scripts/extract_readout_states.py`, convert each full readout batch into a compact comparator entry before saving. Keep:
- all-layer query token vectors
- all-layer HALP vision-token vectors
- all-layer object-token vectors
- cached GLSim image-layer slices for the default five GLSim layers
- `vision_features`
- token index metadata

Do not save `full_hidden_states` in the new cache format.

**Step 4: Teach HALP and GLSim to read compact entries**

Keep backward compatibility:
- if `full_hidden_states` exists, use the old path
- else use the compact fields

**Step 5: Run the compact-cache tests and verify they pass**

Run:

```bash
conda run --no-capture-output -n mind-py311 pytest tests/unit/test_pipeline_scripts.py -q -k 'compact or tiny_readout_cache'
```

**Step 6: Commit**

```bash
git add src/mind/extractors/readouts.py scripts/extract_readout_states.py src/mind/comparators/halp.py src/mind/comparators/glsim.py tests/unit/test_pipeline_scripts.py
git commit -m "fix(readouts): store compact comparator caches"
```

### Task 2: Make the queue disk-bounded

**Files:**
- Modify: `scripts/queue/mind_round2_unified_serial.sh`
- Test: `tests/unit/test_prefill_extraction.py`

**Step 1: Reorder the queue around comparator units**

For each `(model, benchmark)` unit:
- extract readout cache
- run HALP image_grouped
- run HALP object_heldout
- run GLSim image_grouped
- run GLSim object_heldout
- delete the readout cache directory

Do this before moving to the next unit.

**Step 2: Add explicit readout cleanup**

Add a queue helper that removes:
- `outputs/round2_2026_04/readouts/<model>/<dataset>/<split>/`

Only delete after both HALP and GLSim outputs for that unit exist.

**Step 3: Add queue logging for cleanup**

Log:
- bytes before cleanup
- bytes after cleanup
- path removed

**Step 4: Run queue script syntax check**

Run:

```bash
bash -n scripts/queue/mind_round2_unified_serial.sh
```

**Step 5: Commit**

```bash
git add scripts/queue/mind_round2_unified_serial.sh
git commit -m "fix(queue): bound disk by deleting finished readout caches"
```

### Task 3: Delete the old oversized readout tree

**Files:**
- No tracked files required

**Step 1: Verify no MIND queue is running**

Run:

```bash
ps -eo pid,cmd | rg 'extract_readout_states|run_halp|run_glsim'
tmux ls
```

**Step 2: Delete the old readout cache tree**

Run:

```bash
rm -rf outputs/round2_2026_04/readouts
```

**Step 3: Verify the space is freed**

Run:

```bash
df -h /home/team
du -sh outputs/round2_2026_04
```

### Task 4: Relaunch the queue safely on GPU 0

**Files:**
- Modify if needed: `docs/review/2026-04-round4-status-audit.md`

**Step 1: Start the queue in tmux**

Run:

```bash
tmux new-session -d -s mind_round2_unified_queue 'cd /home/team/zhanghaonan/mind && GPU_ID=0 QUEUE_LOG=outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_disk_bounded.log bash scripts/queue/mind_round2_unified_serial.sh'
```

**Step 2: Verify live progress**

Run:

```bash
tmux ls
ps -eo pid,cmd | rg 'mind_round2_unified_serial|extract_readout_states|run_halp|run_glsim'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
tail -n 40 outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_disk_bounded.log
```

**Step 3: Update the status note**

Record:
- the oversized readout tree was deleted
- the queue now uses compact caches
- the queue deletes each readout unit after comparator results are saved
- the active MIND device is `GPU 0`

**Step 4: Commit**

```bash
git add docs/review/2026-04-round4-status-audit.md
git commit -m "docs(status): record disk-bounded comparator recovery"
```

### Task 5: Verification

**Files:**
- No new files required

**Step 1: Run focused tests**

```bash
conda run --no-capture-output -n mind-py311 pytest tests/unit/test_prefill_extraction.py -q
conda run --no-capture-output -n mind-py311 pytest tests/unit/test_baselines.py -q
conda run --no-capture-output -n mind-py311 pytest tests/unit/test_pipeline_scripts.py -q -k 'halp or glsim or compact'
```

**Step 2: Check clean git state**

```bash
git status --short
git log --oneline -6
```
