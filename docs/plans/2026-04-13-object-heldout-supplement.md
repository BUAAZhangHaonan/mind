# Object-Heldout Supplement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run fresh mainline supplementary experiments to fill the missing `object_heldout` transfer-control evidence and update the tracked outputs from those new runs.

**Architecture:** Keep the maintained mainline pipeline intact. First, patch the detached launcher so experiment jobs can run under `tmux` on GPU 0 or GPU 1 only. Next, run the four fresh `object_heldout` baseline jobs that produce the missing transfer-control reports. Last, refresh the tracked export artifacts and report text from those new report directories.

**Tech Stack:** Bash queue scripts, Python CLI scripts, pytest, tmux, Pandas/JSON export pipeline

---

### Task 1: Record The Execution Plan

**Files:**
- Create: `docs/plans/2026-04-13-object-heldout-supplement.md`

**Step 1: Save the plan document**

Write this plan file into `docs/plans/`.

**Step 2: Verify the file exists**

Run: `test -f docs/plans/2026-04-13-object-heldout-supplement.md`
Expected: exit code `0`

**Step 3: Commit**

```bash
git add docs/plans/2026-04-13-object-heldout-supplement.md
git commit -m "docs: add object-heldout supplement plan"
```

### Task 2: Allow Detached Queue Launches On GPU 0 Or GPU 1

**Files:**
- Modify: `scripts/queue/mind_round2_unified_serial.sh`
- Modify: `scripts/queue/start_mind_round2_tmux.sh`
- Modify: `README.md`
- Modify: `docs/runbooks/experiments.md`
- Test: shell syntax checks on the touched queue scripts

**Step 1: Write the failing test/validation target**

Define the expected behavior:
- `GPU_ID` accepts only `0` or `1`
- `start_mind_round2_tmux.sh` status text and wait logic use the selected GPU
- docs say detached jobs must use `tmux` or `nohup`, and GPU 0 or GPU 1 only

**Step 2: Run validation to show the current mismatch**

Run:

```bash
rg -n "GPU 1 only|GPU 0 only|CUDA_VISIBLE_DEVICES=0,1,2|GPU 1 reserved|GPU 0 reserved" README.md docs/runbooks/experiments.md scripts/queue
```

Expected: matches that show the current policy drift.

**Step 3: Write the minimal implementation**

Update the queue scripts and docs so the detached launcher is parameterized by `GPU_ID` with valid values `0` and `1` only.

**Step 4: Run validation to verify the fix**

Run:

```bash
bash -n scripts/queue/mind_round2_unified_serial.sh scripts/queue/start_mind_round2_tmux.sh
rg -n "GPU 0 or GPU 1|tmux|nohup" README.md docs/runbooks/experiments.md scripts/queue
```

Expected: syntax checks pass, and the updated policy text is present.

**Step 5: Commit**

```bash
git add scripts/queue/mind_round2_unified_serial.sh scripts/queue/start_mind_round2_tmux.sh README.md docs/runbooks/experiments.md
git commit -m "fix: allow detached experiments on gpu0 or gpu1"
```

### Task 3: Run Fresh Object-Heldout Experiments For Qwen And InternVL

**Files:**
- Create: fresh report directories under `outputs/round2_2026_04/reports/`
- Verify: generated `baselines.json`, `variant_results/*.csv`, `ablations.csv`, `split_sensitivity.csv`

**Step 1: Launch the two jobs in a detached tmux session**

Run the exact commands inside `tmux`, with the selected allowed GPU:

```bash
tmux new-session -d -s mind_object_heldout_batch1 "cd /home/team/zhanghaonan/mind/.worktrees/object-heldout-supplement-20260413 && conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py --features-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/features/round2-qwen3-vl-8b-popular/popular.parquet --cache-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/cache/qwen3-vl-8b/pope/popular --reference-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reference_banks --model-name qwen3-vl-8b --output-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports --experiment-name round2-qwen3-vl-8b-popular-object-heldout --split-strategy object_heldout --num-folds 2 --bank-scope object --full-variant raw_plus_calibrated_simple --variants full,linear_probe && conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py --features-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/features/round2-internvl3.5-8b-popular/popular.parquet --cache-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/cache/internvl3.5-8b/pope/popular --reference-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reference_banks --model-name internvl3.5-8b --output-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports --experiment-name round2-internvl3.5-8b-popular-object-heldout --split-strategy object_heldout --num-folds 2 --bank-scope object --full-variant raw_plus_calibrated_simple --variants full,linear_probe"
```

**Step 2: Wait for completion and inspect outputs**

Run:

```bash
tmux capture-pane -pt mind_object_heldout_batch1
test -f /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-object-heldout/baselines.json
test -f /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular-object-heldout/baselines.json
```

Expected: both report directories exist and include `baselines.json`.

**Step 3: Commit**

```bash
git add /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-object-heldout /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular-object-heldout
git commit -m "data: add qwen and internvl object-heldout results"
```

### Task 4: Run Fresh Object-Heldout Experiments For LLaVA And Molmo

**Files:**
- Create: fresh report directories under `outputs/round2_2026_04/reports/`
- Verify: generated `baselines.json`, `variant_results/*.csv`, `ablations.csv`, `split_sensitivity.csv`

**Step 1: Launch the two jobs in a detached tmux session**

Run the exact commands inside `tmux`, with the selected allowed GPU:

```bash
tmux new-session -d -s mind_object_heldout_batch2 "cd /home/team/zhanghaonan/mind/.worktrees/object-heldout-supplement-20260413 && conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py --features-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/features/round2-llava-onevision-7b-popular/popular.parquet --cache-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/cache/llava-onevision-7b/pope/popular --reference-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reference_banks --model-name llava-onevision-7b --output-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports --experiment-name round2-llava-onevision-7b-popular-object-heldout --split-strategy object_heldout --num-folds 2 --bank-scope object --full-variant raw_plus_calibrated_simple --variants full,linear_probe && conda run --no-capture-output -n mind-py311 python scripts/compute_baselines.py --features-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/features/round2-molmo-7b-d-0924-popular/popular.parquet --cache-path /home/team/zhanghaonan/mind/outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/popular --reference-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reference_banks --model-name molmo-7b-d-0924 --output-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports --experiment-name round2-molmo-7b-d-0924-popular-object-heldout --split-strategy object_heldout --num-folds 2 --bank-scope object --full-variant raw_plus_calibrated_simple --variants full,linear_probe"
```

**Step 2: Wait for completion and inspect outputs**

Run:

```bash
tmux capture-pane -pt mind_object_heldout_batch2
test -f /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular-object-heldout/baselines.json
test -f /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular-object-heldout/baselines.json
```

Expected: both report directories exist and include `baselines.json`.

**Step 3: Commit**

```bash
git add /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular-object-heldout /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular-object-heldout
git commit -m "data: add llava and molmo object-heldout results"
```

### Task 5: Refresh The Tracked Transfer-Control Outputs

**Files:**
- Modify: `docs/tables/round2/table3_transfer_controls.md`
- Modify: `docs/tables/round2/table3_transfer_controls.csv`
- Modify: `Project_Summary_Report.md`
- Verify: exporter/tests as needed

**Step 1: Refresh the export bundle from the new reports**

Run:

```bash
conda run --no-capture-output -n mind-py311 python scripts/export_paper_package.py --reports-root /home/team/zhanghaonan/mind/outputs/round2_2026_04/reports --output-root artifacts/paper_closeout
```

**Step 2: Verify the transfer-control table is now populated**

Run:

```bash
rg -n "object_heldout|\\[Not Found\\]" docs/tables/round2/table3_transfer_controls.md Project_Summary_Report.md
conda run --no-capture-output -n mind-py311 python -m pytest -q tests/integration/test_paper_export.py
```

Expected: `table3_transfer_controls.md` contains populated `object_heldout` cells, and the export test passes.

**Step 3: Commit**

```bash
git add docs/tables/round2/table3_transfer_controls.md docs/tables/round2/table3_transfer_controls.csv Project_Summary_Report.md
git commit -m "docs: publish object-heldout transfer-control results"
```
