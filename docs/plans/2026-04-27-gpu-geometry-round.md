# MIND GPU Geometry Round Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the GPU round for DASH-B cache coverage, full bank identity controls, query-local neighbor banks, and final decision docs.

**Architecture:** Keep the existing cache, reference-bank, feature parquet, and baseline schemas intact. Add GPU primitives behind new experiment scripts and shared geometry modules, then run long jobs in tmux on GPU 0 only. Commit and push after each phase.

**Tech Stack:** Python, PyTorch CUDA, pandas/parquet, sklearn metrics for validation, tmux, existing MIND extraction and baseline scripts.

---

## Hard Gate: CUDA Environment

**Files:** none

1. Verify `master` equals `origin/master`.
2. Verify GPU 0 is free with `nvidia-smi -i 0`.
3. Verify `mind-py311` has CUDA-enabled PyTorch.
4. If PyTorch is CPU-only, install the requested CUDA PyTorch runtime in `mind-py311`.
5. Re-run `torch.cuda.is_available()` with `CUDA_VISIBLE_DEVICES=0`.

Completion: `mind-py311` can allocate a CUDA tensor on GPU 0.

## Phase 1: DASH-B Cache Regeneration

**Files:**
- Create: `scripts/experiments/verify_cache_coverage.py`
- Output: `docs/review/phase1_dash_b_cache_coverage.md`

1. Verify `outputs/round2_2026_04/normalized/dash-b/main.jsonl` has 2682 rows.
2. Run InternVL3.5-8B DASH-B eval extraction in tmux `mind_phase1_cache` on GPU 0.
3. Run LLaVA-OneVision-7B DASH-B eval extraction in the same tmux session on GPU 0.
4. Regenerate missing InternVL/LLaVA DASH-B reference state caches and reference banks if needed.
5. Add `verify_cache_coverage.py` and run it for all four DASH-B model caches.
6. Write the verification output document.
7. Commit and push: `Regenerate missing DASH-B eval caches for InternVL and LLaVA on GPU 0`.

Completion: all four DASH-B eval caches cover all 2682 sample IDs, with matching layer counts and vector dimensions.

## Phase 2: GPU Bank Identity Control

**Files:**
- Create: `src/mind/geometry/gpu_distances.py`
- Create: `scripts/experiments/build_gpu_drift_features.py`
- Create: `scripts/experiments/train_gpu_detector.py`
- Create: `tests/unit/test_gpu_distances.py`
- Create: `tests/unit/test_gpu_detector.py`
- Output: `docs/tables/experiment_bank_identity_v2.csv`
- Output: `docs/tables/experiment_bank_identity_v2.md`
- Output: `docs/review/experiment2_bank_analysis_v2.md`

1. Write GPU distance parity tests against CPU reference formulas.
2. Implement GPU angular, Euclidean, kNN angular, centroid angular, and centroid Euclidean distances.
3. Write GPU detector tests for split behavior and metric output shape.
4. Implement GPU logistic regression with PyTorch tensors on GPU 0.
5. Implement GPU feature construction while preserving existing parquet schema.
6. Run the full 72-row Experiment 2 rerun in tmux `mind_phase2_bank`.
7. Validate zero `missing_cache` rows and compare v2 against CPU table.
8. Commit and push: `Complete GPU-accelerated bank identity control experiment with full coverage`.

Completion: `experiment_bank_identity_v2.csv` has exactly 72 rows, all metric cells filled.

## Phase 3: Query-Local Neighbor Bank

**Files:**
- Create: `scripts/experiments/build_pooled_bank.py`
- Create: `src/mind/geometry/local_neighborhood.py`
- Create: `tests/unit/test_local_neighborhood.py`
- Output: `docs/tables/experiment_query_local_bank.csv`
- Output: `docs/tables/experiment_query_local_bank.md`
- Output: `docs/review/experiment3_query_local_analysis.md`

1. Build pooled per-model, per-layer grounded reference banks and metadata.
2. Verify pooled layer counts equal summed object-bank counts.
3. Implement GPU angular kNN local reference selection.
4. Implement local centroid angular, local PCA residual, neighbor mean angular, and neighbor std angular features.
5. Run query-local k=30 detectors plus requested baselines in tmux `mind_phase3_local`.
6. Validate the 48-row comparison table has zero missing rows.
7. Commit and push: `Add query-local neighbor bank experiment with GPU acceleration`.

Completion: query-local table and analysis answer all six requested questions.

## Phase 4: Final Documentation And Tests

**Files:**
- Modify: `docs/results_summary.md`
- Modify: `docs/review/next_step_decision.md`

1. Update results summary with the GPU round section and inline rankings.
2. Update next-step decision with the final conclusion from all GPU experiments.
3. Run `make test`.
4. Fix any failures.
5. Commit and push final docs and fixes.

Completion: `make test` passes, `master` is clean, and local `master` equals `origin/master`.
