# Wavelet Course V2 Paired Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a paired v2 wavelet-course run that compares every configured method on the same validated sample grid.

**Architecture:** Keep the v1 runner, config, and outputs intact. V2 gets its own config and output root, and every training, evaluation, report, and feature artifact must be derived from one `paired_grid`. The `paired_grid` is the hard correctness constraint: if a method cannot produce features for every required row in the same order with the same split and label, the v2 run must fail closed or record that method as failed without changing the grid.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PyWavelets, scikit-learn, optional xgboost gated by config.

---

## Tasks

1. Create `configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml` with `experiment_name: wavelet_course_repope_qwen3_vl_8b_v2_paired`, `output_root: outputs/wavelet_course_v2_paired`, and the v2 paired method grid. Do not change `configs/wavelet_course/repope_qwen3_vl_8b.yaml`.
2. Add a `paired_grid` builder in `src/mind/wavelet_course/` that consumes the existing primary population, keeps `population_key`, `image_id`, subset, split, and label, and writes a stable grid artifact before feature extraction.
3. Make `scripts/wavelet_course_run.py` select v2 behavior from the v2 config while preserving the v1 default behavior and the existing `outputs/wavelet_course` artifacts.
4. Route Teacher-Bagua, Ours-Wavelet, and cached hidden-state baselines through the same `paired_grid` row order. No method may rebuild splits, drop rows, reorder rows, or use method-local filtering after the grid is fixed.
5. Add fail-closed checks for missing keys, duplicate keys, split drift, label drift, row-order drift, non-finite features, and method outputs whose row count does not equal `len(paired_grid)`.
6. Extend reports under `outputs/wavelet_course_v2_paired/` with the grid path, grid row count, per-split class counts, method status rows, best configs, and a summary that says the comparison is paired.
7. Add focused tests in `tests/wavelet_course/` for `paired_grid` construction, identical row order across methods, failure on missing or reordered method features, v1 default output preservation, and v2 output-root isolation.
8. Run `pytest tests/wavelet_course -q`, then v2 preflight, then v2 quick run, then the full v2 run if quick succeeds and resources allow.

## Completion Criteria

- V1 outputs under `outputs/wavelet_course` are preserved; v2 writes only under `outputs/wavelet_course_v2_paired`.
- `outputs/wavelet_course_v2_paired/audit/paired_grid.json` exists and is the only source of row order, split, and label for all v2 methods.
- Every successful v2 metrics row is computed on exactly `len(paired_grid)` rows with matching `population_key`, split, and label.
- Any method that cannot satisfy the `paired_grid` contract is listed as a failure row with a concrete reason; it must not silently change the grid.
- Tests cover the `paired_grid` contract and the v1/v2 output isolation.
- The v2 summary and metrics clearly state that the evaluation is paired.
