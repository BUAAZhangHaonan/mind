# Wavelet Course Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add a self-contained RePOPE wavelet-course experiment that compares a strict Teacher-Bagua wavelet/LSTM baseline, an adapted layer-trace wavelet method, and cached hidden-state logistic baselines.

**Architecture:** Keep Stage 0 and Stage A untouched. Add `src/mind/wavelet_course/` for cache loading, population, features, baselines, metrics, and reporting. Add `scripts/wavelet_course_run.py` as the only CLI entry point and write outputs under `outputs/wavelet_course/`.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PyWavelets, scikit-learn, optional xgboost gated by config.

---

## Checklist

1. Create `configs/wavelet_course/repope_qwen3_vl_8b.yaml` with the exact scoped config list from the course spec.
2. Create `src/mind/wavelet_course/cache_loading.py`, `population.py`, `utils.py`, and `metrics.py` with fail-closed validation, grouped split construction, primary population labels, and unified metrics.
3. Create `teacher_bagua_features.py` with DWT denoising, 28 empirical features, epsilon accounting, memmap output, and fixed `(9, 114688)` sequence shape.
4. Create `ours_wavelet_features.py` with six layer-wise traces, SWT feasibility checks, feature names, final-logit broadcast yes/no traces, and feature dimension limit.
5. Create `baselines.py`, `reporting.py`, and `scripts/wavelet_course_run.py` for preflight, feature extraction, training, failure rows, metrics, best configs, and summary.
6. Create `tests/wavelet_course/` covering shape validation, primary population, teacher feature shape, ours feature shape, and fail-closed paths.
7. Run `pytest tests/wavelet_course -q`, then preflight, then quick, then full if the quick run succeeds and resources allow.

## Completion Criteria

- `--preflight-only` writes `outputs/wavelet_course/audit/cache_acceptance.json`.
- Full/quick run writes the required audit, feature, report, and resolved config files.
- Metrics include success and failure rows; failed configs are not hidden.
- No silent fallback is introduced for missing cache, wrong shapes, missing pywt, missing xgboost when disallowed, one-class training data, NaN/Inf features, or unavailable CUDA when CPU is not allowed.
- Summary text honestly compares Teacher-Bagua, Ours-Wavelet, and cached hidden-state baselines.
