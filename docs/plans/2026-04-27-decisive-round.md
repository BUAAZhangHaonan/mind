# Decisive Round Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run only the decisive MIND experiments that test whether the useful signal comes from object-conditioned manifolds or from a compact hidden-state probe.

**Architecture:** Keep the existing round-two pipeline. Change the headline `full` MIND alias from simple calibrated statistics to the full calibrated curve, then reuse saved features, caches, and reference banks to recompute reports in `outputs/decisive_round_2026_04/`. Build shared and shuffled banks from the existing object bank where possible, and only launch tmux jobs for stages that need sustained runtime.

**Tech Stack:** Python 3.11, pandas, scikit-learn logistic detectors, PyTorch tensor reference banks, existing `scripts/compute_baselines.py`, `scripts/build_manifolds.py`, `scripts/run_experiment.py`, and `scripts/export_paper_package.py`.

---

## Completion Criteria

- `full` MIND uses `raw_plus_calibrated_full_curve` by default in library code, CLI code, generated experiment commands, and queue scripts.
- Main tables are regenerated with full-curve MIND as `full_MIND`, while the old simple-stats row remains visible through the feature-ablation table and a side-by-side comparison report.
- Bank identity is evaluated for object, shared, and shuffled-object banks on DASH-B and POPE popular object-heldout for `qwen3-vl-8b` and `molmo-7b-d-0924`.
- Layer-count sensitivity is evaluated on the same two models and settings, using full-curve features and the selected bank scope.
- Every command runs with `PYTHONNOUSERSITE=1`; long jobs run in `tmux`; GPU extraction jobs must avoid GPU 0 unless explicitly capped below half usage.
- Each code/report node is committed and pushed before the next node starts.

## Task 1: Switch Headline Full MIND Default

**Files:**
- Modify: `src/mind/evaluation/baselines.py`
- Modify: `scripts/compute_baselines.py`
- Modify: `scripts/queue/mind_round2_unified_serial.sh`
- Modify: `scripts/export_paper_package.py`
- Modify: `tests/unit/test_baselines.py`
- Modify: `tests/unit/test_pipeline_scripts.py`
- Modify: `tests/integration/test_paper_export.py`
- Optional docs update after numbers: `Project_Summary_Report.md`

**Steps:**
1. Update tests so `DEFAULT_FULL_VARIANT` is expected to be `raw_plus_calibrated_full_curve`.
2. Run the narrow tests and confirm they fail for the old simple-stats default.
3. Update the default constants and hard-coded queue full-variant argument.
4. Update export figure text from simple-stat language to full-curve language.
5. Add or update tests proving `scripts/train_detector.py` inherits the new default and paper export maps `full_MIND` to the full-curve payload while table 2 still keeps simple stats.
6. Run:
   `PYTHONNOUSERSITE=1 conda run --no-capture-output -n mind-py311 python -m pytest -q tests/unit/test_baselines.py::test_default_full_variant_is_full_curve tests/unit/test_pipeline_scripts.py::test_run_experiment_uses_default_full_curve_for_baselines_stage tests/unit/test_pipeline_scripts.py::test_compute_baselines_can_apply_label_overrides_and_full_variant`
7. Run:
   `PYTHONNOUSERSITE=1 conda run --no-capture-output -n mind-py311 python -m pytest -q tests/integration/test_paper_export.py`
8. Commit and push with message: `Update MIND default to full curve`

## Task 2: Regenerate Full-Curve Main Tables

**Files/Outputs:**
- Read: `outputs/round2_2026_04/features/**`
- Read: `outputs/round2_2026_04/cache/**`
- Read: `outputs/round2_2026_04/reference_banks/**`
- Write: `outputs/decisive_round_2026_04/reports/**`
- Write: `artifacts/decisive_round_2026_04/**`
- Modify generated tables in `docs/tables/`

**Steps:**
1. Use existing feature/cache/reference files. Do not re-extract model states.
2. Run `scripts/compute_baselines.py` for all existing retained report names with `--full-variant raw_plus_calibrated_full_curve`.
3. Export the paper package from `outputs/decisive_round_2026_04/reports`.
4. Produce a side-by-side simple-stats vs full-curve comparison table keyed by model, benchmark, and protocol.
5. Verify all retained main table rows exist for POPE popular, DASH-B, POPE adversarial, RePOPE, and transfer/object-heldout.
6. Commit and push with message: `Regenerate full curve main tables`

## Task 3: Run Bank-Identity Control

**Files/Outputs:**
- Read: `outputs/round2_2026_04/reference_banks/**`
- Write: `outputs/decisive_round_2026_04/reference_banks_shared/**`
- Write: `outputs/decisive_round_2026_04/reference_banks_shuffled/**`
- Write: `outputs/decisive_round_2026_04/reports/**`
- Write: `artifacts/decisive_round_2026_04/bank_identity.*`

**Steps:**
1. Build shared and shuffled banks for `qwen3-vl-8b` and `molmo-7b-d-0924` from the saved object banks.
2. Run object, shared, and shuffled-object `compute_baselines.py` jobs for DASH-B image-grouped and POPE popular object-heldout.
3. Keep detector type, split strategy, seeds, folds, and `raw_plus_calibrated_full_curve` fixed.
4. Run these commands inside a tmux session named `mind_decisive_bank_identity`.
5. Generate a bank ranking table by PR-AUC first and ROC-AUC second.
6. Commit and push with message: `Add bank identity control results`

## Task 4: Run Layer-Count Sensitivity

**Files/Outputs:**
- Write: `outputs/decisive_round_2026_04/layer_scan/**`
- Write: `artifacts/decisive_round_2026_04/layer_count_sensitivity.*`

**Steps:**
1. Use layer counts `8,12,16,20,24` when the cached selected-layer lists support those counts.
2. Keep the feature set as `raw_plus_calibrated_full_curve`.
3. Use the object bank unless Task 3 shows a clear shared/shuffled winner.
4. Run on DASH-B and POPE popular object-heldout for `qwen3-vl-8b` and `molmo-7b-d-0924`.
5. Launch sustained work in tmux session `mind_decisive_layer_scan`.
6. Generate a PR-AUC/ROC-AUC table by layer count.
7. Commit and push with message: `Add layer count sensitivity results`

## Task 5: Final Audit and Decision Summary

**Files/Outputs:**
- Write: `artifacts/decisive_round_2026_04/decision_summary.md`
- Modify: `Project_Summary_Report.md`

**Steps:**
1. Check each goal against the completion criteria.
2. Run targeted tests and report-generation sanity checks.
3. State whether object-conditioned geometry survives bank identity.
4. State whether full-curve/layer scan closes the gap to `linear_probe` or `no_manifold`.
5. Commit and push with message: `Summarize decisive round decision`
