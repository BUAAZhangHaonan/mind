# Stage D Final Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Stage D cross-domain, domain-expansion, and model-family evaluation for the frozen MIND-main method.

**Architecture:** Stage D reads only the unified full-cache manifest, reuses the frozen `Sphere-Traj-LSTM + Proxy Anchor + 0.50` training surface, and evaluates fixed methods across a small source-target protocol set. It writes independent outputs under `outputs/stageD` and does not mutate prior Stage outputs.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, pytest, existing `mind.trajectory` helpers.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/stage_d/conftest.py`
- Create: `tests/stage_d/test_stage_d_manifest_loading.py`
- Create: `tests/stage_d/test_stage_d_protocols.py`
- Create: `tests/stage_d/test_stage_d_baseline_tiers.py`
- Create: `tests/stage_d/test_stage_d_family_summary.py`
- Create: `tests/stage_d/test_stage_d_verdicts.py`

**Steps:**
1. Write failing tests for frozen manifest loading, protocols, Tier A/B method sets, family grouping, and verdict labels.
2. Run `conda run --no-capture-output -n mind-py311 python -m pytest -q tests/stage_d` and verify the tests fail because Stage D modules are missing.

### Task 2: Stage D Core Modules

**Files:**
- Create: `src/mind/trajectory/stage_d_manifest.py`
- Create: `src/mind/trajectory/stage_d_protocols.py`
- Create: `src/mind/trajectory/stage_d_status.py`
- Create: `src/mind/trajectory/stage_d_baselines.py`
- Create: `scripts/stage_d_run.py`

**Steps:**
1. Implement frozen constants and preflight.
2. Implement exact source-target protocol set and oracle diagnostic labeling.
3. Implement Tier A/Tier B method contracts and related-method feasibility payload.
4. Implement family grouping and verdict helpers.
5. Implement runner that trains frozen embeddings per source/model/seed and evaluates fixed baselines.
6. Run Stage D tests until green.

### Task 3: Documentation

**Files:**
- Create: `docs/STAGE_D.md`
- Modify: `docs/DESIGN_NOTES.md`
- Modify: `README.md`

**Steps:**
1. Document frozen assumptions, protocols, baselines, output paths, and scope boundaries.
2. Run Stage D tests and full regression tests.
3. Commit code/tests and docs separately.

### Task 4: Experiment Outputs

**Files:**
- Create under `outputs/stageD/`

**Steps:**
1. Run `scripts/stage_d_run.py`.
2. Validate all required artifacts exist.
3. Run final tests.
4. Commit forced output artifacts and push.
