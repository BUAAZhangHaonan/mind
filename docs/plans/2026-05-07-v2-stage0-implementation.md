# v2 Stage 0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Stage 0 runner that audits normalized records, creates deterministic grouped splits, extracts full-layer model caches, and writes the exact `outputs/v2_stage0` contract.

**Architecture:** Use a small script wrapper in `scripts/v2/stage0_run.py` and keep reusable logic in `src/mind/trajectory`. The runner should load YAML config, audit records, assign grouped splits once, extract model caches, write manifests, and log every run step.

**Tech Stack:** Python 3.11, PyTorch, pandas, PyYAML, Pydantic, pytest, existing `mind.data`, `mind.config`, `mind.models`, and `mind.extractors` helpers.

---

## Exact Main Files To Create

- `scripts/v2/stage0_run.py`
- `src/mind/trajectory/stage0.py`
- `src/mind/trajectory/stage0_config.py`
- `src/mind/trajectory/stage0_audit.py`
- `src/mind/trajectory/stage0_split.py`
- `src/mind/trajectory/stage0_cache.py`
- `src/mind/trajectory/stage0_manifest.py`
- `tests/unit/test_v2_stage0_config.py`
- `tests/unit/test_v2_stage0_audit.py`
- `tests/unit/test_v2_stage0_split.py`
- `tests/unit/test_v2_stage0_manifest.py`
- `tests/unit/test_v2_stage0_cache.py`
- `tests/integration/test_v2_stage0_runner.py`

## Milestones 2-7 Checklist

### Milestone 2: Config And CLI

- Create `Stage0Config`, model config, dataset config, split config, and run-mode parsing in `src/mind/trajectory/stage0_config.py`.
- Create `scripts/v2/stage0_run.py` with `--config` and `--full-run`.
- Make smoke configs reject `--full-run`; make the full config require it.
- Test config loading for the three checked-in Stage 0 YAML files.
- Check: `pytest tests/unit/test_v2_stage0_config.py -q`
- Commit after the check passes.

### Milestone 3: Dataset Audit

- Create record loading and required-field checks in `src/mind/trajectory/stage0_audit.py`.
- Write `dataset_audit.csv`, `object_name_audit.csv`, `label_balance.csv`, and `sample_overlap_audit.csv`.
- Fail closed on missing required fields, duplicate sample IDs inside a dataset, missing image paths, and empty object names.
- Test audit success, audit failure, object-name counts, label counts, and overlap rows.
- Check: `pytest tests/unit/test_v2_stage0_audit.py -q`
- Commit after the check passes.

### Milestone 4: Grouped Splits

- Create deterministic split assignment in `src/mind/trajectory/stage0_split.py`.
- Use `image_id` groups with split order `encoder_train`, `bank`, `cal`, `test`.
- Use ratios `0.50`, `0.20`, `0.10`, `0.20`.
- Write split membership data for `split_manifest.json`.
- Test stable assignment, no image leakage, exact split names, and ratio rounding.
- Check: `pytest tests/unit/test_v2_stage0_split.py -q`
- Commit after the check passes.

### Milestone 5: Cache Shards And Sidecars

- Create cache writing in `src/mind/trajectory/stage0_cache.py`.
- Store shards at `outputs/v2_stage0/cache/<model_name>/<dataset_name>/<split_or_subset>/shard-00000.pt`.
- Store sidecars at the same path with `.json` appended.
- Retain all model layers with `dtype: float16`, `max_new_tokens: 1`, and `token_index: -1`.
- Reuse existing model wrappers and extractor helpers where they match the contract.
- Test shard path building, sidecar contents, dtype, full-layer shape, and row metadata.
- Check: `pytest tests/unit/test_v2_stage0_cache.py -q`
- Commit after the check passes.

### Milestone 6: Manifests And Logs

- Create manifest writing in `src/mind/trajectory/stage0_manifest.py`.
- Write `cache_manifest.json`, `split_manifest.json`, `stage0_summary.json`, and `logs/stage0_run.log`.
- Include config path, model names, dataset names, row counts, shard paths, sidecar paths, split group counts, audit CSV paths, and run status.
- Test manifest schemas, relative output paths, row-count consistency, and log creation.
- Check: `pytest tests/unit/test_v2_stage0_manifest.py -q`
- Commit after the check passes.

### Milestone 7: End-To-End Runner

- Wire the full flow in `src/mind/trajectory/stage0.py`.
- Run audit before cache extraction.
- Make the smoke command process only `smoke_limit: 8`.
- Make the full command cover POPE popular, random, and adversarial with both configured models.
- Test a tiny fake-model end-to-end run without GPU work.
- Check: `pytest tests/integration/test_v2_stage0_runner.py -q`
- Commit after the check passes.

## Required Commands From This Prompt

Run these checks before the docs commit:

```bash
git status --short
git diff --check
stale='manifest.y''aml|bank, tr''ain|valid''ation|python -m mind.trajectory.stage''0|splits.j''sonl|cache_ind''ex'
rg -n "$stale" docs/v2/STAGE0.md docs/plans/2026-05-07-v2-stage0-implementation.md
git status --short
```

Create the docs commit:

```bash
git add docs/v2/STAGE0.md docs/plans/2026-05-07-v2-stage0-implementation.md
git commit -m "docs: align v2 stage0 plan with contract"
```

Run these checks after the docs commit:

```bash
git status --short
git log -1 --oneline
```

## Commit Cadence

- Commit after each milestone's narrow check passes.
- Keep one milestone per commit.
- Do not include unrelated file changes.
