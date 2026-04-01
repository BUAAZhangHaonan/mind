# MIND Paper Closeout Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Finish the remaining MIND paper closeout work on `feat/mind-paper-closeout`, export the final paper package, and merge the verified branch into `master`.

**Architecture:** The repo already contains the corrected detection pipeline, grouped evaluation, RePOPE relabeling, shared-bank control, and paper export code. The remaining work is narrow: recreate the canonical environment, finish the blocked InternVL adversarial rerun, export the paper package, sync the paper-facing docs, and merge the closeout branch into `master` without widening the method scope.

**Tech Stack:** Python 3.11, Conda, PyTorch 2.6.0, Transformers, scikit-learn, PyWavelets, pandas, matplotlib, git

---

## Completion Criteria

- `mind-py311` exists and passes the documented verification commands.
- `outputs/correction_phase/reports/correction-internvl3.5-8b-adversarial/metrics.json` exists.
- `artifacts/paper_closeout/` exists with `tables/`, `figures/`, and `figure_manifest.json`.
- `README.md`, `docs/results_summary.md`, `docs/paper_outline.md`, `docs/runbooks/experiments.md`, `docs/migration_to_a100_server.md`, and `journal/progress.md` agree on the final closeout state.
- `feat/mind-paper-closeout` is merged into `master` locally and the merged `master` passes the documented verification commands.

## Ordered Checklist

### 1. Sync branch and keep work isolated

- Work only from `feat/mind-paper-closeout`.
- Keep `master` untouched until final merge time.
- Record all milestone decisions and results in `journal/progress.md`.

### 2. Recreate and verify the canonical environment

- Use env name `mind-py311`.
- Run:
  - `make env`
  - `make verify-env`
  - `make install`
  - `make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct`
  - `make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF`
  - `make test`
- Export `HF_ENDPOINT=https://hf-mirror.com` before Hugging Face access.

### 3. Verify required assets only

- Confirm presence of:
  - `data/pope/`
  - `data/repope/`
  - `data/coco/annotations/`
  - `data/coco/train2017/`
  - `data/coco/val2014/`
  - `outputs/normalized/`
  - `outputs/reference_candidates/`
  - `outputs/cache/internvl3.5-8b/pope-reference-64/train/`
  - `outputs/cache/internvl3.5-8b/pope/popular/`
  - `outputs/correction_phase/`
- Reuse the existing local model directories first:
  - `/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct`
  - `/home/team/lvshuyang/Models/InternVL3_5-8B`

### 4. Run only the missing InternVL adversarial closeout path

- Verify GPU health with `nvidia-smi` and a short Torch CUDA check.
- Run `scripts/extract_eval_states.py` for InternVL adversarial on POPE adversarial.
- After cache creation, run:
  - `scripts/compute_drift.py`
  - `scripts/train_detector.py --split-strategy image_grouped --num-folds 5`
  - `scripts/evaluate.py`
  - `scripts/plot_results.py`
- If the rerun fails on a healthy A100 machine, treat it as an implementation bug in the current extraction path only.

### 5. Export the paper package and sync the docs

- Run `scripts/export_paper_package.py` after the missing InternVL adversarial report exists.
- Confirm:
  - `artifacts/paper_closeout/tables/`
  - `artifacts/paper_closeout/figures/`
  - `artifacts/paper_closeout/figure_manifest.json`
- Update the paper-facing docs with the final adversarial metrics and the final paper-safe framing.
- Clean the Qwen adversarial ablation inconsistency by either regenerating the missing ablation artifacts or removing any claim that they already exist.

### 6. Verify, commit, and merge

- Run:
  - `make verify-env`
  - `make test`
  - `pytest -q tests/integration/test_paper_export.py`
- Commit only closeout-stage changes with focused milestone messages.
- Merge `feat/mind-paper-closeout` into `master` with `--no-ff`.
- Re-run:
  - `make install`
  - `make verify-env`
  - `make test`
- Finish with a clean `git status`.
