# MIND Remaining Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the remaining MIND stages on the recovered 3 x RTX 3090 machine and leave the repository in a reproducible, push-ready state with final metrics, plots, and written conclusions.

**Architecture:** The codebase is already built. The remaining work is execution and documentation: finish the InternVL cross-family popular run, verify the H-POPE asset blocker, then update the committed docs so the final state matches the actual outputs and conclusions. The scope stays detection-only.

**Tech Stack:** Python 3.11, PyTorch, Transformers, scikit-learn, PyWavelets, pandas, matplotlib, seaborn, conda, git

---

## Completion Criteria

- The current machine state is rechecked and recorded as 3 usable RTX 3090 GPUs with `mind-py311` as the only supported environment.
- The InternVL popular cross-family run finishes and writes:
  - `outputs/cache/internvl3.5-8b/pope-reference-64/train/`
  - `outputs/cache/internvl3.5-8b/pope/popular/`
  - `outputs/reference_banks/internvl3.5-8b/`
  - `outputs/features/cross-internvl3.5-8b-popular/popular.parquet`
  - `outputs/reports/cross-internvl3.5-8b-popular/`
  - `outputs/reports/cross-internvl3.5-8b-popular-repope/`
  - `outputs/plots/cross-internvl3.5-8b-popular/`
- The H-POPE status is explicit: either public assets are found and runnable, or the exact blocker is written into the docs and journal.
- The committed docs match the real state of the repo and the real experiment results.
- Final verification passes with fresh command output.
- The remaining milestone commits are created cleanly and the repository is pushed to the remote if auth works.

## Ordered Checklist

### 1. Reconfirm runtime state

- Check `nvidia-smi`, `nvcc --version`, and `torch.cuda.device_count()`.
- Check `conda env list` and confirm `mind-py311` is the only project environment.
- Check `git status`, `git remote -v`, and SSH auth to GitHub.

### 2. Finish the InternVL cross-family run

- Let the live reference extraction finish.
- Verify the reference cache reached the full expected shard count.
- Run or continue:
  - evaluation-state extraction on POPE popular
  - manifold building
  - drift feature computation
  - detector training
  - POPE evaluation
  - RePOPE relabel evaluation
  - baseline and ablation reporting
  - plotting
- Record the final metrics and output paths.

### 3. Resolve H-POPE status

- Search for directly usable public H-POPE assets.
- If no public asset package is reachable, keep the existing loader/config surface and document the blocker clearly.

### 4. Update the written project state

- Update `README.md`.
- Update `journal/progress.md`.
- Update `docs/runbooks/experiments.md`.
- Update `docs/paper_outline.md`.
- Add a compact final result summary under `docs/`.

### 5. Verify and publish

- Run `make verify-env`.
- Run `PYTHONWARNINGS=ignore make test`.
- Recheck `git status`.
- Commit the final doc and cleanup updates with clean milestone messages.
- Push the completed history to the GitHub remote.
