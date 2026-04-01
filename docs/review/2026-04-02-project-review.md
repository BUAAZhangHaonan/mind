# MIND Project Review

## Quick Answer

- Conda environment used for all verified runs: `mind-py311`
- Canonical path: `/home/team/zhanghaonan/miniconda3/envs/mind-py311`
- Python version: `3.11`
- Verified runtime:
  - `torch==2.6.0+cu124`
  - `transformers==4.57.1`
  - `2 x NVIDIA A100 80GB PCIe`

## Plan Completion Check

This review checks the closeout plan in [docs/plans/2026-04-01-mind-paper-closeout-completion.md](/home/team/zhanghaonan/mind/docs/plans/2026-04-01-mind-paper-closeout-completion.md).

Plan status:

- branch isolation and closeout work on `feat/mind-paper-closeout`: completed
- canonical `mind-py311` environment recreation and verification: completed
- required data and cache verification: completed
- missing `InternVL3.5-8B` adversarial closeout path: completed
- final paper package export: completed
- paper-facing doc sync: completed
- merge of `feat/mind-paper-closeout` into `master`: completed
- post-merge verification on `master`: completed

Local repository state now:

- branch: `master`
- merge commit: `a4ba139`
- working tree: clean

## Project Scope

MIND is a detection-only project for multimodal object hallucination.

Fixed scope:

- object hallucination detection only
- no editing or repair modules
- no segmentation branch
- no attention-entropy observer branch
- no image perturbation or counterfactual editing branch
- POPE as the main benchmark
- RePOPE as the relabel follow-up on the same predictions

Method summary:

- build object-conditioned grounded reference banks from reference images
- clean the banks with `parsed_answer == 1`
- measure local manifold deviation with normalized normal residual
- build a 16-layer pre-answer drift curve
- keep raw drift magnitude features
- calibrate drift with cleaned bank statistics
- apply Haar wavelet features only to the calibrated curve
- train a lightweight logistic detector

## Repository Structure

Core code layout:

- [src/mind/data](/home/team/zhanghaonan/mind/src/mind/data)
- [src/mind/models](/home/team/zhanghaonan/mind/src/mind/models)
- [src/mind/extractors](/home/team/zhanghaonan/mind/src/mind/extractors)
- [src/mind/manifolds](/home/team/zhanghaonan/mind/src/mind/manifolds)
- [src/mind/drift](/home/team/zhanghaonan/mind/src/mind/drift)
- [src/mind/wavelets](/home/team/zhanghaonan/mind/src/mind/wavelets)
- [src/mind/detectors](/home/team/zhanghaonan/mind/src/mind/detectors)
- [src/mind/evaluation](/home/team/zhanghaonan/mind/src/mind/evaluation)
- [src/mind/visualization](/home/team/zhanghaonan/mind/src/mind/visualization)

Main runnable scripts:

- [scripts/prepare_data.py](/home/team/zhanghaonan/mind/scripts/prepare_data.py)
- [scripts/cache_reference_states.py](/home/team/zhanghaonan/mind/scripts/cache_reference_states.py)
- [scripts/extract_eval_states.py](/home/team/zhanghaonan/mind/scripts/extract_eval_states.py)
- [scripts/build_manifolds.py](/home/team/zhanghaonan/mind/scripts/build_manifolds.py)
- [scripts/compute_drift.py](/home/team/zhanghaonan/mind/scripts/compute_drift.py)
- [scripts/train_detector.py](/home/team/zhanghaonan/mind/scripts/train_detector.py)
- [scripts/compute_baselines.py](/home/team/zhanghaonan/mind/scripts/compute_baselines.py)
- [scripts/evaluate.py](/home/team/zhanghaonan/mind/scripts/evaluate.py)
- [scripts/plot_results.py](/home/team/zhanghaonan/mind/scripts/plot_results.py)
- [scripts/export_paper_package.py](/home/team/zhanghaonan/mind/scripts/export_paper_package.py)
- [scripts/verify_env.py](/home/team/zhanghaonan/mind/scripts/verify_env.py)

## Data And Models

Available benchmark files:

- `data/pope/random.jsonl`: `3000` rows
- `data/pope/popular.jsonl`: `3000` rows
- `data/pope/adversarial.jsonl`: `3000` rows
- `data/repope/random.jsonl`: `2774` rows
- `data/repope/popular.jsonl`: `2727` rows
- `data/repope/adversarial.jsonl`: `2684` rows

Normalized benchmark outputs:

- `outputs/normalized/pope/*.jsonl`
- `outputs/normalized/repope/*.jsonl`

Model families used in the closeout:

- `Qwen/Qwen3-VL-8B-Instruct`
- `OpenGVLab/InternVL3_5-8B-HF`

Important model-loading note:

- the compatible InternVL path for the current wrapper is the Hugging Face `OpenGVLab/InternVL3_5-8B-HF` layout
- the local `/home/team/lvshuyang/Models/InternVL3_5-8B` directory is the GitHub-format release, which is not a drop-in replacement for the current `AutoProcessor` path

## End-to-End Pipeline

### 1. Data preparation

- POPE and RePOPE files were normalized into a stable repo format
- reference candidate generation was prepared from COCO annotations

### 2. Hidden-state extraction

- pre-answer hidden states are extracted before free-form answer continuation
- selected layers are cached into shard files
- batched extraction support was added so real 8B runs are practical

### 3. Reference-bank construction

- grounded reference states are cleaned with `parsed_answer == 1`
- object-conditioned local banks are built by model, object, and layer
- a shared-bank control is also available for transfer analysis

### 4. Drift and feature construction

- raw normalized normal residual is computed layer by layer
- calibrated drift uses cleaned-bank mean and standard deviation
- raw magnitude is kept
- Haar features are computed only from the calibrated curve

### 5. Detector training and evaluation

- logistic regression is the main detector
- grouped protocols:
  - `row`
  - `image_grouped`
  - `object_heldout`
- baselines:
  - drift-only
  - no-manifold
  - linear probe

### 6. Export

- final tables and figures are exported by [scripts/export_paper_package.py](/home/team/zhanghaonan/mind/scripts/export_paper_package.py)
- output bundle: [artifacts/paper_closeout](/home/team/zhanghaonan/mind/artifacts/paper_closeout)

## Main Results

The final grouped closeout table is in [table1_main_grouped_results.csv](/home/team/zhanghaonan/mind/artifacts/paper_closeout/tables/table1_main_grouped_results.csv).

### Primary grouped protocol: `image_grouped`

Qwen popular:

- full MIND: `ROC-AUC 0.9171`, `PR-AUC 0.2839`, `TPR@1%FPR 0.1441`
- drift-only: `ROC-AUC 0.8497`, `PR-AUC 0.1253`
- no-manifold: `ROC-AUC 0.8385`, `PR-AUC 0.1983`
- linear probe: `ROC-AUC 0.9161`, `PR-AUC 0.3803`

InternVL popular:

- full MIND: `ROC-AUC 0.9142`, `PR-AUC 0.5438`, `TPR@1%FPR 0.2539`
- drift-only: `ROC-AUC 0.8802`, `PR-AUC 0.4270`
- no-manifold: `ROC-AUC 0.8559`, `PR-AUC 0.4033`
- linear probe: `ROC-AUC 0.9367`, `PR-AUC 0.6551`

### RePOPE relabel follow-up

Qwen popular + RePOPE:

- `ROC-AUC 0.8887`
- `PR-AUC 0.2578`
- `TPR@1%FPR 0.1301`

InternVL popular + RePOPE:

- `ROC-AUC 0.8826`
- `PR-AUC 0.4887`
- `TPR@1%FPR 0.2089`

### Adversarial closeout

Qwen adversarial:

- `ROC-AUC 0.8708`
- `PR-AUC 0.2653`
- `TPR@1%FPR 0.0702`

InternVL adversarial:

- `ROC-AUC 0.8596`
- `PR-AUC 0.4430`
- `TPR@1%FPR 0.1429`

### Object transfer boundary

The object-transfer table is in [table3_object_transfer_boundary.csv](/home/team/zhanghaonan/mind/artifacts/paper_closeout/tables/table3_object_transfer_boundary.csv).

Qwen `object_heldout`:

- object bank: `ROC-AUC 0.7244`, `PR-AUC 0.0638`
- shared bank: `ROC-AUC 0.8624`, `PR-AUC 0.1319`
- linear probe: `ROC-AUC 0.7432`, `PR-AUC 0.1233`

InternVL `object_heldout`:

- object bank: `ROC-AUC 0.8398`, `PR-AUC 0.4097`
- shared bank: `ROC-AUC 0.8307`, `PR-AUC 0.2544`
- linear probe: `ROC-AUC 0.7830`, `PR-AUC 0.2707`

### Structure comparison

The structure-comparison table is in [table2_structure_comparison.csv](/home/team/zhanghaonan/mind/artifacts/paper_closeout/tables/table2_structure_comparison.csv).

Main reading:

- full MIND beats drift-only and no-manifold on the primary grouped protocol for both model families
- linear probe is still the stronger upper baseline on PR-AUC
- shared bank hurts popular performance on both model families
- shared bank helps Qwen transfer but not InternVL transfer

## Paper Achievements

This is the paper-facing achievement list.

### Method achievements

- completed a full detection-only multimodal hallucination pipeline
- corrected the earlier signal-definition mistake by preserving raw drift magnitude
- added cleaned-bank calibration
- restricted Haar features to calibrated drift only
- kept the method simple and faithful to the original closeout scope

### Experiment achievements

- completed corrected `image_grouped` reruns on both Qwen and InternVL
- completed corrected `row` comparisons
- completed corrected `object_heldout` comparisons
- completed RePOPE relabel evaluation on corrected predictions
- completed shared-bank control experiments
- completed both adversarial closeout reruns

### Paper-position achievements

- established that the MIND signal is real and useful on the primary grouped protocol
- showed that the paper should be framed around low-dimensional geometry-aware early warning
- avoided the unsupported claim that MIND is the strongest overall detector
- clarified that the more interesting follow-up question is cross-object geometry, not a larger detector head

### Artifact achievements

- exported all three paper tables
- exported all three paper figures
- generated [figure_manifest.json](/home/team/zhanghaonan/mind/artifacts/paper_closeout/figure_manifest.json)
- kept the final repo narrative aligned across README, runbook, results summary, paper outline, and journal

## Verification

Verified locally on the merged `master` branch:

- `make install`
- `make verify-env`
- `make test`
- `pytest -q tests/integration/test_paper_export.py`

Observed results:

- `make verify-env`: passed
- `make test`: `93 passed`
- export integration test: `1 passed`
- `git status`: clean

## Important Fixes During Closeout

- added fail-fast behavior in the feature path so missing reference coverage raises an error instead of silently dropping rows
- removed the dead public export of the retired drift-normalization helper
- verified the guardrail change with focused unit tests before running the final InternVL adversarial closeout

## Final Deliverables

Main review files:

- [README.md](/home/team/zhanghaonan/mind/README.md)
- [docs/results_summary.md](/home/team/zhanghaonan/mind/docs/results_summary.md)
- [docs/paper_outline.md](/home/team/zhanghaonan/mind/docs/paper_outline.md)
- [docs/runbooks/experiments.md](/home/team/zhanghaonan/mind/docs/runbooks/experiments.md)
- [journal/progress.md](/home/team/zhanghaonan/mind/journal/progress.md)
- [docs/migration_to_a100_server.md](/home/team/zhanghaonan/mind/docs/migration_to_a100_server.md)

Paper package:

- [artifacts/paper_closeout](/home/team/zhanghaonan/mind/artifacts/paper_closeout)

## Remaining Non-Blocking Notes

- H-POPE is still not included because the public benchmark package was not available in a directly usable form
- this review confirms local completion; it does not imply a remote push has already been done
