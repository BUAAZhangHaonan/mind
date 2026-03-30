# MIND

MIND stands for `Multi-scale Internal Normal-residual Drift`. It is a detection-only research codebase for multimodal object hallucination. The corrected method builds local visual-grounded manifolds from grounded reference states, preserves the raw normal-residual magnitude, calibrates cross-layer drift with cleaned reference-bank statistics, and applies Haar wavelet features only to the calibrated 16-layer drift curve before training a lightweight detector.

The scope is fixed:

- object hallucination detection only
- no editing or repair modules
- no segmentation, attention-entropy observers, or image-side counterfactual modules
- POPE is the main benchmark
- RePOPE is a mandatory relabeling pass on the same predictions
- H-POPE stays optional and blocked until public files are actually available

## Repository Layout

```text
mind/
  configs/
  docs/
  journal/
  scripts/
  src/mind/
  tests/
```

Key modules:

- `src/mind/data`: POPE, RePOPE, H-POPE loaders and reference candidate utilities
- `src/mind/models`: Qwen and InternVL wrapper layer
- `src/mind/extractors`: pre-generation hidden-state extraction helpers
- `src/mind/manifolds`: local PCA manifolds and normalized normal residual
- `src/mind/drift`: drift-curve construction
- `src/mind/wavelets`: Haar wavelet features over drift curves
- `src/mind/detectors`: logistic detector
- `src/mind/evaluation`: metrics and report writers
- `src/mind/visualization`: drift, ROC, heatmap, and ablation plots

## Environment

The project environment name is `mind-py311`. The canonical runtime is the named conda environment, and all documented commands now use `conda run --no-capture-output -n mind-py311 ...` through the shipped `Makefile`.

```bash
make env
make verify-env
make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
make test
```

What has been verified in this session:

- `PYTHONWARNINGS=ignore conda run --no-capture-output -n mind-py311 python -m pytest -q tests/unit tests/integration`
- `83 passed`
- if you switch branches or worktrees, rerun `make install` so the editable package points at the active checkout
- `scripts/verify_env.py` succeeded for:
  - `Qwen/Qwen3-VL-8B-Instruct`
  - `OpenGVLab/InternVL3_5-8B-HF`

If Hugging Face access is slow, export `HF_ENDPOINT=https://hf-mirror.com`.

## Local Data Status

Question files already downloaded locally:

- `data/pope/random.jsonl` with 3000 rows
- `data/pope/popular.jsonl` with 3000 rows
- `data/pope/adversarial.jsonl` with 3000 rows
- `data/repope/random.jsonl` with 2774 rows
- `data/repope/popular.jsonl` with 2727 rows
- `data/repope/adversarial.jsonl` with 2684 rows

Normalized copies already written to:

- `outputs/normalized/pope/*.jsonl`
- `outputs/normalized/repope/*.jsonl`

Local multimodal assets already present:

- `data/coco/annotations/instances_train2017.json`
- `data/coco/train2017/`
- `data/coco/val2014/`
- model assets for the completed Qwen stages

Still not publicly obtainable in a directly usable package:

- H-POPE benchmark files

## Core Scripts

The main scripts are now wired to real repo paths:

- `scripts/prepare_data.py`
- `scripts/cache_reference_states.py`
- `scripts/extract_eval_states.py`
- `scripts/build_manifolds.py`
- `scripts/compute_drift.py`
- `scripts/train_detector.py`
- `scripts/evaluate.py`
- `scripts/export_paper_package.py`
- `scripts/plot_results.py`
- `scripts/run_experiment.py`
- `scripts/verify_env.py`

`scripts/cache_reference_states.py`, `scripts/build_manifolds.py`, and `scripts/compute_drift.py` accept either a single shard file or a shard directory, which makes the staged pipeline usable on real runs.

## Experiment Presets

Included presets:

- `configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml`
- `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`
- `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`
- `configs/experiments/ablations/qwen3_vl_8b_popular.yaml`

Preview the planned commands for a preset:

```bash
conda run --no-capture-output -n mind-py311 python scripts/run_experiment.py \
  --config configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml \
  --stages prepare,extract_eval
```

Run the full stage list once the remaining assets are in place:

```bash
make plan-smoke
```

## Outputs

Canonical artifact layout:

- `outputs/cache/<model>/<dataset>/<split>/shard-xxxxx.pt`
- `outputs/reference_banks/<model>/<object>/layer-xx.pt`
- `outputs/features/<experiment>/<split>.parquet`
- `outputs/reports/<experiment>/metrics.json`
- `outputs/reports/<experiment>/results.csv`
- `outputs/plots/<experiment>/*.png`
- corrected signal-evaluation reruns in this phase were written under `outputs/correction_phase/`

## Current Status

Implemented and verified:

- typed config loading and experiment presets
- POPE, RePOPE, and H-POPE loader surface
- Qwen and InternVL model wrappers
- pre-generation hidden-state extraction
- grounded reference record expansion and cache writing
- local manifold construction with cleaned reference-bank stats
- corrected drift and Haar wavelet feature extraction
- grouped evaluation protocols: `row`, `image_grouped`, and `object_heldout`
- shared-bank control with `bank_scope = object | shared`
- logistic detector training, baseline comparison, and plotting
- script-generated paper closeout export for the three main tables and three main figures
- synthetic end-to-end integration coverage and corrected full test suite

Correction-phase experiment checkpoints completed on the existing popular caches:

- cleaned reference banks rebuilt for `qwen3-vl-8b` and `internvl3.5-8b`
- corrected Qwen popular rerun under `image_grouped`
- corrected Qwen popular rerun under legacy `row`
- corrected Qwen popular rerun under `object_heldout`
- corrected InternVL popular rerun under `image_grouped`
- corrected InternVL popular rerun under legacy `row`
- corrected InternVL popular rerun under `object_heldout`
- grouped comparison figure written to `outputs/correction_phase/plots/correction_summary_protocols.png`
- corrected popular RePOPE relabel reports written for both model families:
  - `correction-qwen3-vl-8b-popular-repope`
  - `correction-internvl3.5-8b-popular-repope`
- corrected Qwen adversarial rerun completed under `image_grouped`:
  - `ROC-AUC 0.8708`
  - `PR-AUC 0.2653`
  - `TPR@1%FPR 0.0702`

Primary corrected findings:

- `Qwen/Qwen3-VL-8B-Instruct`, `image_grouped`:
  - full MIND: `ROC-AUC 0.9171`, `PR-AUC 0.2839`, `TPR@1%FPR 0.1441`
  - drift-only: `ROC-AUC 0.8497`, `PR-AUC 0.1253`
  - no-manifold: `ROC-AUC 0.8385`, `PR-AUC 0.1983`
  - linear probe: `ROC-AUC 0.9161`, `PR-AUC 0.3803`
- `OpenGVLab/InternVL3_5-8B-HF`, `image_grouped`:
  - full MIND: `ROC-AUC 0.9142`, `PR-AUC 0.5438`, `TPR@1%FPR 0.2539`
  - drift-only: `ROC-AUC 0.8802`, `PR-AUC 0.4270`
  - no-manifold: `ROC-AUC 0.8559`, `PR-AUC 0.4033`
  - linear probe: `ROC-AUC 0.9367`, `PR-AUC 0.6551`
- `object_heldout` is mixed:
  - Qwen drops to `ROC-AUC 0.7244`, `PR-AUC 0.0638`
  - InternVL holds at `ROC-AUC 0.8398`, `PR-AUC 0.4097`

Current paper-safe interpretation:

- the corrected MIND signal is real and materially stronger than corrected drift-only and corrected no-manifold on the primary grouped protocol for both model families
- the direct hidden-state linear probe still keeps a clear PR-AUC advantage on the primary grouped protocol, so the repo should not claim strongest overall detector performance
- the strongest current framing is low-dimensional geometry-aware early warning, interpretability, and cross-model stability
- the closeout control question is now object-conditioned bank versus shared bank, not detector-head escalation

Current environment note:

- as of `2026-03-31`, the machine has an intermittent GPU-side failure mode:
  - `nvidia-smi` can fall back to `Unable to determine the device handle for GPU1: 0000:3B:00.0: Unknown Error`
  - once that happens, fresh `mind-py311` PyTorch processes report `torch.cuda.is_available() == False`
  - once that happens, fresh `mind-py311` PyTorch processes report `torch.cuda.device_count() == 0`
  - this currently blocks both the pooled shared-bank closeout control and the fresh InternVL adversarial extraction
- older `3 GPU` notes in the journal remain historical incident notes, but the live state is now less stable than the recovered `4 GPU` snapshot from `2026-03-30`
- H-POPE remains blocked because the public benchmark package was not found in a directly usable release

See `docs/results_summary.md` for the corrected tables, `docs/runbooks/experiments.md` for the staged and corrected commands, `journal/progress.md` for the full command log, and `docs/paper_outline.md` for the revised writing direction.
