# MIND

MIND is a detection-only research codebase for multimodal hallucination in open MLLMs. The method treats hallucination as a pre-generation representation drift problem. It builds local visual-grounded manifolds from grounded reference states, measures cross-layer normal residual drift for each evaluation sample, and applies Haar wavelet features to that drift curve before training a lightweight detector.

The repository stays focused on object hallucination detection. It does not include editing, segmentation, attention-entropy observers, or image-side counterfactual perturbations.

## Method

The working hypothesis is simple.

1. Grounded samples define local manifold neighborhoods in the model's middle layers.
2. Hallucinated samples drift away from those neighborhoods before the final answer token is produced.
3. That drift is not only large, but unstable across layers.
4. Haar wavelet features over the drift curve make that instability easier to detect with a lightweight classifier.

## Repository Layout

```text
mind/
  configs/
    data/
    experiments/
    models/
  docs/
    paper_outline.md
    runbooks/
  journal/
    progress.md
  scripts/
  src/mind/
    config/
    data/
    detectors/
    drift/
    evaluation/
    extractors/
    manifolds/
    models/
    visualization/
    wavelets/
  tests/
```

## Environment

The practical environment name is `mind-py311`.

On this machine, the only stable conda prefix found during execution was `/tmp/mind-py311`, so the repo Make targets use that path directly.

```bash
make env
make verify-env
make test
```

The verification script checks:

- core imports
- CUDA visibility
- visible GPU names
- optional Hugging Face config loading when a model id is passed

Important environment defaults live in `env.example`. If Hugging Face access is unstable, export `HF_ENDPOINT=https://hf-mirror.com` before downloads.

## Data Scope

Current benchmark scope:

- `POPE` as the main benchmark
- `RePOPE` as label overrides on the same prediction table
- `H-POPE` as an optional extension if public assets are available locally

Expected local roots:

- `data/pope`
- `data/repope`
- `data/hpope`
- `data/coco`

## Available Scripts

- `scripts/prepare_data.py`
- `scripts/cache_reference_states.py`
- `scripts/extract_eval_states.py`
- `scripts/build_manifolds.py`
- `scripts/compute_drift.py`
- `scripts/train_detector.py`
- `scripts/evaluate.py`
- `scripts/plot_results.py`
- `scripts/run_experiment.py`
- `scripts/verify_env.py`

## Experiment Presets

Included experiment presets:

- `configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml`
- `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`
- `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`
- `configs/experiments/ablations/qwen3_vl_8b_popular.yaml`

The run order is:

1. Smoke run
2. Medium run
3. Main POPE run
4. Ablations
5. RePOPE relabel evaluation
6. H-POPE if public assets are available

See `docs/runbooks/experiments.md` for the staged procedure.

## Outputs

The canonical artifact layout is:

- `outputs/cache/<model>/<dataset>/<split>/shard-xxxxx.pt`
- `outputs/reference_banks/<model>/<object>/layer-xx.pt`
- `outputs/features/<experiment>/<split>.parquet`
- `outputs/reports/<experiment>/metrics.json`
- `outputs/reports/<experiment>/results.csv`
- `outputs/plots/<experiment>/*.png`

## Current Status

Implemented and tested:

- typed config loading
- POPE, RePOPE, and H-POPE data interfaces
- model wrapper registry for Qwen and InternVL
- pre-generation state extraction helpers
- local PCA manifold construction
- drift and wavelet feature extraction
- logistic detector training helper
- evaluation metrics and report writing
- plot generation helpers
- synthetic end-to-end integration test

Current practical blocker:

- real end-to-end benchmark runs were not completed in this session because direct Hugging Face access was unavailable, and actual model downloads are still required before the real extraction and evaluation stages can run on POPE

See `journal/progress.md` for the execution log and `docs/paper_outline.md` for the paper structure.
