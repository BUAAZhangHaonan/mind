# MIND

MIND is a detection-only research codebase for multimodal object hallucination. The method builds local visual-grounded manifolds from grounded reference states, measures cross-layer drift at the final pre-generation token, and applies Haar wavelet features to that 16-layer drift curve before training a lightweight detector.

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

The project environment name is `mind-py311`. On this machine the stable local prefix was `/tmp/mind-py311`, so the Make targets use that path by default.

```bash
make env
make verify-env
make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
make test
```

What has been verified in this session:

- `/tmp/mind-py311/bin/python -m pytest -q tests/unit tests/integration`
- `56 passed`
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

Still missing for full multimodal runs:

- `data/coco/annotations/instances_train2017.json`
- `data/coco/train2017/`
- public H-POPE benchmark files
- full local checkpoint downloads for the planned models

## Core Scripts

The main scripts are now wired to real repo paths:

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

`scripts/cache_reference_states.py`, `scripts/build_manifolds.py`, and `scripts/compute_drift.py` accept either a single shard file or a shard directory, which makes the staged pipeline usable on real runs.

## Experiment Presets

Included presets:

- `configs/experiments/smoke/qwen3_5_4b_pope_popular.yaml`
- `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`
- `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`
- `configs/experiments/ablations/qwen3_vl_8b_popular.yaml`

Preview the planned commands for a preset:

```bash
/tmp/mind-py311/bin/python scripts/run_experiment.py \
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

## Current Status

Implemented and tested:

- typed config loading and experiment presets
- POPE, RePOPE, and H-POPE loader surface
- Qwen and InternVL model wrappers
- pre-generation hidden-state extraction
- grounded reference record expansion and cache writing
- local manifold construction
- drift and Haar wavelet feature extraction
- logistic detector training
- evaluation and RePOPE relabel support
- visualization scripts
- synthetic end-to-end integration coverage
- experiment command planning

Completed experiment checkpoints:

- real smoke run with `Qwen/Qwen3-VL-4B-Instruct` on a `200`-sample POPE popular slice
- smoke plots written under `outputs/plots/smoke-qwen3-vl-4b-popular/`
- partial medium run recovered from the completed shards of a larger `Qwen/Qwen3-VL-8B-Instruct` popular run
- partial medium metrics and ablation files written under `outputs/reports/medium-qwen3-vl-8b-popular-partial/`

Current blockers:

- the public data blockers were cleared in this session:
  - COCO train annotations downloaded and extracted
  - COCO `val2014` images downloaded and extracted
  - COCO `train2017` images downloaded and extracted
- the current hard blocker is machine-level CUDA health after a concurrent `Qwen/Qwen3-VL-8B-Instruct` run
  - GPU1 entered an unknown-error state
  - fresh PyTorch processes now report `torch.cuda.is_available() == False`
  - a local driver reset is not possible from this session without privileged access
- H-POPE remains blocked because the public benchmark package was not found

See `docs/runbooks/experiments.md` for the staged run procedure, `journal/progress.md` for the command log, and `docs/paper_outline.md` for the writing scaffold.
