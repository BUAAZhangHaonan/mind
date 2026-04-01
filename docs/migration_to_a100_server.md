# MIND Migration Handoff for the A100 Server

Last updated: 2026-04-01

## 1. Project Background

### Project name

`MIND` stands for `Multi-scale Internal Normal-residual Drift`.

### Research goal

This project studies multimodal object hallucination detection.

The core question is:

> Does hallucination leave a compressible, interpretable geometric drift signal in pre-answer hidden states that survives grouped evaluation across model families?

### Scope

The scope is intentionally narrow.

- Detection only
- Object hallucination only
- No editing or repair modules
- No segmentation
- No attention-entropy observer
- No image perturbation or counterfactual side module
- Main benchmark family: POPE
- Mandatory relabel benchmark: RePOPE
- H-POPE remains optional and was not completed because public assets were not directly available

### Method summary

The current corrected method is:

1. Extract pre-generation hidden states from selected pre-answer decoder layers.
2. Build grounded reference banks from external COCO reference images.
3. Construct local PCA manifolds from cleaned reference states.
4. Measure normalized normal residual as the geometric drift signal.
5. Keep raw drift magnitude features.
6. Build calibrated drift curves from cleaned reference-bank statistics.
7. Apply Haar wavelet features only to calibrated drift curves.
8. Train a lightweight logistic regression detector.

The current paper framing is:

- MIND is an early-warning signal, not a strongest-overall detector claim.
- The linear probe is treated as an upper baseline, not as the target to beat.
- The main value is low-dimensional geometry-aware detection, interpretability, and cross-model stability.

## 2. Current Repository State

### Remote repository

- Remote: `git@github.com:BUAAZhangHaonan/mind.git`

### Important branches

- `master`
  - stable correction-phase baseline
  - does **not** include the full paper closeout work
- `feat/mind-paper-closeout`
  - this is the branch that should be used on the new server
  - contains shared-bank control, paper export tooling, and closeout documentation

### Branch status at handoff

- `master`: `b5f6ec1`
- `feat/mind-paper-closeout`: `6c032c9`

### Recommendation

Use `feat/mind-paper-closeout` as the working branch on the A100 server.

Do not continue from `master` if the goal is to finish the paper closeout phase.

## 3. What Has Already Been Completed

### Code and infrastructure

The repo already contains:

- Typed config loading
- POPE and RePOPE normalization and loaders
- Qwen and InternVL wrappers
- Pre-generation hidden-state extraction
- Reference-bank construction
- Local PCA manifold fitting
- Drift and wavelet feature extraction
- Grouped evaluation protocols
- Baselines and ablations
- Plotting and report writing
- Shared-bank control
- Paper-package export script

### Environment and tests

The canonical environment is:

- Conda env name: `mind-py311`
- Python: `3.11`

Verified local command pattern:

```bash
conda run --no-capture-output -n mind-py311 ...
```

Core dependencies are already pinned in:

- [environment.yml](/home/d7049/zhanghaonan/mind/environment.yml)
- [requirements.txt](/home/d7049/zhanghaonan/mind/requirements.txt)

### Data already present locally

Already prepared in the source server workspace:

- `data/pope/`
- `data/repope/`
- `data/coco/annotations/`
- `data/coco/train2017/`
- `data/coco/val2014/`
- `outputs/normalized/pope/`
- `outputs/normalized/repope/`
- `outputs/reference_candidates/`

### Main completed experiment results

#### Primary corrected popular results under `image_grouped`

Qwen full MIND:

- ROC-AUC: `0.9171`
- PR-AUC: `0.2839`
- TPR@1%FPR: `0.1441`

InternVL full MIND:

- ROC-AUC: `0.9142`
- PR-AUC: `0.5438`
- TPR@1%FPR: `0.2539`

#### RePOPE relabel results

Qwen popular + RePOPE:

- ROC-AUC: `0.8887`
- PR-AUC: `0.2578`
- TPR@1%FPR: `0.1301`

InternVL popular + RePOPE:

- ROC-AUC: `0.8826`
- PR-AUC: `0.4887`
- TPR@1%FPR: `0.2089`

#### Shared-bank control

Qwen popular + shared bank:

- ROC-AUC: `0.8979`
- PR-AUC: `0.1986`

InternVL popular + shared bank:

- ROC-AUC: `0.8667`
- PR-AUC: `0.3409`

Qwen `object_heldout` + shared bank:

- ROC-AUC: `0.8624`
- PR-AUC: `0.1319`

InternVL `object_heldout` + shared bank:

- ROC-AUC: `0.8307`
- PR-AUC: `0.2544`

#### Corrected adversarial result

Qwen adversarial:

- ROC-AUC: `0.8708`
- PR-AUC: `0.2653`
- TPR@1%FPR: `0.0702`

### Current paper interpretation

The evidence now supports this reading:

- Full MIND is clearly better than drift-only and no-manifold on the main grouped protocol.
- Shared bank hurts popular performance for both models.
- Shared bank improves Qwen object transfer, but not InternVL object transfer.
- Linear probe is still stronger on PR-AUC, so the paper should not claim strongest detector.
- The paper should stay centered on low-dimensional geometry-aware early warning.

## 4. Handoff Gap And Final Resolution

At handoff time, two items were still missing:

- `InternVL adversarial` under the corrected pipeline
- the final `artifacts/paper_closeout/` export

Both were completed on `2026-04-01` on the A100 server.

Completed closeout outputs:

- `outputs/correction_phase/reports/correction-internvl3.5-8b-adversarial/metrics.json`
- `outputs/correction_phase/reports/correction-internvl3.5-8b-adversarial/results.csv`
- `artifacts/paper_closeout/`

Final InternVL adversarial result:

- `ROC-AUC 0.859557`
- `PR-AUC 0.443024`
- `TPR@1%FPR 0.142857`

## 5. Why Migration Was Needed

The current server has a hardware/runtime failure.

Observed state at handoff:

- `nvidia-smi` lists GPUs `0`, `2`, and `3`
- GPU `1` reports a handle failure
- Fresh PyTorch processes in `mind-py311` still report:
  - `torch.cuda.is_available() == False`
  - `torch.cuda.device_count() == 0`

This means the remaining GPU-dependent run cannot complete reliably on the old server.

The new server with `2 x A100 80G` should remove the current memory and device-stability bottleneck.

## 6. What To Copy To the New Server

### Required code

Clone the repository and check out:

- branch: `feat/mind-paper-closeout`

### Required data

Copy these directories if you want to avoid rebuilding them:

- `data/pope/`
- `data/repope/`
- `data/coco/annotations/`
- `data/coco/train2017/`
- `data/coco/val2014/`

### Strongly recommended outputs to copy

These are expensive or inconvenient to rebuild and should be migrated:

- `outputs/normalized/`
- `outputs/reference_candidates/`
- `outputs/cache/qwen3-vl-8b/pope/popular/`
- `outputs/cache/qwen3-vl-8b/pope/adversarial/`
- `outputs/cache/qwen3-vl-8b/pope-reference-64/train/`
- `outputs/cache/internvl3.5-8b/pope/popular/`
- `outputs/cache/internvl3.5-8b/pope-reference-64/train/`
- `outputs/correction_phase/`

### Logs worth keeping

- `logs/paper_closeout/`

These help explain the current blocker and preserve the old run trail.

## 7. Environment Setup on the A100 Server

### Recommended approach

Use the repo’s existing environment setup exactly as written.

The project already expects:

- Python `3.11`
- Conda env `mind-py311`
- PyTorch `2.6.0`
- CUDA wheel family `cu124`

### Setup commands

From the repo root:

```bash
make env
make verify-env
make install
make test
```

### Notes

- After switching branches or worktrees, run `make install` again.
- If Hugging Face access is slow, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

- The current repo uses:

```bash
conda run --no-capture-output -n mind-py311 ...
```

### A100-specific expectation

The repo itself does not require a new software stack just because the GPUs are A100s.

The important part is that the new server’s NVIDIA driver must be compatible with the installed PyTorch CUDA wheel stack. The current environment is built around PyTorch CUDA 12.4 wheels.

### Landing update on 2026-04-01

The target server verification did succeed.

- `make env`, `make verify-env`, `make install`, and `make test` now pass on `2 x NVIDIA A100 80GB PCIe`
- `make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct` succeeds
- `make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF` also succeeds, but the exact Hugging Face snapshot is required for the current wrapper path
- the local `/home/team/lvshuyang/Models/InternVL3_5-8B` folder is the GitHub-format release, so it is not a drop-in replacement for the current `AutoProcessor` path
- after the exact HF snapshot was mirrored locally, the full InternVL adversarial rerun completed and the paper package was exported

## 8. Git Requirements for the New Server

### Minimum git setup

```bash
git clone git@github.com:BUAAZhangHaonan/mind.git
cd mind
git fetch --all --prune
git checkout feat/mind-paper-closeout
```

### Required branch discipline

- Continue work on `feat/mind-paper-closeout`
- Do not continue on `master` for the migration closeout
- Keep commits small and checkpointed

### Current useful branch meanings

- `master`
  - older correction-phase baseline
- `feat/mind-paper-closeout`
  - active closeout branch to continue from
- `feat/mind-signal-correction`
  - older intermediate branch kept mostly as history

### Recommended verification

After checkout:

```bash
git status
git branch -vv
git log --oneline --decorate -6
```

You should see `feat/mind-paper-closeout` at the latest closeout doc checkpoint.

## 9. Immediate Future Plan on the A100 Server

This was the exact recommended order at handoff. It has now been completed.

### Step 1: Verify the new machine

Run:

```bash
nvidia-smi
conda run --no-capture-output -n mind-py311 python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Success condition:

- PyTorch sees the A100 GPUs correctly

### Step 2: Verify the repo environment

Run:

```bash
make verify-env
make test
```

### Step 3: Resume only the missing experiment

This step completed on `2026-04-01`.

Use `docs/runbooks/experiments.md` as the versioned source of truth for the exact commands.

Completed runtime facts:

- `24` adversarial cache shards
- `3000` cached entries
- `outputs/correction_phase/reports/correction-internvl3.5-8b-adversarial/metrics.json`

### Step 4: Export the final paper package

This step also completed on `2026-04-01`:

```bash
conda run --no-capture-output -n mind-py311 python scripts/export_paper_package.py \
  --reports-root outputs/correction_phase/reports \
  --output-root artifacts/paper_closeout
```

Completed export target:

- `artifacts/paper_closeout/tables/`
- `artifacts/paper_closeout/figures/`
- `artifacts/paper_closeout/figure_manifest.json`

## 10. Recommended Handoff Checklist

Before leaving the old server:

- Confirm the remote branch is pushed
- Copy code, data, caches, and correction outputs
- Copy `logs/paper_closeout/`

On the new A100 server:

- Clone the repo
- Check out `feat/mind-paper-closeout`
- Recreate `mind-py311`
- Run verification
- Resume only InternVL adversarial
- Export the final paper package

## 11. Short Status Summary

The migration goal is complete.

The method, corrected evaluation, shared-bank control, RePOPE relabeling, both adversarial reruns, documentation, and paper-package export are all in place.

Migration was the right move because it removed the final hardware blocker without changing the method or widening scope.
