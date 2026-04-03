# MIND Paper Outline

## Title

MIND: Multi-scale Internal Normal-residual Drift for Early Object Hallucination Detection

## Name Meaning

- `MIND` here expands to `Multi-scale Internal Normal-residual Drift`
- `normal-residual` is the core quantity measured against the local visual-grounded manifold
- `drift` is the cross-layer trajectory built from that quantity before answer generation

## One-Sentence Claim

Object hallucination leaves a compressible, interpretable geometric drift signal in pre-answer hidden states, and that low-dimensional signal survives grouped evaluation across model families even when a full hidden-state linear probe still performs as an upper baseline.

## Abstract Skeleton

Problem:

- Many multimodal hallucination studies judge the final output only.
- That misses earlier internal evidence and often hides what the model is doing before it answers.

Method:

- Build object-conditioned local manifolds from grounded reference states.
- Keep the raw normal-residual magnitude as the main geometric signal.
- Calibrate that signal with cleaned reference-bank statistics.
- Apply Haar only to the calibrated cross-layer drift curve.
- Train a lightweight logistic detector on the combined raw-plus-calibrated feature set.

Evaluation:

- Current evidence: `image_grouped` on POPE popular, RePOPE relabeling, and POPE adversarial
- Next-round expansion: DASH-B on the same object yes-no task
- Main comparisons: output-side confidence baselines, drift-only, no-manifold, and linear probe

Result:

- On corrected `image_grouped` evaluation, full MIND beats corrected drift-only and corrected no-manifold on both Qwen and InternVL.
- The linear probe still leads on PR-AUC on the primary grouped protocol.
- InternVL is notably more stable than Qwen under `object_heldout`.
- RePOPE popular relabeling preserves the main ordering on the corrected popular predictions.
- Adversarial closeout also completed for both model families:
  - Qwen adversarial: `ROC-AUC 0.8708`, `PR-AUC 0.2653`
  - InternVL adversarial: `ROC-AUC 0.8596`, `PR-AUC 0.4430`

Takeaway:

- The paper should argue for geometry-aware early warning and low-dimensional structure, not strongest overall detector performance.

## Section Plan

### 1. Introduction

- Object hallucination is still mostly framed as an output-side problem.
- Internal states let us ask an earlier and cleaner question: does the model leave the visual-grounded region before it answers?
- The main challenge is not whether a signal exists. It is whether that signal can be kept in a low-dimensional form without throwing away the useful magnitude information.
- MIND answers that by measuring manifold normal-residual drift across pre-answer layers.

### 2. Related Work

- POPE-style object hallucination benchmarking and relabeling
- Output-side confidence and logit-margin hallucination baselines
- Internal-state probing and early warning for multimodal hallucination
- Local representation geometry and manifold residuals
- Multi-scale analysis across layers as a descriptive tool, not the theoretical center

### 3. Method

#### 3.1 Clean Grounded Reference Bank

- Build the bank from external COCO reference images only.
- Keep only entries with `parsed_answer == 1`.
- Store both support counts and leave-one-out residual statistics for each object and layer.
- Add one closeout control only:
  - object-conditioned bank as the main method
  - shared bank pooled by `model + layer` as the transfer control

#### 3.2 Local Manifold Score

- For each object and layer, normalize reference states.
- Use local PCA on the nearest neighbors.
- Measure one quantity only: normalized normal residual.

#### 3.3 Cross-layer Drift Signal

- Use the final prefill token right before answer generation.
- Extract 16 selected pre-answer layers from the fixed cache setting.
- Form a raw 16-step drift curve from manifold residuals.

#### 3.4 Raw and Calibrated Features

- Keep raw magnitude features unchanged.
- Build calibrated curves from cleaned-bank mean and standard deviation.
- Apply Haar to calibrated curves only.
- Combine raw magnitude and calibrated wavelet features in one lightweight detector input.

#### 3.5 Detector and Protocol

- Headline detector: logistic regression
- Output-side checks: `p_yes`, yes-minus-no logit margin, and chosen-answer confidence
- Upper baseline: direct hidden-state linear probe
- Main protocol: `image_grouped`
- Secondary protocol: `object_heldout`

### 4. Experiments

- Current benchmark set:
  - POPE popular
  - POPE adversarial
  - RePOPE relabeling of the popular predictions
- Next benchmark to add:
  - DASH-B
- Current models:
  - `Qwen/Qwen3-VL-8B-Instruct`
  - `OpenGVLab/InternVL3_5-8B-HF`
- Next models to add:
  - `llava-hf/llava-onevision-qwen2-7b-ov-hf`
  - `allenai/Molmo-7B-D-0924`
- Baselines:
  - `p_yes`
  - yes-minus-no logit margin
  - chosen-answer confidence
  - drift-only
  - no-manifold
  - linear probe
  - raw curve only
  - raw + calibrated simple stats
  - raw + calibrated full curve
  - raw + calibrated Haar

### 5. Main Results

#### 5.1 Primary grouped result

- Qwen, `image_grouped`:
  - full MIND: `ROC-AUC 0.9171`, `PR-AUC 0.2839`
  - linear probe: `ROC-AUC 0.9161`, `PR-AUC 0.3803`
- InternVL, `image_grouped`:
  - full MIND: `ROC-AUC 0.9142`, `PR-AUC 0.5438`
  - linear probe: `ROC-AUC 0.9367`, `PR-AUC 0.6551`

#### 5.2 Structural comparisons

- On both model families, full MIND beats drift-only and no-manifold on the primary grouped protocol.
- The correction therefore repaired a real signal, not just a reporting issue.
- The shared-bank control hurts popular performance on both model families:
  - Qwen falls from `ROC-AUC 0.9171` and `PR-AUC 0.2839` to `ROC-AUC 0.8979` and `PR-AUC 0.1986`
  - InternVL falls from `ROC-AUC 0.9142` and `PR-AUC 0.5438` to `ROC-AUC 0.8667` and `PR-AUC 0.3409`

#### 5.3 Held-out object behavior

- Qwen drops sharply under `object_heldout`.
- InternVL stays much stronger and full MIND even beats the InternVL linear probe on that secondary protocol.
- The shared-bank control changes the transfer story:
  - on Qwen, shared bank improves `object_heldout` to `ROC-AUC 0.8624`, `PR-AUC 0.1319`
  - on InternVL, shared bank is still worse than the object-conditioned bank under `object_heldout`

#### 5.4 Adversarial closeout

- Qwen adversarial:
  - full MIND: `ROC-AUC 0.8708`, `PR-AUC 0.2653`, `TPR@1%FPR 0.0702`
- InternVL adversarial:
  - full MIND: `ROC-AUC 0.8596`, `PR-AUC 0.4430`, `TPR@1%FPR 0.1429`
- Readout:
  - adversarial is harder than popular for both families
  - InternVL keeps the stronger adversarial precision-recall profile
  - the six-row grouped closeout table is now complete

### 6. Discussion

- The core idea is manifold drift, not Haar itself.
- Wavelets stay in the paper only if they beat simpler curve summaries in the Phase One ablation.
- The corrected results support a modest but solid claim:
  - low-dimensional geometry-aware early warning works
  - it is interpretable
  - it is not the strongest detector in every setting
- The old middle-layer story should be removed.
- The safer wording is `selected pre-answer layers`, with late-layer strength treated as an empirical result rather than a fixed theoretical claim.
- The bank-scope control now gives a sharper interpretation:
  - object conditioning buys accuracy on popular for both models
  - Qwen pays a much larger transfer penalty for that object conditioning than InternVL
  - the next-paper question is therefore shared cross-object geometry, not a larger detector head

## Core Figures

1. Method overview: grounded bank -> manifold residual -> calibrated drift -> detector
2. ROC + PR curves for `popular + image_grouped` on Qwen and InternVL
3. Protocol comparison figure using `row`, `image_grouped`, and `object_heldout`

## Core Tables

1. Main grouped results:
   - Qwen popular
   - Qwen popular + RePOPE
   - Qwen adversarial
   - InternVL popular
   - InternVL popular + RePOPE
   - InternVL adversarial
2. Structure comparison:
   - full MIND (object bank)
   - full MIND (shared bank)
   - drift-only
   - no-manifold
   - linear probe
3. Object transfer boundary:
   - full MIND (object bank)
   - full MIND (shared bank)
   - linear probe

## Writing Direction

Use this framing:

- The first version of MIND was held back by a signal-definition mistake.
- Once raw magnitude was preserved and the bank was cleaned, the geometry signal became clearly useful.
- The strongest final claim is not raw performance supremacy.
- The strongest final claim is that object hallucination leaves an interpretable pre-answer geometric trace that can be compressed into a stable low-dimensional early-warning signal.
- The next-paper question is shared cross-object geometry, not a larger detector head.
