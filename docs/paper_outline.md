# MIND Paper Outline

## Title

MIND: Multi-scale Internal Normal-residual Drift for Early Multimodal Hallucination Detection

## Name Meaning

- `MIND` here expands to `Multi-scale Internal Normal-residual Drift`
- `normal-residual` is the core quantity measured against the local visual-grounded manifold
- `drift` is the cross-layer trajectory built from that quantity before answer generation

## One-Sentence Claim

Pre-answer hidden states carry a real geometry-aware hallucination signal, and a low-dimensional manifold-drift representation can detect that signal early and interpretably even when a high-dimensional linear probe is still stronger in raw PR-AUC.

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

- Primary protocol: `image_grouped` split on POPE popular
- Secondary protocol: `object_heldout`
- Main comparisons: drift-only, no-manifold, and linear probe

Result:

- On corrected `image_grouped` evaluation, full MIND beats corrected drift-only and corrected no-manifold on both Qwen and InternVL.
- The linear probe still leads on PR-AUC on the primary grouped protocol.
- InternVL is notably more stable than Qwen under `object_heldout`.

Takeaway:

- The paper should argue for geometry-aware early warning, not strongest overall detector performance.

## Section Plan

### 1. Introduction

- Object hallucination is still mostly framed as an output-side problem.
- Internal states let us ask an earlier and cleaner question: does the model leave the visual-grounded region before it answers?
- The main challenge is not whether a signal exists. It is whether that signal can be kept in a low-dimensional form without throwing away the useful magnitude information.
- MIND answers that by measuring manifold normal-residual drift across pre-answer layers.

### 2. Related Work

- POPE-style object hallucination benchmarking and relabeling
- Internal-state probing and early warning for multimodal hallucination
- Local representation geometry and manifold residuals
- Multi-scale analysis across layers as a descriptive tool, not the theoretical center

### 3. Method

#### 3.1 Clean Grounded Reference Bank

- Build the bank from external COCO reference images only.
- Keep only entries with `parsed_answer == 1`.
- Store both support counts and leave-one-out residual statistics for each object and layer.

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
- Upper baseline: direct hidden-state linear probe
- Main protocol: `image_grouped`
- Secondary protocol: `object_heldout`

### 4. Experiments

- Benchmark: POPE popular for the correction phase
- Models:
  - `Qwen/Qwen3-VL-8B-Instruct`
  - `OpenGVLab/InternVL3_5-8B-HF`
- Baselines:
  - raw yes-no answer performance
  - drift-only
  - no-manifold
  - linear probe

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

#### 5.3 Held-out object behavior

- Qwen drops sharply under `object_heldout`.
- InternVL stays much stronger and full MIND even beats the InternVL linear probe on that secondary protocol.

### 6. Discussion

- The core idea is manifold drift, not Haar itself.
- Wavelets stay in the paper as a compact way to describe variation across layers.
- The corrected results support a modest but solid claim:
  - low-dimensional geometry-aware early warning works
  - it is interpretable
  - it is not the strongest detector in every setting
- The old middle-layer story should be removed.
- The safer wording is `selected pre-answer layers`, with late-layer strength treated as an empirical result rather than a fixed theoretical claim.

## Core Figures

1. Method overview: grounded bank -> manifold residual -> calibrated drift -> detector
2. Corrected Qwen popular drift curves and ROC
3. Corrected InternVL popular drift curves and ROC
4. Protocol comparison figure from `outputs/correction_phase/plots/correction_summary_protocols.png`
5. Reference-bank cleaning and support table

## Core Tables

1. Primary `image_grouped` results on Qwen and InternVL
2. Structural baseline comparison: full vs drift-only vs no-manifold vs linear probe
3. Secondary `object_heldout` results
4. Reference-bank cleaning and support counts

## Writing Direction

Use this framing:

- The first version of MIND was held back by a signal-definition mistake.
- Once raw magnitude was preserved and the bank was cleaned, the geometry signal became clearly useful.
- The strongest final claim is not raw performance supremacy.
- The strongest final claim is that object hallucination leaves an interpretable pre-answer geometric trace that can be compressed into a stable low-dimensional detector.
