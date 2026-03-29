# MIND Paper Outline

## Title

MIND: Multi-scale Internal Drift on Visual-grounded Manifolds for MLLM Hallucination Detection

## One-Sentence Claim

Object hallucination can be detected before final answer generation by measuring how selected hidden states drift away from local visual-grounded manifolds, then modeling that drift across layers with wavelet features.

## Abstract Skeleton

Problem:
- Current multimodal hallucination detectors mostly judge final outputs or external signals, but useful evidence often appears earlier in internal states.

Method:
- Build local manifolds from grounded reference states.
- Measure normalized normal residual drift across selected layers.
- Apply Haar wavelet features to the drift curve.
- Train a lightweight detector on top of those features.

Result:
- Report overall and per-subset gains on POPE.
- Re-evaluate the same predictions with RePOPE relabeling.
- Show ablations for no-manifold, no-wavelet, and layer-range variants.
- Report that the completed InternVL popular run is stronger than the completed Qwen popular run in ROC-AUC, while the direct linear probe still remains strongest inside each model family.

Takeaway:
- Internal manifold drift is an interpretable and useful early warning signal for object hallucination.

## Section Plan

### 1. Introduction

- Hallucination detection is usually treated as an output judgment problem.
- Internal states can expose failure earlier than final text.
- Existing work motivates internal-layer analysis, but the geometry-plus-dynamics view is still missing.
- MIND proposes that missing view.

### 2. Related Work

- Object hallucination benchmarks and detectors
- Internal-state probing for LVLM hallucination
- Representation geometry and local manifolds
- Frequency and multi-scale analysis for unstable trajectories

### 3. Method

#### 3.1 Local Visual-grounded Reference Bank

- Build per-object reference states from grounded samples.
- Separate reference images from evaluation images.

#### 3.2 Local Manifold Construction

- Use kNN and local PCA for each object and layer.
- Keep one main score, the normalized normal residual.

#### 3.3 Cross-layer Drift Signal

- Extract 16 middle-layer vectors at the final prefill token.
- Compute one drift value per selected layer.

#### 3.4 Wavelet Feature Extraction

- Apply Haar wavelet decomposition to the drift curve.
- Use approximation and detail energies plus raw drift features.

#### 3.5 Lightweight Detection

- Logistic regression as the main detector.
- Optional MLP only as a secondary check.

### 4. Experiments

- Main benchmark: POPE
- Relabel check: RePOPE
- Optional extension: H-POPE
- Main model: Qwen3-VL-8B-Instruct
- Cross-family validation: InternVL3.5-8B-HF

### 5. Ablations and Analysis

- No manifold
- No wavelet
- Early vs middle vs late layers
- Drift-curve visualization
- Wavelet heatmaps

### 6. Discussion

- Why local manifolds instead of global geometry
- Why wavelets on drift curves instead of raw hidden states
- Why the first version stays detection-only

## Core Figures

1. Method overview diagram
2. Drift curve comparison for grounded vs hallucinated samples
3. Wavelet energy heatmap across subsets or labels
4. Main ROC curves
5. Ablation bar chart

## Core Tables

1. Main POPE results by subset
2. RePOPE relabel results on the same predictions
3. Cross-model comparison
4. Ablation results

## Main Experimental Story

1. Show that drift-based features provide a useful early ranking signal before final answer generation.
2. Show that manifold drift is interpretable and useful even when a direct hidden-state probe is stronger in raw ROC-AUC on the completed Qwen runs.
3. Show that wavelet features add modest value over raw drift alone on the corrected Qwen runs.
4. Report layer-location findings empirically; in the corrected popular run, late layers were strongest.
5. Confirm that the main ranking conclusions still hold under RePOPE relabeling and cross-family validation.
