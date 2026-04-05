# MIND Paper Outline

## Title

MIND: Internal Normal-residual Drift for Early Object Hallucination Detection

## Naming

- `MIND` should now be treated as the method name, not as a claim about multi-scale wavelets.
- The paper should stop expanding `MIND` as `Multi-scale Internal Normal-residual Drift`.
- The central idea is simple: measure object-conditioned manifold drift before the answer starts.

## Current Claim

Object hallucination leaves a useful pre-answer geometric signal in VLM hidden states. That signal gets stronger when raw drift is combined with simple calibrated curve statistics, but it is still not the whole story, and it should not be written up as a universal best detector.

## Phase-One Freeze

These decisions now come from the live round-two rerun on the current code path, not from the older March closeout package.

- Default full MIND feature set:
  - `raw + calibrated simple stats`
- Why:
  - Haar does not beat simple stats on either current popular rerun.
  - On Qwen popular, simple stats are better than Haar on both ROC-AUC and PR-AUC.
  - On Intern popular, simple stats are again better than Haar on both ROC-AUC and PR-AUC.
  - The confidence intervals overlap, so the right choice is the simpler feature set.
- What this means for the paper:
  - remove `Multi-scale` from the title
  - remove Haar from the headline method description
  - keep the full calibrated curve and Haar only as ablations

## Framing Decision

The current rerun does not support the pessimistic fallback that logit margin already beats MIND.

- On Qwen popular, logit margin is far below current full MIND.
- On Intern popular, logit margin is far below current full MIND.
- So the paper can still make a modest detector-performance claim.
- The safer wording is:
  - MIND clearly beats simple output confidence baselines
  - MIND does not remove the gap to richer baselines like a linear probe

## What Is Live Right Now

Use only these results as current paper evidence until the remaining reruns finish.

### POPE Popular, `image_grouped`, current code path

- Qwen:
  - raw only: `ROC-AUC 0.8462`, `PR-AUC 0.1159`
  - raw + simple stats: `ROC-AUC 0.8908`, `PR-AUC 0.1741`
  - raw + full curve: `ROC-AUC 0.9145`, `PR-AUC 0.2596`
  - raw + Haar: `ROC-AUC 0.8690`, `PR-AUC 0.1470`
  - logit margin: `ROC-AUC 0.5955`, `PR-AUC 0.0422`
- InternVL:
  - raw only: `ROC-AUC 0.8764`, `PR-AUC 0.4284`
  - raw + simple stats: `ROC-AUC 0.8978`, `PR-AUC 0.5092`
  - raw + full curve: `ROC-AUC 0.9119`, `PR-AUC 0.5333`
  - raw + Haar: `ROC-AUC 0.8929`, `PR-AUC 0.4854`
  - logit margin: `ROC-AUC 0.5454`, `PR-AUC 0.0861`

### How To Read These Numbers

- The calibrated signal matters:
  - both model families improve over raw-only once calibrated information is added
- Haar is dead weight:
  - it is never the best current variant
- Full-curve compression is competitive:
  - it is strongest on both current popular reruns
  - but the win over simple stats is not clean enough to justify the larger feature set as the default
- Output confidence is not enough:
  - grouped logit-margin performance is weak on both current reruns

## Historical Artifact Warning

The older March correction-phase popular tables are not reproducible under the current evaluation path.

- The old correction artifacts remain on disk for audit.
- They should not be used in the paper.
- The live round-two reruns are now the source of truth.

## Section Plan

### 1. Introduction

- Keep the paper on object hallucination only.
- Ask one clean question:
  - does grounded-versus-hallucinated behavior leave a compact geometric trace before the answer begins?
- Keep the paper honest about scope:
  - pre-answer geometry
  - object existence questions
  - grouped evaluation

### 2. Related Work

- POPE, RePOPE, and DASH-B for object hallucination evaluation
- output-side confidence baselines
- pre-generation probe methods such as HALP
- similarity-based grounding methods such as GLSim
- manifold and residual geometry in hidden-state analysis

### 3. Method

#### 3.1 Grounded Reference Bank

- Build object-conditioned banks from external COCO reference images.
- Keep only model outputs with `parsed_answer == 1`.
- State the limitation plainly:
  - this is the model’s own correct-yes regime, not an external grounding oracle

#### 3.2 Local Geometry

- Normalize the hidden states.
- Fit local PCA on the nearest reference neighbors.
- Measure the normalized normal residual.

#### 3.3 Cross-layer Signal

- Use selected pre-answer layers.
- Form the raw drift curve.
- Calibrate the curve with cleaned-bank mean and standard deviation.

#### 3.4 Final Feature Set

- Default:
  - raw drift features
  - calibrated simple statistics:
    - mean
    - max
    - final value
    - slope
    - variance
- Ablations:
  - raw only
  - raw + full calibrated curve
  - raw + Haar

#### 3.5 Detectors

- main detector:
  - logistic regression
- baseline detectors:
  - `p_yes`
  - yes-minus-no logit margin
  - chosen-answer confidence
  - linear probe
  - HALP
  - GLSim

### 4. Experiments

- Main table:
  - `POPE popular`
  - `DASH-B`
- Supplementary:
  - `POPE adversarial`
  - `RePOPE`
- Model set:
  - Qwen3-VL-8B
  - InternVL3.5-8B
  - LLaVA-OneVision-7B
  - Molmo-7B-D-0924
- Main protocol:
  - `image_grouped`
- Secondary protocol:
  - `object_heldout`

### 5. Main Story To Test

- Does the simple calibrated feature set hold up on DASH-B?
- Do the added models split the same way on object transfer?
- Does HALP beat MIND as a detector?
- Does GLSim beat both?

### 6. Discussion

- If HALP or GLSim win, the paper becomes a geometry paper, not a best-detector paper.
- If DASH-B hurts MIND much more than the baselines, say so plainly.
- If transfer behavior splits by architecture family, center that result.
- If all four models behave the same way, center the generality result instead.

## Tables To Keep

1. Main results:
   - models × `{POPE popular, DASH-B}`
   - methods:
     - `p_yes`
     - logit margin
     - chosen confidence
     - raw-only
     - no-manifold
     - full MIND with simple stats
     - linear probe
     - HALP
     - GLSim
2. Feature ablation:
   - raw only
   - raw + simple stats
   - raw + full curve
   - raw + Haar
3. Transfer and control:
   - object bank
   - shared bank
   - shuffled-object bank
   - linear probe
   - HALP
   - GLSim

## Writing Direction

- Do not write the paper around the old March package.
- Do not write Haar as the scientific center.
- Do write the paper around:
  - object hallucination only
  - grouped evaluation only
  - compact geometry as the main idea
  - honest comparison to simpler baselines and stronger probe methods
