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

The current round-two tables support a narrower and cleaner claim than the old drafts did.

- MIND clearly beats simple output confidence baselines on almost every completed row.
- The one visible exception is LLaVA on `POPE popular`, where chosen-answer confidence slightly beats full MIND.
- The linear probe beats full MIND on every completed row.
- So the paper can still make a modest detector-performance claim over simple output baselines.
- It cannot claim that MIND is the strongest internal-state detector.

## Current Round-Two Results

Use only the saved round-two rows in:

- `docs/tables/round2/table1_pope_popular.md`
- `docs/tables/round2/table1_dash_b.md`
- `docs/tables/round2/table2_feature_ablation.md`

### POPE Popular

The clean story on `POPE popular` is:

- full MIND beats the simple output baselines on Qwen, InternVL, and Molmo
- LLaVA is the one row where chosen-answer confidence edges out full MIND
- the linear probe beats full MIND on all four models

The current full-MIND rows are:

- Qwen: `ROC-AUC 0.8908`, `PR-AUC 0.1741`
- InternVL: `ROC-AUC 0.8978`, `PR-AUC 0.5092`
- LLaVA: `ROC-AUC 0.8085`, `PR-AUC 0.0874`
- Molmo: `ROC-AUC 0.8839`, `PR-AUC 0.2992`

The current linear-probe rows are:

- Qwen: `ROC-AUC 0.9161`, `PR-AUC 0.3803`
- InternVL: `ROC-AUC 0.9366`, `PR-AUC 0.6550`
- LLaVA: `ROC-AUC 0.8833`, `PR-AUC 0.3238`
- Molmo: `ROC-AUC 0.9209`, `PR-AUC 0.5606`

### DASH-B

`DASH-B` keeps the basic geometry signal alive, but it changes which part of the method is pulling the weight.

- full MIND still beats the simple output baselines on all four completed rows
- the linear probe beats full MIND by a very large margin on all four rows
- `no_manifold` beats full MIND on all four rows

The current full-MIND rows are:

- Qwen: `ROC-AUC 0.9193`, `PR-AUC 0.7374`
- InternVL: `ROC-AUC 0.8574`, `PR-AUC 0.7084`
- LLaVA: `ROC-AUC 0.8404`, `PR-AUC 0.7234`
- Molmo: `ROC-AUC 0.7795`, `PR-AUC 0.5422`

The current `no_manifold` rows are:

- Qwen: `ROC-AUC 0.9290`, `PR-AUC 0.7784`
- InternVL: `ROC-AUC 0.8769`, `PR-AUC 0.7288`
- LLaVA: `ROC-AUC 0.8996`, `PR-AUC 0.7883`
- Molmo: `ROC-AUC 0.8655`, `PR-AUC 0.6861`

The current linear-probe rows are:

- Qwen: `ROC-AUC 0.9909`, `PR-AUC 0.9779`
- InternVL: `ROC-AUC 0.9858`, `PR-AUC 0.9699`
- LLaVA: `ROC-AUC 0.9923`, `PR-AUC 0.9883`
- Molmo: `ROC-AUC 0.9775`, `PR-AUC 0.9561`

### How To Read These Results

- The compact drift signal is real:
  - it beats simple output confidence methods on almost every completed row
- The richer probe is stronger:
  - linear probing wins on both benchmarks for every model
- The manifold step is the weak point on `DASH-B`:
  - the harder benchmark does not kill the geometry signal
  - it mainly hurts the manifold-normalized version of that signal
- The model families do not line up perfectly:
  - InternVL and Qwen remain strong under full MIND
  - Molmo keeps a strong signal, but the gap to the linear probe is large on `DASH-B`
  - LLaVA is the only popular row where a simple confidence baseline edges out full MIND

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

- The simple calibrated feature set holds up better than output confidence, but not better than the linear probe.
- The manifold step is not robust on `DASH-B`.
- The remaining open question is whether HALP or GLSim beat MIND once the comparator work is finished.
- The transfer story is still open because the held-out control tables are not built yet.

### 6. Discussion

- The paper is already a geometry paper more than a detector paper.
- The current tables justify a claim over simple output baselines.
- They do not justify a claim over richer internal baselines.
- The strongest negative result so far is that `no_manifold` beats full MIND on `DASH-B`.
- If HALP or GLSim also beat MIND, the paper should lean even harder into interpretability and architecture comparison.

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
