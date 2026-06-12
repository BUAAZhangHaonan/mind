# Stage B

Stage B compares metric-aligned objectives on the frozen hyperspherical trajectory representation. It does not choose the final detector.

## Frozen Assumptions

- The input object is the pre-generation full-layer trajectory from the unified full-cache manifest.
- The representation space is hyperspherical.
- The encoder family is fixed to `Sphere-Traj-LSTM`.
- RePOPE is the primary development dataset.
- POPE is a secondary compatibility readout.
- DASH-B is a secondary transfer descriptor.
- Stage C has not started in Stage B.

## Stage B1 Objective Set

Stage B1 compares exactly three objective families:

- `bce`: the baseline objective.
- `supcon`: supervised contrastive learning.
- `proxy_anchor`: proxy-based metric learning.

Raw trajectories, mean-pool controls, shuffled controls, angular-margin losses, and support-estimator optimization are outside Stage B1. They are not part of the Stage B1 main table.

## Diagnostics

The primary geometry diagnostic is auto-tuned geodesic kNN. The candidate set is `{1, 2, 4, 8, 16, 32, 64}`, clipped by the correct-bank size and `floor(sqrt(num_bank_correct))`. The selected `k` is chosen on RePOPE calibration rows by PR-AUC, then ROC-AUC, then smaller `k`.

The secondary geometry diagnostic is a single-vMF prototype score. It records a mean direction and concentration proxy from the correct bank. This is a diagnostic, not the final detector.

The classifier readout is only a control. It checks that the learned embedding has not collapsed into a geometry-only artifact that loses class signal.

## GLM Answer QC

`glm-4.6v-flash` keeps its full-cache panel status, but Stage B1 first runs a small answer-text QC. If GLM answer text is not parseable into yes/no under the frozen population rules, GLM can be excluded from Stage B1 metric tables with an explicit reason. This does not reopen Stage A and does not mark the full cache as failed.

## Canonical Commands

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b_glm_answer_qc.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB

conda run --no-capture-output -n mind-py311 python scripts/stage_b_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B1 writes reports under `outputs/stageB/`. Those reports identify objective-family behavior only. They do not validate the final MIND detector.

## Stage B2 Negative-Budget Efficiency

Stage B2 freezes the Stage B1 winner, `proxy_anchor`, and varies only the hard-hallucination negative budget. It does not compare new objective families.

The required ratios are:

- `1.00`
- `0.50`
- `0.25`
- `0.10`

The fixed seeds are `20260506`, `20260507`, and `20260508`. Correct samples from `encoder_train` are not subsampled. Hard hallucination samples are subsampled without replacement. If a model-ratio pair would leave fewer than 20 hard negatives, that pair is skipped and reported.

Stage B2 uses the same frozen object, space, and encoder:

- pre-generation full-layer trajectory;
- layerwise hyperspherical normalization;
- `Sphere-Traj-LSTM`;
- `Proxy Anchor` objective only.

The primary diagnostic remains auto-tuned geodesic kNN on RePOPE pooled/test. The selected `k` is chosen on RePOPE calibration rows and then frozen for test evaluation. The classifier readout is a secondary control, not the decision signal. The single-vMF prototype remains a tertiary diagnostic. Stage B2 does not choose the final detector and does not start Stage C.

Canonical command:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b2_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB2 \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B2 writes reports under `outputs/stageB2/`. Those reports measure negative-budget efficiency for the frozen Proxy Anchor trajectory representation only.

## Stage B3 Geodesic kNN Scale Robustness

Stage B3 freezes the Stage B1/B2 decisions:

- objective: `proxy_anchor`;
- encoder: `Sphere-Traj-LSTM`;
- hard-negative budget: `0.50`;
- seeds: `20260506`, `20260507`, `20260508`.

It varies only the geodesic kNN neighborhood scale. The candidate set remains `{1, 2, 4, 8, 16, 32, 64}`, clipped by the correct-bank size and `floor(sqrt(num_bank_correct))`. For each model and seed, `k*` is selected on RePOPE calibration rows. Stage B3 then evaluates every valid `k` on RePOPE, POPE, and DASH-B test rows.

The stability band is defined on RePOPE pooled/test. It contains the maximal contiguous set of `k` values around the selected `k*` whose PR-AUC is within `0.02` of the PR-AUC at `k*`. Per-model verdicts are `scale_stable`, `scale_sensitive`, or `insufficient_coverage`. The panel verdict is one of `scale_stable_panel`, `scale_mixed_panel`, or `scale_sensitive_panel`.

Classifier control remains a lightweight logistic readout. It is only a sanity check that the embedding has not collapsed. The single-vMF probe remains tertiary. Stage B3 does not choose the final detector and does not start Stage C.

Canonical command:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b3_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB3 \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B3 writes reports under `outputs/stageB3/`. Those reports test readout scale robustness for the frozen Proxy Anchor representation only.

## Stage B4 Parametric Hyperspherical Support Diagnostics

Stage B4 freezes the Stage B1-B3 decisions:

- objective: `proxy_anchor`;
- encoder: `Sphere-Traj-LSTM`;
- hard-negative budget: `0.50`;
- seeds: `20260506`, `20260507`, `20260508`.

It compares support-family diagnostics on the frozen embedding. The nonparametric reference is selected geodesic kNN. The parametric family is vMF support:

- single-vMF, fitted as one mean direction and concentration proxy on the correct bank;
- mixture-vMF, fitted with directional initialization and EM-style updates on the correct bank;
- candidate mixture component counts `{1, 2, 4, 8}`, with `K=1` equivalent to single-vMF.

Mixture `K` is selected only on RePOPE calibration rows by PR-AUC, then ROC-AUC, then smaller `K`. The selected vMF family is then frozen for RePOPE, POPE, and DASH-B test rows. kNN remains the nonparametric reference and is not renamed as a one-class method.

The vMF stability band is defined on RePOPE pooled/test. It contains the maximal contiguous set of `K` values around the selected `K*` whose PR-AUC is within `0.02` of the PR-AUC at `K*`. The classifier readout remains a lightweight logistic control only. It does not decide the support-family verdict.

Stage B4 writes:

- `outputs/stageB4/reports/repope_support_family_knn.csv`;
- `outputs/stageB4/reports/repope_support_family_single_vmf.csv`;
- `outputs/stageB4/reports/repope_support_family_mixture_vmf.csv`;
- `outputs/stageB4/reports/vmf_selected_k.csv`;
- `outputs/stageB4/reports/vmf_stability_band.csv`;
- `outputs/stageB4/reports/per_model_support_family_summary.csv`;
- `outputs/stageB4/reports/STAGE_B4_SUMMARY.md`.

Canonical command:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b4_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB4 \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B4 compares support-family diagnostics only. It does not choose the final detector. Stage C starts after B4 and keeps the B4 representation decisions frozen.
