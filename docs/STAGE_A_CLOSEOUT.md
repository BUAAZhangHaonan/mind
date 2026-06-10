# Stage A Closeout

Stage A closeout exists to answer one narrow question: when the sequence model is held fixed, is the hyperspherical full-layer trajectory beneficial, neutral, or harmful relative to the raw full-layer trajectory?

It does not validate the final MIND detector. It does not start Stage B. It does not tune support estimators, radius balls, contrastive objectives, conformal prediction, or final detector heads.

## Input

Stage A closeout reads:

```text
outputs/full_cache/manifests/unified_full_cache_manifest.json
```

The manifest is the only source of cache roots. Main-env, Stage0-accepted, and separate-env cache roots are all supported through the manifest fields.

## Splits

Closeout writes new grouped split manifests under:

```text
outputs/stageA_closeout/manifests/
```

The split key is `image_id`. POPE and RePOPE each pool popular, random, and adversarial into one family split. DASH-B uses the `all` subset. The split ratios are `0.50, 0.20, 0.10, 0.20` for encoder train, bank, calibration, and test.

## Variants

The closeout table has exactly six variants:

- `Raw-Static`
- `Sphere-Static`
- `Raw-Traj-MeanPool`
- `Sphere-Traj-MeanPool`
- `Raw-Traj-LSTM`
- `Sphere-Traj-LSTM`

Only `Raw-Traj-LSTM` is newly added. Norm-only and shuffled controls are legacy appendix material and are not expanded across the full panel in this closeout.

## Readouts

The primary readout is `Diag-Classifier`. It measures separable signal in the representation. The secondary readout is `Diag-KNN`. It measures correct-bank geometry and remains a diagnostic readout, not a final one-class method.

## Dataset Roles

RePOPE pooled test is the primary closeout dataset because its labels are cleaner for this representation pretest. POPE pooled test is a compatibility readout. DASH-B pooled test is a transfer descriptor only.

## Outputs

Closeout writes:

```text
outputs/stageA_closeout/audit/cache_label_balance.csv
outputs/stageA_closeout/audit/closeout_population_audit.csv
outputs/stageA_closeout/manifests/pope_family_split_manifest.json
outputs/stageA_closeout/manifests/repope_family_split_manifest.json
outputs/stageA_closeout/manifests/dash_b_split_manifest.json
outputs/stageA_closeout/reports/closeout_metrics_long.csv
outputs/stageA_closeout/reports/repope_main_table_classifier.csv
outputs/stageA_closeout/reports/repope_main_table_knn.csv
outputs/stageA_closeout/reports/pope_secondary_table.csv
outputs/stageA_closeout/reports/dash_b_secondary_table.csv
outputs/stageA_closeout/reports/per_model_summary.csv
outputs/stageA_closeout/reports/STAGE_A_CLOSEOUT_SUMMARY.md
outputs/stageA_closeout/reports/STAGE_A_CLOSEOUT_SUMMARY.json
```

The `outputs/` tree is ignored by default, so tracked docs record the canonical artifact paths.

## Closure Rule

Stage A is closed after Raw-Traj-LSTM is added and the closeout summary is written. Later stages must not reopen Stage A except if the frozen theory note is explicitly revised.

## Current Closeout Result

The canonical closeout summary is:

```text
outputs/stageA_closeout/reports/STAGE_A_CLOSEOUT_SUMMARY.md
outputs/stageA_closeout/reports/STAGE_A_CLOSEOUT_SUMMARY.json
```

The merged full-panel closeout contains 16 panel models and 576 metric rows. The RePOPE primary closeout verdict is `beneficial` under the frozen closeout semantics.

One model is explicitly recorded as failed for Stage A closeout metrics:

```text
glm-4.6v-flash: no primary closeout population on RePOPE, POPE, or DASH-B because parsed_answer is None across the checked cache population.
```

This is a Stage A population issue, not a claim about the model asset being unusable. It does not change the Experiment 2 full-cache status.
