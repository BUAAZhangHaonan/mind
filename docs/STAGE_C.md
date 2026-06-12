# Stage C

Stage C compares support estimator families on frozen hyperspherical Proxy Anchor embeddings. It does not change the embedding family and does not start Stage D.

## Frozen Assumptions

- The input object is the pre-generation full-layer trajectory from the unified full-cache manifest.
- The representation space is hyperspherical.
- The encoder family is fixed to `Sphere-Traj-LSTM`.
- The objective is fixed to `proxy_anchor`.
- The hard-negative budget is fixed to `0.50`.
- The fixed seeds are `20260506`, `20260507`, and `20260508`.
- RePOPE is the primary detector-selection dataset.
- POPE is a secondary compatibility readout.
- DASH-B is a descriptor readout.
- `glm-4.6v-flash` stays in the panel summary but is excluded from population-based metrics while its answer format is incompatible with the frozen yes/no rule.

## Unified Support-Estimation Frame

Stage C treats each support method as a support estimate on frozen unit-sphere embeddings:

```text
q(z | B) -> support level
s(z) = -log q(z | B) -> anomaly score
```

`B` is the correct-sample bank for the dataset family. Logistic regression is not part of this support family. It is only a supervised comparator.

## Methods

Stage C compares exactly five methods:

- `single_vmf`: one vMF support model fitted on the correct bank.
- `mixture_vmf`: a vMF mixture fitted on the correct bank.
- `knn`: density-style geodesic kNN support.
- `radius_ball`: fixed-radius geodesic local support.
- `logistic`: lightweight supervised comparator on frozen embeddings.

No other detector family is part of Stage C.

## Calibration

All hyperparameters are selected only on RePOPE calibration rows:

- `single_vmf`: no structural hyperparameter.
- `mixture_vmf`: `K in {2, 4, 8}`.
- `knn`: `k in {1, 2, 4, 8, 16, 32, 64}`, clipped by bank size and `floor(sqrt(num_bank_correct))`.
- `radius_ball`: candidate radii are quantiles `0.50, 0.65, 0.80, 0.90, 0.95` of RePOPE calibration correct-sample local support radii.
- `logistic`: `C in {0.1, 1, 10}`.

Selection uses PR-AUC first, ROC-AUC second, and the simpler or smaller hyperparameter last. Selected values are then frozen for RePOPE, POPE, and DASH-B test rows.

## Outputs

Stage C writes reports under `outputs/stageC/`:

- `outputs/stageC/preflight/stageC_preflight.json`
- `outputs/stageC/manifests/repope_family_split_manifest.json`
- `outputs/stageC/manifests/pope_family_split_manifest.json`
- `outputs/stageC/manifests/dash_b_split_manifest.json`
- `outputs/stageC/reports/stageC_metrics_long.csv`
- `outputs/stageC/reports/repope_main_table.csv`
- `outputs/stageC/reports/pope_secondary_table.csv`
- `outputs/stageC/reports/dash_b_secondary_table.csv`
- `outputs/stageC/reports/knn_selected_k.csv`
- `outputs/stageC/reports/radius_ball_selected_rho.csv`
- `outputs/stageC/reports/vmf_selected_k.csv`
- `outputs/stageC/reports/logistic_selected_c.csv`
- `outputs/stageC/reports/per_model_detector_summary.csv`
- `outputs/stageC/reports/STAGE_C_SUMMARY.md`

## Canonical Command

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_c_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageC \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage C selects a support estimator family on the frozen embedding. It does not validate MIND as a final detector and does not start Stage D.
