# Stage D

Stage D evaluates the frozen MIND method. It does not redesign the method, change the embedding, or start a later stage.

## Frozen Methods

Stage D keeps the Stage A, Stage B, and Stage C decisions fixed:

- `MIND-main`: `Sphere-Traj-LSTM + Proxy Anchor + radius_ball`
- `MIND-param`: `Sphere-Traj-LSTM + Proxy Anchor + single-vMF`
- `logistic(z)`: lightweight supervised comparator on the same frozen embedding

The object is still the pre-generation full-layer trajectory. The space is still hyperspherical. The hard-negative budget is fixed at `0.50`, with seeds `20260506`, `20260507`, and `20260508`.

## Three Tasks

Stage D has exactly three evaluation tasks:

- Cross-domain dataset generalization across fixed source-target protocols.
- Domain-expansion comparison under the same practical constraints.
- Model-family analysis across the final full-cache panel.

It does not introduce new objective families, detector families, model panels, or population rules.

## Cross-Domain Protocols

The primary protocols are:

- `repope_to_repope`
- `repope_to_pope`
- `repope_to_dashb`
- `pope_to_dashb`

`pope_to_repope` may be run only as an optional secondary protocol. Each protocol trains and calibrates on the source family and evaluates on the target family. Oracle target-calibration rows are diagnostic only and are not deployable rows.

## Domain-Expansion Tiers

Tier A is the primary fair-comparison table. It contains only methods that use the same practical constraints:

- `MIND-main`
- `MIND-param`
- `logistic(z)`
- `final-hidden linear probe`
- `output-confidence`
- `HALP-lite`

Tier B is a ceiling or broader-access table. It may contain `official HALP` or the closest faithful reproduction when feasible. Tier B is not part of the primary fair-comparison claim.

## Model-Family Analysis

Stage D groups the panel into:

- `qwen`
- `internvl`
- `llava`
- `gemma`
- `phi`
- `minicpm`
- `glm`
- `molmo`

`glm-4.6v-flash` remains in the panel summary. It is excluded from population-based quantitative metrics while its answer format is incompatible with the frozen yes/no rule. Molmo remains eligible through its separate-env accepted cache.

## Outputs

Stage D writes under `outputs/stageD/`:

- `outputs/stageD/preflight/stageD_preflight.json`
- `outputs/stageD/manifests/repope_family_split_manifest.json`
- `outputs/stageD/manifests/pope_family_split_manifest.json`
- `outputs/stageD/manifests/dash_b_split_manifest.json`
- `outputs/stageD/reports/cross_domain_metrics_long.csv`
- `outputs/stageD/reports/cross_domain_primary_table.csv`
- `outputs/stageD/reports/cross_domain_oracle_recalibration_table.csv`
- `outputs/stageD/reports/domain_expansion_tierA.csv`
- `outputs/stageD/reports/domain_expansion_tierB.csv`
- `outputs/stageD/reports/related_method_feasibility.md`
- `outputs/stageD/reports/model_family_summary.csv`
- `outputs/stageD/reports/per_model_stageD_summary.csv`
- `outputs/stageD/reports/STAGE_D_SUMMARY.md`

## Canonical Command

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_d_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageD \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage D is evaluation only. It does not claim universal validation or final scientific success beyond the written Stage D evidence.
