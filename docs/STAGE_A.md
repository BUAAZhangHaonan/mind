# Stage A

Stage A is the representation-space acid test. It tests whether cached full-layer hidden states contain useful hallucination-detection signal. Stage A tests representation hypotheses only. Stage A does not validate the final MIND detector.

## What It Does

- Reads the completed Stage 0 cache from `outputs/stage0`.
- Writes all Stage A artifacts under `outputs/stageA`.
- Runs POPE popular, random, and adversarial for `qwen3-vl-8b`.
- Runs `internvl3.5-8b` only when requested and when Qwen does not fail.
- Builds a new POPE-family split by `image_id` across popular, random, and adversarial.
- Evaluates representation variants with `Diag-Classifier` and `Diag-KNN`.

## What It Does Not Do

- Stage A does not validate the final MIND detector.
- Stage A does not start Stage B.
- Stage A does not use DASH-B or RePOPE for Stage A metrics.
- Stage A does not implement contrastive learning, conformal prediction, final detection heads, bank-scope experiments, or later-stage radius-ball logic.
- Stage A does not import the retired drift, manifold, or wavelet logic.

## Preflight

`scripts/stage_a_preflight.py` checks Stage 0 before any experiment starts. It reads:

```text
outputs/stage0/manifests/stage0_summary.json
outputs/stage0/manifests/cache_manifest.json
outputs/stage0/audit/cache_label_balance.csv
```

It writes:

```text
outputs/stageA/audit/stage0_acceptance.json
```

The preflight requires Stage 0 to be passed, to record a git commit, to include both primary models, and to include complete POPE, RePOPE, and DASH-B cache matrix rows. RePOPE and DASH-B are Stage 0-complete checks only. They are not Stage A primary inputs.

## Population

Stage A uses the primary binary population:

- Correct sample: `parsed_answer` is not `None` and equals `label`.
- Hard hallucination: `label == 0` and `parsed_answer == 1`.
- Target `y = 0` for correct samples.
- Target `y = 1` for hard hallucination samples.

False-negative non-hallucination errors and parsed-none samples are excluded from primary metrics, but they are counted and reported.

## Split

`scripts/stage_a_build_family_splits.py` builds:

```text
outputs/stageA/manifests/pope_family_split_manifest.json
```

It combines POPE popular, random, and adversarial. It splits by `image_id` into `encoder_train`, `bank`, `cal`, and `test` with ratios `0.50, 0.20, 0.10, 0.20` and seed `20260506`. Stage 0 per-subset split conflicts are reported but not used.

## Representations

Stage A evaluates:

- `Raw-Static`
- `Sphere-Static`
- `Norm-Static`
- `Raw-Traj-MeanPool`
- `Sphere-Traj-MeanPool`
- `Norm-Traj`
- `Sphere-Traj-LSTM-v0`
- `Sphere-Traj-Shuffled-LSTM`

The shuffled LSTM uses one deterministic layer permutation per model. Stage A may conclude that only multi-layer aggregation is useful while layer order remains unproven.

## Readouts

`Diag-Classifier` tests whether the representation contains separable hallucination signal. Non-LSTM variants use logistic regression fit on `encoder_train`. LSTM variants use the supervised BCE classifier head.

`Diag-KNN` tests correct-bank geometry. It builds the bank from correct samples in the `bank` split and scores samples by mean distance to the `k` nearest bank samples.

## Metrics

Stage A reports PR-AUC, ROC-AUC, TPR at 1 percent FPR, FPR at 95 percent TPR, average precision, split counts, bank counts, and excluded-sample counts. PR-AUC and ROC-AUC include bootstrap confidence intervals.

## Gate Logic

The gate compares sphere against raw, trajectory against static, ordered LSTM against shuffled LSTM, and norm-only diagnostics against non-norm variants. Overall Stage A decisions are:

- `pass`
- `mixed_positive`
- `fail`

If layer order is mixed or failed, the conclusion says: multi-layer aggregation may be useful, but sequence order is not yet proven.

If norm diagnostics are strong, the conclusion says: magnitude carries useful signal and cannot be dismissed as pure noise.

## Outputs

Required outputs include:

```text
outputs/stageA/audit/stage0_acceptance.json
outputs/stageA/audit/cache_label_balance.csv
outputs/stageA/audit/stageA_population_audit.csv
outputs/stageA/manifests/pope_family_split_manifest.json
outputs/stageA/reports/qwen3-vl-8b/stageA_metrics.csv
outputs/stageA/reports/qwen3-vl-8b/stageA_gate.json
outputs/stageA/reports/qwen3-vl-8b/stageA_summary.md
outputs/stageA/manifests/stageA_summary.json
```

InternVL writes the same report files if it runs. If it does not run, Stage A writes `outputs/stageA/reports/internvl3.5-8b/not_run.json`.

## Commands

Dry run:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_a_run.py \
  --stage0-root outputs/stage0 \
  --output-root outputs/stageA \
  --models qwen3-vl-8b \
  --subsets popular random adversarial \
  --device cuda:0 \
  --dry-run
```

Qwen primary:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_a_run.py \
  --stage0-root outputs/stage0 \
  --output-root outputs/stageA \
  --models qwen3-vl-8b \
  --subsets popular random adversarial \
  --device cuda:0 \
  --bootstrap 1000 \
  --lstm-epochs 10 \
  --knn-k 10
```

Conditional two-model run:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_a_run.py \
  --stage0-root outputs/stage0 \
  --output-root outputs/stageA \
  --models qwen3-vl-8b internvl3.5-8b \
  --subsets popular random adversarial \
  --device cuda:0 \
  --bootstrap 1000 \
  --lstm-epochs 10 \
  --knn-k 10 \
  --include-internvl-after-qwen-pass
```

Stage B has not started.
