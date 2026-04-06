# Round-Four Status Audit

Date: 2026-04-06

This note records the live round-two state from the current filesystem and process list. It replaces older queue notes that were written while jobs were still mid-flight.

## Bottom Line

- `POPE popular` is complete for the four-model main table.
- `DASH-B` is partially complete: Qwen is effectively done, InternVL is close, and LLaVA plus Molmo still need feature generation and report runs.
- Comparator work has started only for `DASH-B` readout extraction on LLaVA and Molmo. HALP and GLSim results do not exist yet.

## Status Matrix

| Model | Benchmark | Eval cache | Reference bank | Drift features | Main report | Readouts | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | `POPE popular` | existing original-model cache root | complete | complete | complete | missing | complete for Table 1 |
| InternVL3.5-8B | `POPE popular` | existing original-model cache root | complete | complete | complete | missing | complete for Table 1 |
| LLaVA-OneVision-7B | `POPE popular` | complete | complete | complete | complete | missing | complete for Table 1 |
| Molmo-7B-D-0924 | `POPE popular` | complete | complete | complete | complete | missing | complete for Table 1 |
| Qwen3-VL-8B | `DASH-B` | complete | complete | complete | complete | missing | ready for DASH-B table |
| InternVL3.5-8B | `DASH-B` | complete | complete | complete | partial | missing | missing `no_manifold` and `linear_probe` |
| LLaVA-OneVision-7B | `DASH-B` | complete | complete | missing | missing | partial | needs full CPU-side pipeline |
| Molmo-7B-D-0924 | `DASH-B` | complete | complete | missing | missing | partial | needs full CPU-side pipeline |

## POPE Popular

All four `POPE popular` report trees now contain the full seven-variant suite plus the summary files required for the paper:

- `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/`
- `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular/`
- `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular/`
- `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular/`

Each directory has:

- `variant_results/full.csv`
- `variant_results/drift_only.csv`
- `variant_results/no_manifold.csv`
- `variant_results/linear_probe.csv`
- `variant_results/output_p_yes.csv`
- `variant_results/output_logit_margin.csv`
- `variant_results/output_chosen_answer_confidence.csv`
- `baselines.json`
- `ablations.csv`
- `split_sensitivity.csv`

The tracked popular main table already exists:

- `docs/tables/round2/table1_pope_popular.md`
- `docs/tables/round2/table1_pope_popular.csv`

## DASH-B

All four models now have complete `DASH-B` eval caches and complete `DASH-B` reference-bank roots under `outputs/round2_2026_04/reference_banks_dash_b/`.

Reference-bank coverage:

- Qwen3-VL-8B: `70` objects, `1120` layer rows
- InternVL3.5-8B: `70` objects, `1120` layer rows
- LLaVA-OneVision-7B: `70` objects, `980` layer rows
- Molmo-7B-D-0924: `70` objects, `980` layer rows

Current feature/report state:

- Qwen3-VL-8B
  - `outputs/round2_2026_04/features/round2-qwen3-vl-8b-dash-b/main.parquet` exists
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b/` is complete, including `linear_probe.csv`, `baselines.json`, `ablations.csv`, and `split_sensitivity.csv`
- InternVL3.5-8B
  - `outputs/round2_2026_04/features/round2-internvl3.5-8b-dash-b/main.parquet` exists
  - `outputs/round2_2026_04/reports/round2-internvl3.5-8b-dash-b/` has:
    - `full.csv`
    - `drift_only.csv`
    - `output_p_yes.csv`
    - `output_logit_margin.csv`
    - `output_chosen_answer_confidence.csv`
    - `raw_curve_only.csv`
    - `raw_plus_calibrated_simple.csv`
    - `raw_plus_calibrated_full_curve.csv`
    - `raw_plus_calibrated_haar.csv`
    - `baselines.json`
    - `ablations.csv`
    - `split_sensitivity.csv`
  - missing:
    - `no_manifold.csv`
    - `linear_probe.csv`
- LLaVA-OneVision-7B
  - no `DASH-B` feature parquet yet
  - no `DASH-B` report tree yet
- Molmo-7B-D-0924
  - no `DASH-B` feature parquet yet
  - no `DASH-B` report tree yet

## POPE Adversarial And RePOPE

- LLaVA-OneVision adversarial eval cache is complete: `24` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/pope/adversarial/`
- Molmo adversarial eval cache is complete: `24` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/adversarial/`
- No adversarial report trees exist yet for any model.
- No RePOPE report trees exist yet for any model.

The original-model cache layout is still mixed:

- Qwen and InternVL popular and adversarial caches live under the older cache roots.
- LLaVA and Molmo caches live under `outputs/round2_2026_04/cache/`.

That split does not block round-two reporting, but it is easy to misuse and should be treated carefully in every launch command.

## Comparator State

Round-two readouts now exist, but only for `DASH-B` and only for two models so far:

- `outputs/round2_2026_04/readouts/llava-onevision-7b/dash-b/main/` currently has `3` shards
- `outputs/round2_2026_04/readouts/molmo-7b-d-0924/dash-b/main/` currently has `5` shards

Missing entirely:

- Qwen `DASH-B` readouts
- InternVL `DASH-B` readouts
- all `POPE popular` readouts
- HALP result files
- GLSim result files

## Live Process State

At the time of this audit, the long-running processes are:

- Qwen `DASH-B` `linear_probe` evaluation on CPU
- LLaVA `DASH-B` readout extraction on GPU 0
- Molmo `DASH-B` readout extraction on GPU 1

They are still alive in the process list. The GPUs show `0%` utilization in the short snapshot, but both readout jobs still hold memory and keep writing shards, so they have not died silently.

## Design Problems Exposed By This Audit

- The pipeline still mixes original-model cache roots with round-two cache roots. The reports are fine, but the path split is a constant source of operator error.
- Large readout shards are very heavy. They keep the GPU reserved while most of the wall time shifts to disk writes, which makes the machine look idle when it is not.
- Report directories fill in incrementally. That is practical for long jobs, but it means every table step must verify file completeness instead of assuming a directory is final.
