# Round-Three Status Audit

Date: 2026-04-06

This note records the live round-two state at the start of finalization.

## Conclusion

Round two is now execution-heavy rather than extraction-heavy.

- `POPE popular` features and reference banks exist for all four models.
- The popular reports are almost complete, but each still needs `linear_probe.csv`.
- `DASH-B` is partly ready: three eval caches exist, one benchmark-specific bank tree exists, and the readout pipeline has not started.

## Popular Pipeline

### Features

These round-two-local feature parquets exist and each contains `3000` rows.

- `outputs/round2_2026_04/features/round2-qwen3-vl-8b-popular/popular.parquet`
- `outputs/round2_2026_04/features/round2-internvl3.5-8b-popular/popular.parquet`
- `outputs/round2_2026_04/features/round2-llava-onevision-7b-popular/popular.parquet`
- `outputs/round2_2026_04/features/round2-molmo-7b-d-0924-popular/popular.parquet`

### Reference banks

These object-conditioned `POPE popular` bank roots exist for all four models.

- `outputs/round2_2026_04/reference_banks/qwen3-vl-8b`
- `outputs/round2_2026_04/reference_banks/internvl3.5-8b`
- `outputs/round2_2026_04/reference_banks/llava-onevision-7b`
- `outputs/round2_2026_04/reference_banks/molmo-7b-d-0924`

Each root contains:

- `79` object directories
- `reference_counts.csv`

### Reports

#### Qwen3-VL-8B

- Report root used for the live rerun:
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final`
- Present:
  - `variant_results/output_p_yes.csv`
  - `variant_results/output_logit_margin.csv`
  - `variant_results/output_chosen_answer_confidence.csv`
  - `variant_results/drift_only.csv`
  - `variant_results/full.csv`
  - `variant_results/no_manifold.csv`
  - `baselines.json`
  - `ablations.csv`
  - `split_sensitivity.csv`
- Missing:
  - `variant_results/linear_probe.csv`

#### InternVL3.5-8B

- Report root:
  - `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular`
- Present:
  - `variant_results/output_p_yes.csv`
  - `variant_results/output_logit_margin.csv`
  - `variant_results/output_chosen_answer_confidence.csv`
  - `variant_results/drift_only.csv`
  - `variant_results/full.csv`
  - `variant_results/no_manifold.csv`
  - `baselines.json`
  - `ablations.csv`
  - `split_sensitivity.csv`
- Missing:
  - `variant_results/linear_probe.csv`

#### LLaVA-OneVision-7B

- Report root:
  - `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular`
- Present:
  - `variant_results/output_p_yes.csv`
  - `variant_results/output_logit_margin.csv`
  - `variant_results/output_chosen_answer_confidence.csv`
  - `variant_results/drift_only.csv`
  - `variant_results/full.csv`
  - `variant_results/no_manifold.csv`
  - `baselines.json`
  - `ablations.csv`
  - `split_sensitivity.csv`
- Missing:
  - `variant_results/linear_probe.csv`

#### Molmo-7B-D-0924

- Report root:
  - `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular`
- Present:
  - `variant_results/output_p_yes.csv`
  - `variant_results/output_logit_margin.csv`
  - `variant_results/output_chosen_answer_confidence.csv`
  - `variant_results/drift_only.csv`
  - `variant_results/full.csv`
  - `variant_results/no_manifold.csv`
  - `baselines.json`
  - `ablations.csv`
  - `split_sensitivity.csv`
- Missing:
  - `variant_results/linear_probe.csv`

## DASH-B State

### Eval caches

These eval cache roots are complete at `21` shards each.

- `outputs/round2_2026_04/cache/internvl3.5-8b/dash-b/main`
- `outputs/round2_2026_04/cache/llava-onevision-7b/dash-b/main`
- `outputs/round2_2026_04/cache/molmo-7b-d-0924/dash-b/main`

### Positive-reference caches

These positive-image cache roots exist at `11` shards each.

- `outputs/round2_2026_04/cache/internvl3.5-8b/dash-b-reference/train`
- `outputs/round2_2026_04/cache/llava-onevision-7b/dash-b-reference/train`
- `outputs/round2_2026_04/cache/molmo-7b-d-0924/dash-b-reference/train`

### Reference banks

- Present:
  - `outputs/round2_2026_04/reference_banks_dash_b/molmo-7b-d-0924`
- Missing:
  - `outputs/round2_2026_04/reference_banks_dash_b/qwen3-vl-8b`
  - `outputs/round2_2026_04/reference_banks_dash_b/internvl3.5-8b`
  - `outputs/round2_2026_04/reference_banks_dash_b/llava-onevision-7b`

## Comparator State

- `outputs/round2_2026_04/readouts` does not exist yet.
- HALP has not started on round-two artifacts.
- GLSim has not started on round-two artifacts.

## Paper And Export State

- `docs/paper_outline.md` has the right frozen framing but still needs final round-two numbers.
- `docs/results_summary.md` is stale and still describes earlier unfinished cache state.
- `scripts/export_paper_package.py` now prefers the most complete round-two report when duplicate names exist, but the paper package still depends on the missing report and comparator outputs above.

## Decision

The clean critical path is:

1. finish the four popular `linear_probe` runs
2. generate the `POPE popular` main table from those completed reports
3. finish `DASH-B`, then the comparator readouts and paper package
