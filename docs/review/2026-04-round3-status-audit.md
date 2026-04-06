# Round-Three Status Audit

Date: 2026-04-06

## Live State

- `POPE popular` feature parquets exist for all four target models under `outputs/round2_2026_04/features/`.
- `POPE popular` reference-bank roots exist for all four target models under `outputs/round2_2026_04/reference_banks/`.
- Each popular reference-bank root contains `79` object folders plus `reference_counts.csv`.
- Only one round-two popular report directory exists:
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular/`
- That Qwen popular report is partial:
  - present: `baselines.json`, `ablations.csv`, `split_sensitivity.csv`
  - present: `variant_results/output_p_yes.csv`, `output_logit_margin.csv`, `output_chosen_answer_confidence.csv`
  - missing: `variant_results/full.csv`, `drift_only.csv`, `no_manifold.csv`, `linear_probe.csv`
- A debug Qwen popular run exists at:
  - `outputs/round2_2026_04/reports/debug-round2-qwen3-vl-8b-popular/variant_results/full.csv`
  - it uses the lean result schema and contains `3000` rows, but it is not the canonical paper report path.
- No round-two popular report directory exists yet for:
  - InternVL3.5-8B
  - LLaVA-OneVision-7B
  - Molmo-7B-D-0924
- `DASH-B` eval caches exist for:
  - InternVL3.5-8B: `21` shards
  - LLaVA-OneVision-7B: `21` shards
  - Molmo-7B-D-0924: `21` shards
- `DASH-B` positive-reference cache directories exist for:
  - InternVL3.5-8B: `11` shard files
  - LLaVA-OneVision-7B: `11` shard files
  - Molmo-7B-D-0924: `11` shard files
- Only Molmo currently has a built `DASH-B` reference-bank tree under `outputs/round2_2026_04/reference_banks_dash_b/`.
- No round-two readout cache tree exists yet under `outputs/round2_2026_04/readouts/`.
- `docs/results_summary.md` is stale. It still describes new-model work as pending.
- `docs/paper_outline.md` has the correct frozen framing, but still needs final round-two numbers.

## Pipeline Notes

- The round-two pipeline does not persist a separate `manifolds/` tree.
- The durable checkpoints are:
  - cache shards
  - reference-bank layer files
  - feature parquets
  - report artifacts
- `compute_baselines.py` is the canonical evaluator for the paper-facing baseline suite.
- `train_detector.py` should be used only when a saved detector artifact is explicitly needed.

## Immediate Consequences

- `POPE popular` must be completed for all four models before any main paper table can be trusted.
- `DASH-B` is partially prepared but not paper-ready:
  - model eval caches exist for three models
  - reference banks and reports are still missing for most models
- HALP and GLSim are fully pending because round-two readout extraction has not started.
