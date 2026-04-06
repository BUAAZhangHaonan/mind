# Round-Three Status Audit

Date: 2026-04-06

## Live State

- `POPE popular` feature parquets exist for all four paper models:
  - `round2-qwen3-vl-8b-popular`
  - `round2-internvl3.5-8b-popular`
  - `round2-llava-onevision-7b-popular`
  - `round2-molmo-7b-d-0924-popular`
- `POPE popular` reference banks exist for all four models under `outputs/round2_2026_04/reference_banks/`.
  - each root has `79` object folders plus `reference_counts.csv`
- The only live round-two popular report tree is `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular/`.
  - it currently contains `output_p_yes.csv`, `output_logit_margin.csv`, `output_chosen_answer_confidence.csv`
  - it also contains `baselines.json`, `ablations.csv`, and `split_sensitivity.csv`
  - it does not currently contain `full.csv`, `drift_only.csv`, `no_manifold.csv`, or `linear_probe.csv`
- A debug popular report exists for Qwen at `outputs/round2_2026_04/reports/debug-round2-qwen3-vl-8b-popular/variant_results/full.csv`.
  - the file uses the lean result schema
  - it is not a paper-ready replacement for the missing detector-side report set
- There are no live round-two popular report dirs yet for InternVL, LLaVA-OneVision, or Molmo.

## DASH-B State

- `DASH-B` eval cache shards are complete for:
  - InternVL3.5-8B: `21`
  - LLaVA-OneVision-7B: `21`
  - Molmo-7B-D-0924: `21`
- `DASH-B` positive-image cache shards exist for:
  - InternVL3.5-8B: `11`
  - LLaVA-OneVision-7B: `11`
  - Molmo-7B-D-0924: `11`
- The only built `DASH-B` reference-bank tree right now is:
  - `outputs/round2_2026_04/reference_banks_dash_b/molmo-7b-d-0924`
- There is still no round-two Qwen `DASH-B` cache tree under `outputs/round2_2026_04/cache/`.

## Comparator State

- There is still no `outputs/round2_2026_04/readouts/` tree.
- That means HALP and GLSim have not started on round-two artifacts.

## Paper-State Gaps

- `docs/results_summary.md` is stale.
  - it still describes new-model work as pending even though popular features and reference banks already exist
- `docs/paper_outline.md` has the right frozen framing
  - it still needs final round-two tables and final round-two numbers
- `scripts/export_paper_package.py` is already aimed at round-two report names
  - the live artifact tree does not yet satisfy what the exporter expects

## Pipeline Note

- This pipeline does not persist a separate round-two `manifolds/` tree.
- The durable checkpoints are:
  - cache shards
  - reference-bank layer files
  - feature parquets
  - report artifacts
