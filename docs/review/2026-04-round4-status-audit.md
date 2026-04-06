# Round-Four Status Audit

Date: 2026-04-06

This note records the live round-two state from the filesystem and process list after the latest audit pass. It replaces older live notes where the job state had already moved on.

## Main Table Readiness

### POPE popular

The first main table is ready. All four `POPE popular` report trees now contain the full seven-method report set plus the summary artifacts.

| Model | Report dir | Report status |
| --- | --- | --- |
| Qwen3-VL-8B | `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/` | complete |
| InternVL3.5-8B | `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular/` | complete |
| LLaVA-OneVision-7B | `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular/` | complete |
| Molmo-7B-D-0924 | `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular/` | complete |

Required files confirmed in every popular report dir:

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

### DASH-B

The second main table is only partially ready.

| Model | Eval cache | Reference bank | Drift features | Report status | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | complete (`21` shards) | complete (`70` objects, `reference_counts.csv`, `1120` rows) | complete (`main.parquet`) | complete | main image-grouped report is ready |
| InternVL3.5-8B | complete (`21` shards) | complete (`70` objects, `reference_counts.csv`, `1120` rows) | complete (`main.parquet`) | partial | `no_manifold.csv` and `linear_probe.csv` still missing |
| LLaVA-OneVision-7B | complete (`21` shards) | complete (`70` objects, `reference_counts.csv`, `980` rows) | missing | not started | banks are ready but no `main.parquet` yet |
| Molmo-7B-D-0924 | complete (`21` shards) | complete (`70` objects, `reference_counts.csv`, `980` rows) | missing | not started | banks are ready but no `main.parquet` yet |

Current `DASH-B` report dirs:

- `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b/` is complete for the seven main variants.
- `outputs/round2_2026_04/reports/round2-internvl3.5-8b-dash-b/` currently has:
  - `full.csv`
  - `drift_only.csv`
  - `output_p_yes.csv`
  - `output_logit_margin.csv`
  - `output_chosen_answer_confidence.csv`
  - `baselines.json`
  - `ablations.csv`
  - `split_sensitivity.csv`
- `outputs/round2_2026_04/reports/round2-llava-onevision-7b-dash-b/` does not exist yet.
- `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-dash-b/` does not exist yet.

## Supplementary Benchmark State

### POPE adversarial

| Model | Cache root | Status | Notes |
| --- | --- | --- | --- |
| Qwen3-VL-8B | `outputs/cache/qwen3-vl-8b/pope/adversarial/` | present | legacy cache root, `94` shards |
| InternVL3.5-8B | `outputs/cache/internvl3.5-8b/pope/adversarial/` | present | legacy cache root, `24` shards |
| LLaVA-OneVision-7B | `outputs/round2_2026_04/cache/llava-onevision-7b/pope/adversarial/` | complete | `24` shards |
| Molmo-7B-D-0924 | `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/adversarial/` | complete | `24` shards |

No adversarial report dirs have been generated yet in the round-two report tree.

### RePOPE

No round-two RePOPE report dirs are present yet for any model.

## Comparator State

`outputs/round2_2026_04/readouts/` now exists, but comparator execution is still at the first extraction step.

| Model | Benchmark | Readout root | Shards written | Status |
| --- | --- | --- | --- | --- |
| LLaVA-OneVision-7B | DASH-B | `outputs/round2_2026_04/readouts/llava-onevision-7b/dash-b/main/` | `2` | running |
| Molmo-7B-D-0924 | DASH-B | `outputs/round2_2026_04/readouts/molmo-7b-d-0924/dash-b/main/` | `5` | running |
| Qwen3-VL-8B | DASH-B | missing | `0` | not started |
| InternVL3.5-8B | DASH-B | missing | `0` | not started |
| all models | POPE popular | missing | `0` | not started |

No HALP output dirs are present yet.

No GLSim output dirs are present yet.

## Live Process State

The process list still shows three long jobs active:

- `scripts/compute_baselines.py` for `round2-qwen3-vl-8b-dash-b` with `--variants linear_probe`
- `scripts/extract_readout_states.py` for LLaVA `DASH-B`
- `scripts/extract_readout_states.py` for Molmo `DASH-B`

The GPUs report `0%` utilization while both readout extractors still hold memory. That means the machines are not truly idle. The jobs are still alive, but they are currently in a CPU-heavy or I/O-heavy phase rather than active kernel time.

## Design Problems Still Visible

- The execution layer still makes duplicate or stale writers too easy. The main method code is stable, but the process discipline around long jobs is still weak.
- The cache layout is still mixed:
  - older Qwen and InternVL POPE caches live under `outputs/cache/...`
  - newer round-two caches live under `outputs/round2_2026_04/cache/...`
  - this is workable, but easy to misuse when launching follow-up jobs
- Bank persistence and drift computation are now aligned after the low-support fix, but `object_heldout` still has to be resolved from actual feature availability instead of assuming every object-layer pair can support a manifold.

## Immediate Execution Priority

1. Finish the still-running `Qwen DASH-B linear_probe`.
2. Finish the missing InternVL `DASH-B` detector-side rows: `no_manifold` then `linear_probe`.
3. Compute `DASH-B` features for LLaVA and Molmo, then build their first report dirs.
4. Let the current LLaVA and Molmo `DASH-B` readout extractions finish, then launch Qwen and InternVL `DASH-B` readouts.
5. Only after the main `DASH-B` table is complete should CPU time move to adversarial, RePOPE, and the control tables.
