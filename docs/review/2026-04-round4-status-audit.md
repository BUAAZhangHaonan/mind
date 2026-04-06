# Round-Four Status Audit

Date: 2026-04-06

This note records the live round-two state after the latest filesystem and process audit. It is based on the current round-two tree under `outputs/round2_2026_04/` plus the still-running process list.

## Pipeline Matrix

| Model | Benchmark | Eval cache | Ref cache | Reference bank | Drift features | Report artifacts | Readouts | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | POPE popular | complete | complete | complete | complete | complete | missing | complete for Table 1 |
| InternVL3.5-8B | POPE popular | complete | complete | complete | complete | complete | missing | complete for Table 1 |
| LLaVA-OneVision-7B | POPE popular | complete | complete | complete | complete | complete | missing | complete for Table 1 |
| Molmo-7B-D-0924 | POPE popular | complete | complete | complete | complete | complete | missing | complete for Table 1 |
| Qwen3-VL-8B | DASH-B | complete (`21` shards) | complete (`11` shards) | complete (`70` objects, `1120` count rows) | complete | complete | pending | main report complete, comparator work pending |
| InternVL3.5-8B | DASH-B | complete (`21` shards) | complete (`11` shards) | complete (`70` objects, `1120` count rows) | complete | partial | pending | missing `no_manifold` and `linear_probe` report variants |
| LLaVA-OneVision-7B | DASH-B | complete (`21` shards) | complete (`11` shards) | complete (`70` objects, `980` count rows) | missing | missing | started (`3` shards) | needs drift, baselines, and comparator runs |
| Molmo-7B-D-0924 | DASH-B | complete (`21` shards) | complete (`11` shards) | complete (`70` objects, `980` count rows) | missing | missing | started (`5` shards) | needs drift, baselines, and comparator runs |

## POPE Popular Report Check

All four `POPE popular` report trees now contain the required seven variant CSVs plus the three summary files:

- `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/`
- `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular/`
- `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular/`
- `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular/`

The first main paper table can therefore be treated as settled from round-two-local artifacts.

## DASH-B Report Check

Current `DASH-B` report state:

- Qwen:
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b/`
  - complete for the seven paper variants plus `baselines.json`, `ablations.csv`, and `split_sensitivity.csv`
- InternVL:
  - `outputs/round2_2026_04/reports/round2-internvl3.5-8b-dash-b/`
  - has `full`, `drift_only`, and all three output baselines
  - still missing `no_manifold.csv` and `linear_probe.csv`
- LLaVA-OneVision:
  - no report tree yet
- Molmo:
  - no report tree yet

## Adversarial And RePOPE

| Model | POPE adversarial cache | RePOPE eval | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | present under `outputs/cache/qwen3-vl-8b/pope/adversarial/` | missing | extraction can be reused, evaluation pending |
| InternVL3.5-8B | present under `outputs/cache/internvl3.5-8b/pope/adversarial/` | missing | extraction can be reused, evaluation pending |
| LLaVA-OneVision-7B | complete (`24` shards) | missing | fresh cache ready, evaluation pending |
| Molmo-7B-D-0924 | complete (`24` shards) | missing | fresh cache ready, evaluation pending |

Notes:

- `RePOPE` has not started for any model.
- The mixed cache layout remains a real execution hazard:
  - Qwen and InternVL POPE caches live under `outputs/cache/...`
  - new-model and DASH-B caches live under `outputs/round2_2026_04/cache/...`

## Comparator State

`outputs/round2_2026_04/readouts/` now exists.

- `outputs/round2_2026_04/readouts/llava-onevision-7b/dash-b/main/` has `3` shards
- `outputs/round2_2026_04/readouts/molmo-7b-d-0924/dash-b/main/` has `5` shards
- no Qwen or InternVL readout shards exist yet
- no `POPE popular` readout shards exist yet
- no HALP or GLSim result dirs exist yet

## Live Process State

Active processes at audit time:

- Qwen `DASH-B` `linear_probe` report pass:
  - `scripts/compute_baselines.py`
  - writing to `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b/`
- LLaVA `DASH-B` readout extraction:
  - `scripts/extract_readout_states.py`
  - writing to `outputs/round2_2026_04/readouts/llava-onevision-7b/dash-b/main/`
- Molmo `DASH-B` readout extraction:
  - `scripts/extract_readout_states.py`
  - writing to `outputs/round2_2026_04/readouts/molmo-7b-d-0924/dash-b/main/`

No duplicate writer was observed on the same output root at audit time.

## Design Problems Exposed By The Audit

- The pipeline is fine, but execution bookkeeping is still fragile. A report can look complete while a long `linear_probe` pass is still running in the background.
- The cache layout is still inconsistent across model families. That is the easiest way to accidentally mix old POPE paths with round-two paths.
- The comparator workflow still has no durable checkpoint between readout extraction and final scores beyond raw shard files. That is workable, but it makes restart planning more manual than it should be.
