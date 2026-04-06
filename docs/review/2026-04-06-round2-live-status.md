# Round-Two Live Status

Date: 2026-04-06

This note records the actual round-two run state after the next execution pass on the live round-two tree.

## What is complete

- LLaVA-OneVision `POPE popular` eval cache is complete: `24` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/pope/popular`.
- Molmo `POPE popular` eval cache is complete: `24` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/popular`.
- LLaVA-OneVision COCO reference cache is complete: `40` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/pope-reference/train`.
- Molmo COCO reference cache is complete: `40` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope-reference/train`.
- LLaVA-OneVision `DASH-B` eval cache is complete: `21` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/dash-b/main`.
- Molmo `DASH-B` eval cache is complete: `21` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/dash-b/main`.
- InternVL `DASH-B` eval cache is complete: `21` shards under `outputs/round2_2026_04/cache/internvl3.5-8b/dash-b/main`.
- Qwen `DASH-B` eval cache is complete: `21` shards under `outputs/round2_2026_04/cache/qwen3-vl-8b/dash-b/main`.
- Qwen `DASH-B` positive-reference cache is complete: `11` shards under `outputs/round2_2026_04/cache/qwen3-vl-8b/dash-b-reference/train`.
- InternVL, LLaVA-OneVision, and Molmo now all have built `DASH-B` reference-bank trees under `outputs/round2_2026_04/reference_banks_dash_b/`.
- Qwen `POPE popular` now has a materially complete report in `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/`.
  - all output baselines are written
  - `full.csv`, `drift_only.csv`, `no_manifold.csv`, and `linear_probe.csv` are written
  - `baselines.json`, `ablations.csv`, and `split_sensitivity.csv` are present
- Qwen `POPE popular` current linear-probe metric is now visible in the live report:
  - `ROC-AUC 0.9161`
  - `PR-AUC 0.3803`
- The frozen feature decision remains unchanged: full MIND uses `raw + calibrated simple stats`.

## What is still running

- InternVL `POPE popular` linear probe is still live. The rest of the report is already written.
- LLaVA-OneVision `POPE popular` linear probe is still live. The rest of the report is already written.
- Molmo `POPE popular` linear probe is still live. The rest of the report is already written.
- Qwen `DASH-B` object-bank build is still live under `outputs/round2_2026_04/reference_banks_dash_b/qwen3-vl-8b/`.
- Molmo `DASH-B` object-bank drift feature generation is still live for `round2-molmo-7b-d-0924-dash-b`.

## What is incomplete or blocked

- InternVL, LLaVA-OneVision, and Molmo `POPE popular` reports are still waiting on `linear_probe.csv`. They are not final paper inputs yet.
- There are still no round-two `readouts/` outputs, so HALP and GLSim have not started.
- There are still no tracked round-two main tables beyond the frozen phase-one decision note.
- `DASH-B` feature parquets and report suites are still missing for all four models.
- `POPE adversarial`, `RePOPE`, transfer controls, bank-size ablation, layer-count ablation, HALP, and GLSim are still pending.

## What this means

- The project is still in execution, but the old Qwen `DASH-B` network blocker is gone.
- The live bottleneck is now compute completion, not model access.
- The next paper-facing table can be generated as soon as the remaining three `POPE popular` linear probes finish.

## Queue Notes

- The live queue needed cleanup during this pass:
  - duplicate `molmo DASH-B compute_drift` workers were trimmed back to one writer
  - duplicate `qwen DASH-B` positive-reference cache workers were trimmed back to one writer
- The masked-GPU launch pattern needs care:
  - under `CUDA_VISIBLE_DEVICES=1`, the job should use `--device cuda`, not `--device cuda:1`
