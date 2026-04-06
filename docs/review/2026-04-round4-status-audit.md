# Round-Four Status Audit

Date: 2026-04-06

This note captures the live round-two state after the next audit pass. It reflects the current filesystem and process list, not older status notes.

## Popular Reports

All four `POPE popular` report trees are now materially complete for the first main table.

| Model | Report dir | Variants present | Summary files | Status |
| --- | --- | --- | --- | --- |
| Qwen3-VL-8B | `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/` | `full`, `drift_only`, `no_manifold`, `linear_probe`, `output_p_yes`, `output_logit_margin`, `output_chosen_answer_confidence` | `baselines.json`, `ablations.csv`, `split_sensitivity.csv` | complete |
| InternVL3.5-8B | `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular/` | `full`, `drift_only`, `no_manifold`, `linear_probe`, `output_p_yes`, `output_logit_margin`, `output_chosen_answer_confidence` | `baselines.json`, `ablations.csv`, `split_sensitivity.csv` | complete |
| LLaVA-OneVision-7B | `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular/` | `full`, `drift_only`, `no_manifold`, `linear_probe`, `output_p_yes`, `output_logit_margin`, `output_chosen_answer_confidence` | `baselines.json`, `ablations.csv`, `split_sensitivity.csv` | complete |
| Molmo-7B-D-0924 | `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular/` | `full`, `drift_only`, `no_manifold`, `linear_probe`, `output_p_yes`, `output_logit_margin`, `output_chosen_answer_confidence` | `baselines.json`, `ablations.csv`, `split_sensitivity.csv` | complete |

## DASH-B State

| Model | Eval cache | Positive ref cache | Reference bank | Drift features | Status |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | `21` shards | `11` shards | `reference_counts.csv` with `70` objects, `1120` rows | missing | banks complete, features missing |
| InternVL3.5-8B | `21` shards | `11` shards | `reference_counts.csv` with `70` objects, `1120` rows | missing | banks complete, features missing |
| LLaVA-OneVision-7B | `21` shards | `11` shards | `reference_counts.csv` with `70` objects, `980` rows | missing | banks complete, features missing |
| Molmo-7B-D-0924 | `21` shards | `11` shards | `reference_counts.csv` with `70` objects, `980` rows | missing | bank complete, `compute_drift` running |

Notes:

- All four `DASH-B` bank roots exist under `outputs/round2_2026_04/reference_banks_dash_b/`.
- None of the four current `reference_counts.csv` files has `supports_manifold == True` for every row, so later `object_heldout` settings must still be resolved from actual feature frames.
- There are still no `DASH-B` feature parquets on disk.

## Adversarial and RePOPE

| Model | POPE adversarial cache | RePOPE eval | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | missing | missing | pending |
| InternVL3.5-8B | missing | missing | pending |
| LLaVA-OneVision-7B | launched on GPU 0 | missing | running |
| Molmo-7B-D-0924 | launched on GPU 1 | missing | running |

Notes:

- `outputs/round2_2026_04/normalized/pope/adversarial.jsonl` is present.
- The adversarial image root matches the popular caches: `data/coco/val2014`.

## Comparator State

| Stage | Status |
| --- | --- |
| `outputs/round2_2026_04/readouts/` | missing |
| HALP runs | not started |
| GLSim runs | not started |

## Live Process State

- Four old `POPE popular` `linear_probe` runner processes are still active on CPU.
  - Their output files are already present.
  - Their `baselines.json` files already contain `linear_probe` metrics.
  - Their `split_sensitivity.csv` files currently contain only the seed-13 `linear_probe` row.
- The current live GPU jobs are:
  - LLaVA `POPE adversarial` extraction on GPU 0
  - Molmo `POPE adversarial` extraction on GPU 1
- The current live CPU jobs are:
  - Molmo `DASH-B` `compute_drift`
  - the four old popular `linear_probe` runners

## Design Problems Exposed By The Audit

- The execution layer still makes it too easy to leave stale long-running processes alive after the output files they were meant to produce have already been written.
- The masked-GPU launch pattern is fragile. Under `CUDA_VISIBLE_DEVICES=1`, the correct device string is logical `cuda`, not physical `cuda:1`.
- Qwen and InternVL still use the older `outputs/cache/...` popular cache roots while the newer models use `outputs/round2_2026_04/cache/...`. The pipeline handles it, but the mixed cache layout is easy to misuse.
