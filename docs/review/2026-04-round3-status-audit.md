# Round-Three Status Audit

Date: 2026-04-06

This note records the live round-two execution state before final table assembly.

## POPE popular

- Qwen popular is partially complete, but it is closer to paper-ready than the earlier live note suggested.
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/` already has:
    - `variant_results/full.csv`
    - `variant_results/drift_only.csv`
    - `variant_results/no_manifold.csv`
    - `variant_results/output_p_yes.csv`
    - `variant_results/output_logit_margin.csv`
    - `variant_results/output_chosen_answer_confidence.csv`
    - `baselines.json`
    - `ablations.csv`
    - `split_sensitivity.csv`
  - The only missing popular artifact is `variant_results/linear_probe.csv`.
- InternVL popular has the same status as Qwen:
  - six variant CSVs are present
  - `baselines.json`, `ablations.csv`, and `split_sensitivity.csv` are present
  - `linear_probe.csv` is still missing
- LLaVA-OneVision popular is in the same state:
  - six variant CSVs are present
  - `baselines.json`, `ablations.csv`, and `split_sensitivity.csv` are present
  - `linear_probe.csv` is still missing
- Molmo popular is also in the same state:
  - six variant CSVs are present
  - `baselines.json`, `ablations.csv`, and `split_sensitivity.csv` are present
  - `linear_probe.csv` is still missing

## Popular features and banks

- `POPE popular` feature parquets exist for all four paper models:
  - Qwen3-VL-8B
  - InternVL3.5-8B
  - LLaVA-OneVision-7B
  - Molmo-7B-D-0924
- `POPE popular` reference banks exist for all four models under `outputs/round2_2026_04/reference_banks/`.
- Each popular reference-bank tree has `79` object directories plus `reference_counts.csv`.

## DASH-B caches

- `DASH-B` eval caches are complete for:
  - InternVL3.5-8B: `21` shards
  - LLaVA-OneVision-7B: `21` shards
  - Molmo-7B-D-0924: `21` shards
- `DASH-B` positive-image cache shards are complete for the same three models:
  - InternVL3.5-8B: `11` shards
  - LLaVA-OneVision-7B: `11` shards
  - Molmo-7B-D-0924: `11` shards
- Qwen `DASH-B` eval cache has not been built yet under the round-two tree.

## DASH-B reference banks

- `outputs/round2_2026_04/reference_banks_dash_b/` exists.
- Molmo already has a complete `DASH-B` bank tree there:
  - `70` object directories
  - `reference_counts.csv` present
- InternVL and LLaVA do not yet have built `DASH-B` bank trees there.
- Qwen also has no `DASH-B` bank tree yet.

## Readout caches for HALP and GLSim

- `outputs/round2_2026_04/readouts/` does not exist yet.
- That means HALP and GLSim have not started on round-two artifacts.

## Model-path status

- A tracked local Qwen config already exists: `configs/models/qwen3_vl_8b_local.yaml`.
- The local model path `/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct` exists.
- The local Hugging Face cache for `Qwen3-VL-8B-Instruct` also exists under `~/.cache/huggingface/hub/`.
- Qwen `DASH-B` should therefore be treated as a local-path execution task, not a network-download problem.

## Documents

- `docs/paper_outline.md` has the right frozen round-two framing, but it still needs final round-two numbers.
- `docs/results_summary.md` is stale and still describes unfinished cache work that has already completed.

## Main design issue exposed by the audit

- The round-two report state is harder to reason about than it should be because the pipeline writes partial report trees and then fills them variant by variant.
- That is workable for long jobs, but it means report completeness has to be checked explicitly before any table or exporter step trusts a directory.
