# Round-Four Status Audit

Date: 2026-04-07

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and job logs.

## Live Queue

There are no live MIND jobs now.

The GPUs are occupied, but not by MIND:

- GPU 0: `magformer` training via `python tools/train.py`
- GPU 1: `magformer` training via `python train_net_mgm_0831.py`

That means the MIND queue is empty. Any unfinished round-two work is not still running. It is either already complete on disk or still missing.

## Main Matrix State

| Model | POPE popular report | DASH-B reference bank | DASH-B features | DASH-B report | DASH-B readouts |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | complete | complete | complete | complete | complete (`42` shards, `shard-00041.pt`) |
| InternVL3.5-8B | complete | complete | complete | complete | partial (`36` shards, stops at `shard-00035.pt`) |
| LLaVA-OneVision-7B | complete | complete | complete | complete | partial (`7` shards, stops at `shard-00006.pt`) |
| Molmo-7B-D-0924 | complete | complete | complete | complete | complete after rerun (`42` shards in `main/`; older partial backup retained) |

### POPE popular

The first main table is already grounded in complete saved reports for all four models:

- `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/`
- `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular/`
- `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular/`
- `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular/`

The tracked paper table already exists:

- `docs/tables/round2/table1_pope_popular.md`
- `docs/tables/round2/table1_pope_popular.csv`

### DASH-B

All four saved DASH-B report trees are now complete:

- `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b/`
- `outputs/round2_2026_04/reports/round2-internvl3.5-8b-dash-b/`
- `outputs/round2_2026_04/reports/round2-llava-onevision-7b-dash-b/`
- `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-dash-b/`

Each one now contains:

- the seven main method CSVs
- the four feature-ablation CSVs
- `baselines.json`
- `ablations.csv`
- `split_sensitivity.csv`

The missing piece is the tracked paper table. `docs/tables/round2/table1_dash_b.md` and `.csv` have not been generated yet.

## Supplementary State

| Planned item | State | Notes |
| --- | --- | --- |
| POPE adversarial | partial | round-two eval caches exist for LLaVA and Molmo; only legacy cache roots exist for Qwen and InternVL; no round-two adversarial report dirs yet |
| RePOPE | missing | no round-two report dirs yet |
| Transfer and control matrix | missing | no round-two shared-bank, shuffled-bank, or heldout control report dirs yet |
| Bank-size ablation | missing | no tracked table files yet |
| Layer-count ablation | missing | no tracked table files yet |

## Comparator State

`outputs/round2_2026_04/readouts/` now exists, but comparator coverage is still incomplete.

| Model | POPE popular readouts | DASH-B readouts | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | missing | present, likely complete (`42` shards) | DASH-B only |
| InternVL3.5-8B | missing | partial (`36` shards) | stalled before completion |
| LLaVA-OneVision-7B | missing | partial (`7` shards) | stalled early |
| Molmo-7B-D-0924 | missing | present, complete after rerun (`42` shards in `main/`) | DASH-B only |

No HALP output directories are present.

No GLSim output directories are present.

So comparator execution has not actually started yet at the scoring stage. The project only has partial shared readout extraction, and only for DASH-B.

## What The Current Results Already Say

The saved round-two artifacts already support these conclusions:

- On `POPE popular`, full MIND clearly beats the simple output baselines on all four models.
- On `POPE popular`, the linear probe beats full MIND on all four models, especially on PR-AUC.
- On `DASH-B`, full MIND still beats the simple output baselines on all four completed report rows, but the linear probe is dramatically stronger.
- On `DASH-B`, the manifold step is not helping in the current saved reports. `no_manifold` is numerically better than full MIND on every completed row.

That means the paper can still claim superiority over simple output confidence methods, but it cannot claim superiority over stronger internal baselines from the current round-two evidence.

## Execution Problems That Are Still Real

- GPU occupancy alone is misleading. The machine can look busy while MIND is completely idle.
- The readout extraction layer is still brittle. Qwen and Molmo reached a clean `42`-shard DASH-B endpoint, while InternVL and especially LLaVA stopped partway and left no active writer behind.
- The report layer is now ahead of the paper layer. The saved DASH-B reports exist, but the tracked table files and paper docs still lag them.

## Immediate Priority

1. Generate the tracked DASH-B main table from the already-complete saved report trees.
2. Finish the missing comparator readout coverage, starting with `POPE popular`, because HALP and GLSim still have no usable popular inputs.
3. Run HALP and GLSim before spending more effort on lower-priority supplementary tables.
