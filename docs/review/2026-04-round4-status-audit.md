# Round-Four Status Audit

Date: 2026-04-07

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and job logs.

## Live Queue

The live machine state has changed materially since the earlier audit.

- GPU 0 is still occupied by `magformer` training via `python tools/train.py`.
- GPU 1 is now reserved for MIND through tmux session `mind_gpu1_round2_queue`.
- The queue log is:
  - `outputs/round2_2026_04/job_logs/mind_gpu1_serial_queue_20260407.log`
- The first live MIND job is:
  - `python scripts/extract_readout_states.py`
  - model: `qwen3-vl-8b`
  - benchmark: `POPE popular`
  - split: `popular`
  - output root: `outputs/round2_2026_04/readouts/qwen3-vl-8b/pope/popular/`

The queue is serial and resumable. It skips completed outputs and archives partial readout trees before reruns.

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

The tracked paper table already exists:

- `docs/tables/round2/table1_dash_b.md`
- `docs/tables/round2/table1_dash_b.csv`

## Supplementary State

| Planned item | State | Notes |
| --- | --- | --- |
| POPE adversarial | partial | round-two eval caches exist for LLaVA and Molmo; only legacy cache roots exist for Qwen and InternVL; no round-two adversarial report dirs yet |
| RePOPE | missing | no round-two report dirs yet |
| Transfer and control matrix | missing | no round-two shared-bank, shuffled-bank, or heldout control report dirs yet |
| Bank-size ablation | missing | no tracked table files yet |
| Layer-count ablation | missing | no tracked table files yet |

## Comparator State

`outputs/round2_2026_04/readouts/` now exists, and comparator recovery is now live on GPU 1.

| Model | POPE popular readouts | DASH-B readouts | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | running | present, complete (`42` shards) | current live queue head |
| InternVL3.5-8B | queued | partial (`36` shards) | queued for popular extraction, then DASH-B repair |
| LLaVA-OneVision-7B | queued | partial (`7` shards) | queued for popular extraction, then DASH-B repair |
| Molmo-7B-D-0924 | queued | present, complete after rerun (`42` shards in `main/`) | queued for popular extraction only |

No HALP output directories are present.

No GLSim output directories are present.

So comparator scoring is still missing, but the readout recovery path is no longer stalled. The live tmux queue is now moving through the missing readout coverage first.

## What The Current Results Already Say

The saved round-two artifacts already support these conclusions:

- On `POPE popular`, full MIND clearly beats the simple output baselines on all four models.
- On `POPE popular`, the linear probe beats full MIND on all four models, especially on PR-AUC.
- On `DASH-B`, full MIND still beats the simple output baselines on all four completed report rows, but the linear probe is dramatically stronger.
- On `DASH-B`, the manifold step is not helping in the current saved reports. `no_manifold` is numerically better than full MIND on every completed row.

That means the paper can still claim superiority over simple output confidence methods, but it cannot claim superiority over stronger internal baselines from the current round-two evidence.

## Execution Problems That Are Still Real

- GPU occupancy alone is misleading. The machine can look busy while MIND is idle, or it can look mostly quiet while a long model load is still alive.
- The readout extraction layer is still brittle. Qwen and Molmo reached a clean `42`-shard DASH-B endpoint, while InternVL and especially LLaVA stopped partway and left no active writer behind.
- The queue layer was previously too informal. Without a persistent runner, stopped shells looked too much like finished work.

## Immediate Priority

1. Finish the missing comparator readout coverage, starting with `POPE popular`, because HALP and GLSim still have no usable popular inputs there.
2. Run GLSim on both main benchmarks once the readouts for a model are complete.
3. Run HALP after the GPU-heavy readout and GLSim steps finish.
