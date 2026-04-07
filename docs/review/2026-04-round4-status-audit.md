# Round-Four Status Audit

Date: 2026-04-07

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the serial queue log.

## Live Queue

The previous GPU 1 queue did not finish. It stopped after one successful step.

- GPU 0 is still occupied by `magformer`.
- GPU 1 is now free.
- The tmux session `mind_gpu1_round2_queue` no longer exists.
- The queue log is:
  - `outputs/round2_2026_04/job_logs/mind_gpu1_serial_queue_20260407.log`

The queue completed:

- `qwen3-vl-8b`
- benchmark: `POPE popular`
- stage: readout extraction
- output root: `outputs/round2_2026_04/readouts/qwen3-vl-8b/pope/popular/`
- result: complete (`47/47` shards)

The queue then failed on the next step:

- `qwen3-vl-8b`
- benchmark: `POPE popular`
- stage: `GLSim image_grouped`
- command: `python scripts/run_glsim.py ... --device cuda --split-strategy image_grouped`
- result: killed with `exit=137`

So the queue failure was not a readout failure. It was the first comparator step being run on the GPU queue.

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

`outputs/round2_2026_04/readouts/` exists, but the recovery queue is no longer live.

| Model | POPE popular readouts | DASH-B readouts | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | complete (`47` shards) | complete (`42` shards) | readouts are usable; GLSim failed before writing results |
| InternVL3.5-8B | missing | partial (`36` shards) | still needs popular readouts and DASH-B repair |
| LLaVA-OneVision-7B | missing | partial (`7` shards) | still needs popular readouts and DASH-B repair |
| Molmo-7B-D-0924 | missing | complete after rerun (`42` shards in `main/`) | still needs popular readouts |

No HALP output directories are present.

No GLSim output directories are present.

So comparator scoring is still missing. The next recovery pass needs to split GPU-only extraction from CPU-side comparator runs instead of keeping them in one queue.

## What The Current Results Already Say

The saved round-two artifacts already support these conclusions:

- On `POPE popular`, full MIND clearly beats the simple output baselines on all four models.
- On `POPE popular`, the linear probe beats full MIND on all four models, especially on PR-AUC.
- On `DASH-B`, full MIND still beats the simple output baselines on all four completed report rows, but the linear probe is dramatically stronger.
- On `DASH-B`, the manifold step is not helping in the current saved reports. `no_manifold` is numerically better than full MIND on every completed row.

That means the paper can still claim superiority over simple output confidence methods, but it cannot claim superiority over stronger internal baselines from the current round-two evidence.

## Execution Problems That Are Still Real

- GPU occupancy alone is misleading. The machine can look busy while MIND is idle, or it can look quiet after a killed job.
- The readout extraction layer still lacks true resume semantics. The old queue compensated by archiving partial trees, which is wasteful and fragile.
- The queue shape was wrong. It mixed GPU-bound extraction with comparator runs that can be moved to CPU-side persistent queues.
- There is still no lock-file guard against duplicate writers targeting the same output root.

## Immediate Priority

1. Fix the queue brittle points first: readout resume, retries, object-heldout validation, and lock files.
2. Relaunch GPU 1 with extraction-only work, starting from `internvl3.5-8b` `POPE popular` readouts.
3. Move `GLSim` and `HALP` into separate CPU-side tmux queues so a comparator failure does not waste GPU time.
