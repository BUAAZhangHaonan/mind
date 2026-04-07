# Round-Four Status Audit

Date: 2026-04-07

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the serial queue log.

## Live Queue

The original GPU 1 queue did not finish. It stopped after one successful step, and was then relaunched in a split recovery shape.

- GPU 0 is still occupied by `magformer`.
- GPU 1 is now assigned to the recovered MIND extraction queue.
- Active tmux sessions:
  - `mind_gpu1_round2_queue`
  - `mind_glsim_cpu_queue`
  - `mind_halp_cpu_queue`
- Active queue logs:
  - `outputs/round2_2026_04/job_logs/mind_gpu1_serial_queue_20260407.log`
  - `outputs/round2_2026_04/job_logs/mind_glsim_cpu_queue_20260407.log`
  - `outputs/round2_2026_04/job_logs/mind_halp_cpu_queue_20260407.log`

The queue completed:

- `qwen3-vl-8b`
- benchmark: `POPE popular`
- stage: readout extraction
- output root: `outputs/round2_2026_04/readouts/qwen3-vl-8b/pope/popular/`
- result: complete (`47/47` shards)

The original queue then failed on the next step:

- `qwen3-vl-8b`
- benchmark: `POPE popular`
- stage: `GLSim image_grouped`
- command: `python scripts/run_glsim.py ... --device cuda --split-strategy image_grouped`
- result: killed with `exit=137`

The last queue timestamps in the saved log are:

- `[2026-04-07 15:48:00] DONE qwen3-vl-8b pope popular readouts`
- `[2026-04-07 15:58:16] FAIL qwen3-vl-8b pope popular GLSim image_grouped (exit=137)`

So the queue failure was not a readout failure. It was the first comparator step being run on the GPU queue.

Recovery decision for this pass:

- keep `qwen3-vl-8b` `POPE popular` readouts as complete and do not rerun them
- relaunch GPU 1 with extraction-only work
- move `GLSim` and `HALP` into separate CPU-side persistent queues

The recovered queues are now live:

- GPU queue current step:
  - `internvl3.5-8b`
  - benchmark: `POPE popular`
  - stage: readout extraction
  - output root: `outputs/round2_2026_04/readouts/internvl3.5-8b/pope/popular/`
- CPU comparator queue current step:
  - `qwen3-vl-8b`
  - benchmark: `POPE popular`
  - stage: `GLSim image_grouped`
  - output root: `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/`
- CPU comparator queue current step:
  - `qwen3-vl-8b`
  - benchmark: `POPE popular`
  - stage: `HALP image_grouped`
  - output root: `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/`

Current health check:

- the GPU queue log shows `internvl3.5-8b` model loading and active generation on GPU 1
- `run_glsim.py` and `run_halp.py` are both alive on CPU and consuming real CPU time
- no duplicate writers are targeting the same readout root

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

## Recovery Split

The recovered execution shape should be:

- one GPU 1 tmux queue for GPU-bound readout and eval-cache extraction only
- one CPU tmux queue for `GLSim`
- one CPU tmux queue for `HALP`

That split matches the observed failure. The GPU queue itself did not fail on readout extraction. It failed when a comparator step was run inside the same queue with `--device cuda`.

## Live Relaunch

The recovery queues are now mounted and alive in tmux:

- `mind_gpu1_round2_queue`
- `mind_glsim_cpu_queue`
- `mind_halp_cpu_queue`

Current live heads:

- GPU 1: `internvl3.5-8b` `POPE popular` readout extraction under `scripts/queue/mind_round2_gpu1_serial.sh`
- CPU GLSim queue: `qwen3-vl-8b` `POPE popular` `image_grouped`
- CPU HALP queue: `qwen3-vl-8b` `POPE popular` `image_grouped`

The relaunch is using the serial scripts that survived the queue cleanup:

- `scripts/queue/mind_round2_gpu1_serial.sh`
- `scripts/queue/mind_round2_glsim_cpu_serial.sh`
- `scripts/queue/mind_round2_halp_cpu_serial.sh`
