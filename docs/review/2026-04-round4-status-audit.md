# Round-Four Status Audit

Date: 2026-04-07

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the serial queue log.

## Live Queue

The original GPU 1 queue did not finish. It stopped after one successful step, was relaunched in a split recovery shape, and then the server crashed hard enough to drop all SSH sessions and clear every tmux queue.

Post-restart audit:

- GPU 0 is idle right now, but it remains off-limits for this project.
- GPU 1 is the only GPU allowed for MIND work.
- The restart cleared all old tmux sessions, including:
  - `mind_gpu1_round2_queue`
  - `mind_glsim_cpu_queue`
  - `mind_halp_cpu_queue`
- The pre-crash logs are still available:
  - `outputs/round2_2026_04/job_logs/mind_gpu1_serial_queue_20260407.log`
  - `outputs/round2_2026_04/job_logs/mind_glsim_cpu_queue_20260407.log`
  - `outputs/round2_2026_04/job_logs/mind_halp_cpu_queue_20260407.log`
- The recovered post-restart GPU log is now:
  - `outputs/round2_2026_04/job_logs/mind_gpu1_serial_queue_20260407_postrestart.log`

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

Crash diagnosis for this pass:

- the GPU extraction queue and the CPU comparator queues died for different reasons
- the GPU queue hit a post-restart proxy failure while loading InternVL from a Hub repo id
- both CPU comparator queues were killed with `exit=137` while loading and scoring the full `qwen3-vl-8b` popular readout cache
- the strongest concrete host-risk signal is the pair of CPU comparator jobs, not the GPU extraction itself

Recovery decision for this pass:

- keep `qwen3-vl-8b` `POPE popular` readouts as complete and do not rerun them
- keep MIND on GPU 1 only
- relaunch only the GPU extraction queue first
- leave the CPU comparator queues paused until the host is stable again
- force the queue onto cached local model paths or offline cached Hub paths so the dead proxy does not break extraction

The recovered queue is live again:

- tmux session:
  - `mind_gpu1_round2_queue`
- GPU queue current step:
  - `internvl3.5-8b`
  - benchmark: `POPE popular`
  - stage: readout extraction
  - output root: `outputs/round2_2026_04/readouts/internvl3.5-8b/pope/popular/`
- health check:
  - GPU 1 is occupied by `extract_readout_states.py`
  - the post-restart log shows InternVL loading successfully from the cached local model path
  - the run has moved past the earlier proxy failure and is actively generating on GPU 1
- intentionally not relaunched yet:
  - `mind_glsim_cpu_queue`
  - `mind_halp_cpu_queue`

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

`outputs/round2_2026_04/readouts/` exists. The GPU recovery queue is live again, while the CPU comparator queues remain intentionally paused after the crash.

| Model | POPE popular readouts | DASH-B readouts | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | complete (`47` shards) | complete (`42` shards) | readouts are usable; GLSim failed before writing results |
| InternVL3.5-8B | missing | partial (`36` shards) | still needs popular readouts and DASH-B repair |
| LLaVA-OneVision-7B | missing | partial (`7` shards) | still needs popular readouts and DASH-B repair |
| Molmo-7B-D-0924 | missing | complete after rerun (`42` shards in `main/`) | still needs popular readouts |

No HALP output directories are present yet.

No GLSim output directories are present yet.

So comparator scoring is still missing. The queue shape has been corrected, but the CPU comparator side now needs to be resumed more carefully than before.

## What The Current Results Already Say

The saved round-two artifacts already support these conclusions:

- On `POPE popular`, full MIND clearly beats the simple output baselines on all four models.
- On `POPE popular`, the linear probe beats full MIND on all four models, especially on PR-AUC.
- On `DASH-B`, full MIND still beats the simple output baselines on all four completed report rows, but the linear probe is dramatically stronger.
- On `DASH-B`, the manifold step is not helping in the current saved reports. `no_manifold` is numerically better than full MIND on every completed row.

That means the paper can still claim superiority over simple output confidence methods, but it cannot claim superiority over stronger internal baselines from the current round-two evidence.

## Execution Problems That Are Still Real

- GPU occupancy alone is misleading. A job can spend a long time in model load before GPU utilization spikes, and a machine can also look idle after a killed queue.
- The old CPU comparator shape was unsafe for this host. `GLSim` and `HALP` both load full readout caches, and the two `qwen3-vl-8b` runs were both killed with `exit=137`.
- The post-restart proxy state broke repo-id model loading for InternVL. Cached local model paths or offline cached Hub loads are required now.
- The pipeline now has lock-file guards and retry wrappers, but host-level load still needs operational discipline.

## Immediate Priority

1. Let the GPU 1 extraction queue finish the missing readout and adversarial cache work.
2. Resume comparator scoring only after the host is stable, and do not restart both heavy CPU comparator queues at the same time.
3. As soon as a `glsim.json` or `halp.json` appears for a main-table row, regenerate the tracked paper tables and summary docs.

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
