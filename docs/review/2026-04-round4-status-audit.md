# Round-Four Status Audit

Date: 2026-04-07

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the queue logs.

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
- keep the CPU comparator queues down until a single unified serial queue takes over
- force the queue onto cached local model paths or offline cached Hub paths so the dead proxy does not break extraction
- replace the separate GPU and CPU queue scripts with one unified serial queue that never runs two heavy tasks at once

The recovered GPU queue is live again:

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
  - the live shard count has reached `19`
- intentionally not relaunched:
  - `mind_glsim_cpu_queue`
  - `mind_halp_cpu_queue`
- current memory snapshot during this audit:
  - `125 GiB` total RAM
  - `56 GiB` to `81 GiB` available RAM across repeated checks
  - swap is present and mostly free
  - no lingering `run_halp.py` or `run_glsim.py` workers remain
  - no partial `halp.*` or `glsim.*` outputs were found under `outputs/round2_2026_04/`

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
2. Do not relaunch HALP and GLSim as separate queues.
3. Start the new unified serial queue only after the current GPU queue is fully idle.
4. As soon as a `glsim.json` or `halp.json` appears for a main-table row, regenerate the tracked paper tables and summary docs.

## Unified Queue Policy

The split queue design is now retired.

- New queue script:
  - `scripts/queue/mind_round2_unified_serial.sh`
- Old split runners now refuse future direct launches:
  - `scripts/queue/mind_round2_gpu1_serial.sh`
  - `scripts/queue/mind_round2_glsim_cpu_serial.sh`
  - `scripts/queue/mind_round2_halp_cpu_serial.sh`

The new policy is:

- one task at a time only
- GPU extraction first
- HALP next, one run at a time
- GLSim after HALP, one run at a time
- lighter CPU pipeline stages only after comparator runs finish
- memory and GPU state logged before and after every step
- a virtual memory cap set to `80%` of total RAM
- a hard memory gate that refuses to start a new step if available RAM is below `10 GiB`

The unified queue has been prepared, but it has not been launched yet because the current GPU extraction queue is still active. That is intentional. The queue guard now refuses to start while another MIND extraction process is alive.
