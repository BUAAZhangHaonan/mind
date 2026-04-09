# Round-Four Status Audit

Date: 2026-04-09

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the queue logs.

## Live Queue

The old split recovery queues are gone. The first unified queue also stopped, first on CPU RAM and then on disk pressure from oversized readout caches. The code has now been changed so comparator readouts are compact and transient instead of durable multi-terabyte artifacts.

Current live state:

- GPU 0 is the only GPU allowed for MIND work.
- GPU 1 is idle right now and untouched by MIND.
- The corrected unified queue is running again in `tmux`:
  - session: `mind_round2_unified_queue`
  - wait log: `outputs/round2_2026_04/job_logs/mind_wait_for_gpu0_20260409_resume6.log`
  - queue log: `outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260409_resume6.log`
- The latest remounted queue had previously advanced substantially, then stopped on a new error:
  - `internvl3.5-8b` `POPE popular` official HALP row split finished cleanly
  - `llava-onevision-7b` `POPE popular` official HALP row split finished cleanly
  - `molmo-7b-d-0924` `POPE popular` official HALP row split finished cleanly
  - `qwen3-vl-8b` `DASH-B` official HALP row split finished cleanly
  - `internvl3.5-8b` `DASH-B` readouts finished cleanly
  - `internvl3.5-8b` `DASH-B` official HALP row split then failed with `exit=1`
- That failure was not an OOM kill. The queue log showed:
  - `RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.`
  - the failure happens at the first `HALPProbe(...).to(device)` call
- The unified queue now has a CUDA preflight gate before GPU steps.
- That preflight already passed inside the live queue context:
  - `cuda_preflight_ok device=cuda:0 visible_devices=1`
- The current live MIND process is:
  - `python scripts/run_halp.py`
  - model: `internvl3.5-8b`
  - benchmark: `DASH-B`
  - protocol: official HALP row split
- The current live `run_halp.py` worker is active again under the remounted queue.
- The stop was intentional, not a host crash:
  - the current `HALP` runner was using the wrong baseline definition
  - the current `GLSim` path was mislabeled as if it were the official method
- The stale legacy readout tree was fully deleted before this restart:
  - path: `outputs/round2_2026_04/readouts/`
  - size before cleanup: `6.3T`
  - current compact readout tree size during this audit: about `15G`
  - `/home/team` free space recovered to `6.7T`
- The queue had already finished one compact `qwen3-vl-8b` `POPE popular` readout rebuild under:
  - `outputs/round2_2026_04/readouts/qwen3-vl-8b/pope/popular/`
- The qwen popular readout rebuild is complete with valid `vision_features`.
- Five corrected comparator artifacts are already saved:
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-halp-row/halp.json`
  - `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular-halp-row/halp.json`
  - `outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular-halp-row/halp.json`
  - `outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular-halp-row/halp.json`
  - `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b-halp-row/halp.json`
- The current code reset changed the comparator policy:
  - `run_halp.py` now targets the official 11-probe HALP setup on a stratified row split
  - the old grouped nested HALP path is no longer the paper-facing default
  - the old readout-based GLSim path is now `GLSim-adapted`, not official GLSim
- The `internvl3.5-8b` `POPE popular` and `DASH-B` compact readout rebuilds are complete under:
  - `outputs/round2_2026_04/readouts/internvl3.5-8b/pope/popular/`
  - `outputs/round2_2026_04/readouts/internvl3.5-8b/dash-b/main/`

The concrete design change for this pass is simple:

- readouts are no longer treated as durable artifacts
- new readout entries keep only comparator-needed tensors
- the queue is now being rebuilt around a paper-safe policy:
  - extract
  - corrected official HALP row split
  - delete that readout directory
  - do not schedule `GLSim-adapted` in the official queue

Current resource snapshot during this audit:

- `125 GiB` total RAM
- one live corrected `run_halp.py` worker for `internvl3.5-8b` `DASH-B`
- `GPU 0` is the active MIND lane again, even if instantaneous utilization can still read `0%` during CPU-side probe preparation
- `GPU 1` is idle right now and untouched by MIND

## Main Matrix State

| Model | POPE popular report | DASH-B reference bank | DASH-B features | DASH-B report | Compact comparator readouts |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | complete | complete | complete | complete | `POPE popular` and `DASH-B` corrected official HALP row splits saved |
| InternVL3.5-8B | complete | complete | complete | complete | `POPE popular` corrected official HALP row split saved; `DASH-B` readouts complete; `DASH-B` HALP row failed on CUDA init |
| LLaVA-OneVision-7B | complete | complete | complete | complete | `POPE popular` corrected official HALP row split saved |
| Molmo-7B-D-0924 | complete | complete | complete | complete | `POPE popular` corrected official HALP row split saved |

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

`outputs/round2_2026_04/readouts/` is no longer considered a stable result tree. The old full-hidden-state cache is gone and is now being rebuilt in compact form one unit at a time.

Current comparator state:

- Final official HALP outputs are saved for:
  - `qwen3-vl-8b` on `POPE popular`
  - `internvl3.5-8b` on `POPE popular`
  - `llava-onevision-7b` on `POPE popular`
  - `molmo-7b-d-0924` on `POPE popular`
  - `qwen3-vl-8b` on `DASH-B`
- No official GLSim outputs are saved for these benchmarks.
- The old readout-based similarity path now belongs under the explicit name `GLSim-adapted`.
- Legacy readout counts are no longer treated as progress because the old cache format was deleted.
- Comparator progress now depends on the active resumed `internvl3.5-8b` `DASH-B` HALP row run on `GPU 0`.

## What The Current Results Already Say

The saved round-two artifacts already support these conclusions:

- On `POPE popular`, full MIND clearly beats the simple output baselines on all four models.
- On `POPE popular`, the linear probe beats full MIND on all four models, especially on PR-AUC.
- On `DASH-B`, full MIND still beats the simple output baselines on all four completed report rows, but the linear probe is dramatically stronger.
- On `DASH-B`, the manifold step is not helping in the current saved reports. `no_manifold` is numerically better than full MIND on every completed row.

That means the paper can still claim superiority over simple output confidence methods, but it cannot claim superiority over stronger internal baselines from the current round-two evidence.

## Execution Problems That Are Still Real

- GPU occupancy alone is misleading. A job can spend a long time in model load before GPU utilization spikes, and a machine can also look idle after a killed queue.
- The old CPU comparator shape was unsafe for this host. `GLSim-adapted` and the old grouped `HALP` path both load large readout caches and were previously killed with `exit=137`.
- The old grouped `HALP` path was also methodologically wrong. It swept all layers and used grouped nested best-probe selection instead of the official 11-probe setup.
- The old readout-based `GLSim` path was also methodologically wrong for official claims. It is a queried-object pre-generation adaptation, not the official post-generation MSCOCO + CHAIR method.
- The post-restart proxy state broke repo-id model loading for InternVL. Cached local model paths or offline cached Hub loads are required now.
- The pipeline now has lock-file guards and retry wrappers, but host-level load still needs operational discipline.

## Immediate Priority

1. Let the corrected unified queue keep running corrected official HALP row-split jobs one unit at a time on `GPU 0`.
2. Keep `GPU 1` untouched for non-MIND work.
3. Keep the tracked paper tables MIND-only until corrected comparator artifacts exist.

## Unified Queue Policy

The split queue design is now retired.

- New queue script:
  - `scripts/queue/mind_round2_unified_serial.sh`
- Optional wait launcher:
  - `scripts/queue/start_mind_when_gpu0_free.sh`
- Old split runners now refuse future direct launches:
  - `scripts/queue/mind_round2_gpu1_serial.sh`
  - `scripts/queue/mind_round2_glsim_cpu_serial.sh`
  - `scripts/queue/mind_round2_halp_cpu_serial.sh`

The new policy is:

- one task at a time only
- one `(model, benchmark)` unit at a time
- extract compact readouts first
- corrected official HALP row split next, one run at a time
- do not schedule `GLSim-adapted` in the official queue
- delete the readout directory immediately after that unit has its corrected official comparator output
- lighter CPU pipeline stages only after comparator runs finish
- memory and GPU state logged before and after every step
- the virtual memory cap is now disabled by default because it blocked `HALP` even when the host still had plenty of free RAM
- a hard memory gate that refuses to start a new step if available RAM is below `10 GiB`

The unified queue is the only queue that should be used going forward. It can now resume mixed readout directories that contain both old single-file shards and new chunked manifest shards.
