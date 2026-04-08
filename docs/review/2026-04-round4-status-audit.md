# Round-Four Status Audit

Date: 2026-04-08

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the queue logs.

## Live Queue

The old split recovery queues are gone. The first unified queue also stopped, first on CPU RAM and then on disk pressure from oversized readout caches. The code has now been changed so comparator readouts are compact and transient instead of durable multi-terabyte artifacts.

Current live state:

- GPU 0 remains off-limits for MIND and is occupied by another project.
- GPU 1 is still the only allowed GPU for MIND, but it is currently occupied by `magformer`, not by this repo.
- There is no live MIND extractor, HALP, or GLSim process right now.
- The stale legacy readout tree is being deleted with low I/O priority:
  - path: `outputs/round2_2026_04/readouts/`
  - size before cleanup: `6.3T`
  - size during this audit snapshot: `2.9T`
  - `/home/team` free space recovered from `441G` to `3.8T`
- A wait-launch session is mounted in `tmux` and will start the new queue only when both conditions are true:
  - the legacy readout tree is fully gone
  - `GPU 1` is actually free
- Active wait session:
  - `mind_round2_wait_gpu1`
- Active wait log:
  - `outputs/round2_2026_04/job_logs/mind_wait_for_gpu1_20260408.log`
- The new queue log path will be:
  - `outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_disk_bounded.log`

The concrete design change for this pass is simple:

- readouts are no longer treated as durable artifacts
- new readout entries keep only comparator-needed tensors
- the queue now processes one `(model, benchmark)` unit at a time:
  - extract
  - HALP image_grouped
  - HALP object_heldout
  - GLSim image_grouped
  - GLSim object_heldout
  - delete that readout directory

Current memory snapshot during this audit:

- `125 GiB` total RAM
- no lingering `run_halp.py` or `run_glsim.py` workers
- no live MIND GPU process on `GPU 1`

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

`outputs/round2_2026_04/readouts/` is no longer considered a stable result tree. The old full-hidden-state cache is being deleted and will be rebuilt in compact form one unit at a time.

Current comparator state:

- No final HALP outputs are saved yet.
- No final GLSim outputs are saved yet.
- Legacy readout counts are no longer treated as progress because that old cache format is being removed.
- Comparator progress now depends on the new compact cache queue, not on the deleted legacy tree.

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

1. Finish deleting the legacy readout tree.
2. Let `mind_round2_wait_gpu1` hand off automatically to the unified queue when `GPU 1` is free.
3. Let the new queue rebuild compact readouts one unit at a time and delete each unit after HALP and GLSim finish.
4. As soon as a `glsim.json` or `halp.json` appears for a main-table row, regenerate the tracked paper tables and summary docs.

## Unified Queue Policy

The split queue design is now retired.

- New queue script:
  - `scripts/queue/mind_round2_unified_serial.sh`
- Wait launcher:
  - `scripts/queue/start_mind_when_gpu1_free.sh`
- Old split runners now refuse future direct launches:
  - `scripts/queue/mind_round2_gpu1_serial.sh`
  - `scripts/queue/mind_round2_glsim_cpu_serial.sh`
  - `scripts/queue/mind_round2_halp_cpu_serial.sh`

The new policy is:

- one task at a time only
- one `(model, benchmark)` unit at a time
- extract compact readouts first
- HALP next, one run at a time
- GLSim after HALP, one run at a time
- delete the readout directory immediately after that unit has both comparator outputs
- lighter CPU pipeline stages only after comparator runs finish
- memory and GPU state logged before and after every step
- a virtual memory cap set to `80%` of total RAM
- a hard memory gate that refuses to start a new step if available RAM is below `10 GiB`

The unified queue is the only queue that should be used going forward. It can now resume mixed readout directories that contain both old single-file shards and new chunked manifest shards.
