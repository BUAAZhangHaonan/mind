# Round-Four Status Audit

Date: 2026-04-09

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the queue logs.

## Live Queue

The old split recovery queues are gone. The first unified queue also stopped, first on CPU RAM and then on disk pressure from oversized readout caches. The code has now been changed so comparator readouts are compact and transient instead of durable multi-terabyte artifacts.

Current live state:

- GPU 0 is the only GPU allowed for MIND work.
- GPU 1 stays reserved for `magformer`.
- The old live comparator queue was stopped on purpose.
- There is no active MIND comparator process right now.
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
- No corrected comparator artifacts are saved yet.
- The current code reset changed the comparator policy:
  - `run_halp.py` now targets the official 11-probe HALP setup on a stratified row split
  - the old grouped nested HALP path is no longer the paper-facing default
  - the old readout-based GLSim path is now `GLSim-adapted`, not official GLSim
- No other compact readout unit has been rebuilt yet under `outputs/round2_2026_04/readouts/`.

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
- no live MIND comparator worker
- `GPU 0` is currently free for the corrected MIND queue
- `GPU 1` is occupied by `magformer`

## Main Matrix State

| Model | POPE popular report | DASH-B reference bank | DASH-B features | DASH-B report | Compact comparator readouts |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | complete | complete | complete | complete | `POPE popular` compact readouts rebuilt successfully; no corrected comparator artifact saved yet |
| InternVL3.5-8B | complete | complete | complete | complete | pending compact rebuild |
| LLaVA-OneVision-7B | complete | complete | complete | complete | pending compact rebuild |
| Molmo-7B-D-0924 | complete | complete | complete | complete | pending compact rebuild |

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

- No final official HALP outputs are saved yet.
- No official GLSim outputs are saved for these benchmarks.
- The old readout-based similarity path now belongs under the explicit name `GLSim-adapted`.
- Legacy readout counts are no longer treated as progress because the old cache format was deleted.
- Comparator progress now depends on the corrected queue being remounted on `GPU 0`.

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

1. Remount the corrected unified queue on `GPU 0`.
2. Run corrected official HALP row-split jobs one unit at a time after validating readouts.
3. Keep `GPU 1` untouched for `magformer`.
4. Keep the tracked paper tables MIND-only until corrected comparator artifacts exist.

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
