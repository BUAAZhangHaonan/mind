# Round-Four Status Audit

Date: 2026-04-08

This note records the live round-two state after checking the process list, GPU owners, output roots, report directories, and the queue logs.

## Live Queue

The split recovery queues are gone. The unified serial queue was relaunched on `GPU 1`, resumed from the unfinished readout work, and then failed on `LLaVA-OneVision-7B` `POPE popular` readout extraction with a CPU allocation error.

Current live state:

- GPU 0 remains off-limits for MIND and is occupied by another project.
- GPU 1 is the only allowed GPU for MIND and is currently idle for this repo.
- There is no live MIND extractor, HALP, or GLSim process right now.
- The last queue log is:
  - `outputs/round2_2026_04/job_logs/mind_round2_unified_serial_20260408_restart.log`

What completed before the stop:

- `qwen3-vl-8b`
  - benchmark: `POPE popular`
  - stage: readout extraction
  - result: complete (`47/47` shards)
- `internvl3.5-8b`
  - benchmark: `POPE popular`
  - stage: readout extraction
  - result: complete (`47/47` shards)

Where it stopped:

- `llava-onevision-7b`
  - benchmark: `POPE popular`
  - stage: readout extraction
  - current output root: `outputs/round2_2026_04/readouts/llava-onevision-7b/pope/popular/`
  - current progress: `29/47` completed top-level shards
  - failure: `RuntimeError: DefaultCPUAllocator: can't allocate memory`
  - failing allocation in the saved traceback: about `747823104` bytes

The concrete failure point was not GPU memory. It was CPU RAM pressure inside readout extraction while building and retaining full hidden-state payloads for a shard.

Recovery decision for this pass:

- keep `qwen3-vl-8b` and `internvl3.5-8b` `POPE popular` readouts as complete
- keep MIND on GPU 1 only
- keep HALP and GLSim down until readout extraction is stable again
- resume from the unfinished `llava-onevision-7b` `POPE popular` readout step
- use the unified serial queue only
- fix the extractor-side host RAM spike instead of changing the benchmark method

The extractor has now been patched to:

- stack prefill hidden states without the extra `torch.stack([...cpu()...])` spike
- write new readout shards in chunked batch parts plus a top-level manifest
- keep queue shard counting compatible with both the old single-file shards and the new manifest shards

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

`outputs/round2_2026_04/readouts/` exists. The GPU recovery queue is live again, while the CPU comparator queues remain intentionally paused after the crash.

| Model | POPE popular readouts | DASH-B readouts | Status |
| --- | --- | --- | --- |
| Qwen3-VL-8B | complete (`47` shards) | complete (`42` shards) | readouts are usable; GLSim failed before writing results |
| InternVL3.5-8B | complete (`47` shards) | partial (`36` shards) | popular readouts are now usable; DASH-B still needs repair |
| LLaVA-OneVision-7B | partial (`29` shards) | partial (`7` shards) | popular readouts need resume from shard `29`; DASH-B still needs repair |
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

1. Resume the unified serial queue on GPU 1 from `llava-onevision-7b` `POPE popular` readouts.
2. Do not relaunch HALP and GLSim until the remaining readouts are complete.
3. After readouts are stable again, let the same unified queue continue through comparator work one step at a time.
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

The unified queue is the only queue that should be used going forward. It can now resume mixed readout directories that contain both old single-file shards and new chunked manifest shards.
