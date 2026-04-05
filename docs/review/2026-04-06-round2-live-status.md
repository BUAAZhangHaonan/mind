# Round-Two Live Status

Date: 2026-04-06

This note records the actual round-two run state after the long jobs from the previous round were checked again.

## What is complete

- LLaVA-OneVision `POPE popular` eval cache is complete: `24` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/pope/popular`.
- Molmo `POPE popular` eval cache is complete: `24` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/popular`.
- LLaVA-OneVision COCO reference cache is complete: `40` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/pope-reference/train`.
- Molmo COCO reference cache is complete: `40` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope-reference/train`.
- LLaVA-OneVision `DASH-B` eval cache is complete: `21` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/dash-b/main`.
- Molmo `DASH-B` eval cache is complete: `21` shards under `outputs/round2_2026_04/cache/molmo-7b-d-0924/dash-b/main`.
- The frozen feature decision remains unchanged: full MIND uses `raw + calibrated simple stats`.

## What is still running

- Qwen `POPE popular` round-two baseline rerun is still live. It has written `full.csv`, `drift_only.csv`, and `no_manifold.csv`, but the rest of the report is not finished yet.
- InternVL `POPE popular` drift regeneration is still live. No round-two-local feature parquet has been written yet.
- LLaVA-OneVision `POPE popular` drift regeneration is still live. No round-two-local feature parquet has been written yet.
- Molmo `POPE popular` drift regeneration is still live. No round-two-local feature parquet has been written yet.
- InternVL `DASH-B` eval extraction is still live. It has written the first `5` shards under `outputs/round2_2026_04/cache/internvl3.5-8b/dash-b/main`.
- LLaVA-OneVision `DASH-B` positive-reference caching is still live. It has written the first `6` shards under `outputs/round2_2026_04/cache/llava-onevision-7b/dash-b-reference/train`.

## What is incomplete or blocked

- The official Qwen and InternVL `POPE popular` round-two report directories are still incomplete. They cannot be treated as final paper inputs yet.
- There are still no round-two `readouts/` outputs, so HALP and GLSim have not started.
- There are still no tracked round-two main tables beyond the frozen phase-one decision note.
- Qwen `DASH-B` eval extraction is blocked by missing local model shards for `Qwen/Qwen3-VL-8B-Instruct`. The local Hugging Face snapshot still lacks the four `model-0000x-of-00004.safetensors` files, and the attempted resume download failed with an SSL EOF error.

## What this means

- The project is still in execution, not export.
- The next valid paper-facing tables must come from the live round-two reruns, not from the old March closeout package.
- `Qwen DASH-B` is the only current blocker that does not look like a local code issue.
