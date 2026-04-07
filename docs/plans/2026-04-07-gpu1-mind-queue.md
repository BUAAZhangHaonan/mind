# GPU1 Round-Two Queue Plan

Date: 2026-04-07

This plan records the remaining MIND model-side execution work now that GPU 1 is free and GPU 0 remains occupied by `magformer`.

## Live Starting Point

- GPU 0 is busy with `magformer` and must not be touched.
- GPU 1 is free.
- There is no live MIND tmux session or background runner.
- The saved round-two baseline reports for `POPE popular` and `DASH-B` already exist.
- The remaining GPU-dependent work is:
  - `POPE popular` readout extraction for all four models
  - `DASH-B` readout repair for InternVL and LLaVA
  - GLSim runs on `POPE popular` and `DASH-B`
  - adversarial eval extraction for Qwen and InternVL
- The remaining probe-training work is:
  - HALP runs on `POPE popular` and `DASH-B`

## Queue Design

- Use one persistent tmux session on GPU 1 only.
- Use one serial script with clear skip rules.
- Skip only when the output is already complete.
- Preserve partial outputs by renaming them before reruns.
- Keep GPU-heavy work first so GPU 1 does not sit idle behind CPU-only probe fitting.

## Queue Order

1. Extract `POPE popular` readouts for:
   - Qwen3-VL-8B
   - InternVL3.5-8B
   - LLaVA-OneVision-7B
   - Molmo-7B-D-0924
2. Repair incomplete `DASH-B` readouts for:
   - InternVL3.5-8B
   - LLaVA-OneVision-7B
   - skip Qwen and Molmo if the saved `42`-shard trees are intact
3. Run GLSim on `POPE popular`:
   - `image_grouped`
   - `object_heldout`
   - all four models
4. Run GLSim on `DASH-B`:
   - `image_grouped`
   - `object_heldout`
   - all four models
5. Extract `POPE adversarial` eval caches for:
   - Qwen3-VL-8B
   - InternVL3.5-8B
   - skip LLaVA and Molmo because the round-two caches already exist
6. Run HALP on `POPE popular`:
   - `image_grouped`
   - `object_heldout`
   - all four models
7. Run HALP on `DASH-B`:
   - `image_grouped`
   - `object_heldout`
   - all four models

## Completion Criteria

- A tmux session is mounted and running under a stable name.
- The queue log is written under `outputs/round2_2026_04/job_logs/`.
- Every queued step either:
  - completes and leaves the expected files behind, or
  - stops the queue with a clear failing command in the log.
- The live queue state is recorded in a tracked review note.

## Notes

- Qwen uses the local model config `configs/models/qwen3_vl_8b_local.yaml` for all new GPU work.
- `POPE popular` and `POPE adversarial` use `data/coco/val2014` as `--image-root`.
- `DASH-B` uses `data/dash_b` as `--image-root` because the normalized records already store paths under `images/...`.
