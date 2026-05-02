# Neighbor Selection Cache Implementation Plan

Goal: implement efficient cache reuse, minimal regeneration, and neighbor-selection comparison on `master` with milestone commits. Do not start production code until the tests below are written and watched fail.

## Completion Criteria

- Raw cache shards store float16 tensors and include enough metadata to detect stale shards.
- Cache planning rewrites only missing or stale shards.
- No pairwise matrices are written to disk.
- Derived neighbor features can be deleted and rebuilt from raw cache shards.
- The comparison reports exactly these methods: `knn_angular_k30`, `kernel_knn_k30`, `radius_ball`, `knn_cosine_k30`, `knn_euclidean_k30`.
- `tests/unit/test_disk_budget.py` and `tests/unit/test_neighbor_selection.py` pass.
- Milestone commits are made and pushed on `master` after Phase A, Phase C code, Phase B cache regeneration, and final reports.

## Phase A: Cache Metadata And Disk Budget

Checklist:

- Add `tests/unit/test_disk_budget.py` with a failing test for the raw cache storage formula.
- Add a failing test that rejects float32 raw cache writes.
- Add a failing test that proves pairwise matrix paths are not part of the raw cache plan.
- Run `PYTHONNOUSERSITE=1 conda run --no-capture-output -n mind-py311 python -m pytest -q tests/unit/test_disk_budget.py` and confirm the expected failures.
- Add cache metadata helpers near the shared extraction/writer path.
- Update `src/mind/extractors/prefill.py` so `save_prefill_cache_shard` records dtype/schema/config identity while preserving the list-of-dicts payload.
- Update `scripts/extract_eval_states.py` and `scripts/cache_reference_states.py` to pass required metadata inputs.
- Extend `scripts/experiments/verify_cache_coverage.py` to flag missing, stale, dtype, selected-layer, and shape issues.
- Re-run the disk-budget tests.
- Commit and push: `Add cache metadata and disk budget checks`.

Completion: valid shards are identifiable, stale shards are reported, raw tensors are float16, and disk-budget tests pass.

## Phase C: Neighbor Selection Code And Tests

Checklist:

- Add `tests/unit/test_neighbor_selection.py` with fixtures for small query/reference tensors.
- Write failing tests for all five method names: `knn_angular_k30`, `kernel_knn_k30`, `radius_ball`, `knn_cosine_k30`, `knn_euclidean_k30`.
- Write a failing test that proves neighbor selection returns indices/features without saving pairwise matrices.
- Run `PYTHONNOUSERSITE=1 conda run --no-capture-output -n mind-py311 python -m pytest -q tests/unit/test_neighbor_selection.py` and confirm the expected failures.
- Implement shared selection logic in `src/mind/geometry/local_neighborhood.py`.
- Reuse GPU primitives from `src/mind/geometry/gpu_distances.py` where available.
- Update `scripts/experiments/build_query_local_features.py` to accept a method parameter and write compact derived features.
- Update `scripts/experiments/train_gpu_detector.py` only if report metadata needs the method name.
- Re-run `tests/unit/test_neighbor_selection.py`.
- Run the narrow feature-builder smoke test selected from the existing test suite.
- Commit and push: `Add neighbor selection comparison methods`.

Completion: all five methods are available through one comparison path, and tests prove no pairwise matrices are persisted.

## Phase B: Minimal tmux Cache Regeneration

Checklist:

- Run the coverage planner against current eval and reference caches.
- Save the missing/stale manifest before launching jobs.
- Start tmux regeneration jobs only for manifest entries.
- Run `scripts/experiments/verify_cache_coverage.py` after each batch.
- Confirm unchanged valid shards keep their paths and modification times.
- Confirm regenerated shards are float16 and metadata-matched.
- Commit and push: `Regenerate only stale cache shards`.

Completion: regenerated cache coverage is complete, and valid shards were not rewritten.

## Final Reports

Checklist:

- Run the neighbor-selection comparison for `knn_angular_k30`, `kernel_knn_k30`, `radius_ball`, `knn_cosine_k30`, and `knn_euclidean_k30`.
- Write the comparison table and short report under the existing docs/results location used by the experiment scripts.
- Re-run `tests/unit/test_disk_budget.py` and `tests/unit/test_neighbor_selection.py`.
- Run the narrow coverage verifier on the final cache root.
- Check `git status --short` and confirm `docs/reference.md` is untouched.
- Commit and push: `Report neighbor selection cache comparison`.

Completion: final reports list all five methods, tests pass, coverage is verified, and pushed commits exist for each milestone.
