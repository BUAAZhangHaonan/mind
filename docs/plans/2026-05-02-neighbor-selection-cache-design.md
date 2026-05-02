# Neighbor Selection Cache Design

Conclusion: add cache identity metadata, shard-level reuse checks, and a comparison runner before regenerating anything. The raw eval/reference tensors should stay compact, while neighbor features should be treated as derived files that can be deleted and rebuilt.

## Phase A: Cache Architecture

- Keep the current shard shape: `torch.save(list[dict])`.
- Add a small metadata sidecar or shard header with model id, dataset id, split, selected layers, dtype, extractor version, input file hash, prompt/config hash, and schema version.
- Reuse `save_prefill_cache_shard` as the single writer path for eval and reference states.
- Add a cache planner that reads existing shards, compares required ids and hashes, and writes only missing or stale shards.
- Extend `scripts/experiments/verify_cache_coverage.py` so it reports covered ids, missing ids, stale ids, dtype, selected layers, and vector shape.

Storage estimate:

```text
raw_bytes = num_examples * num_selected_layers * hidden_dim * bytes_per_value
raw_gib = raw_bytes / 1024^3
total_raw_gib ~= eval_raw_gib + reference_raw_gib
```

Target expectation: the refreshed raw cache should land around 8-10GB when tensors are stored as float16 and only selected layers are kept.

Strict disk behavior:

- Stored tensors are float16.
- No pairwise distance matrices are written to disk.
- Computed feature parquet/csv outputs are derived and deletable.
- Regeneration never rewrites a valid shard only because a later feature experiment changed.

## Phase B: tmux Regeneration

- Run cache planning first and write a manifest of missing/stale work.
- Launch only the planned eval/reference extraction jobs in tmux.
- Prefer model/dataset shards that can finish independently, so failed jobs do not invalidate completed shards.
- After each tmux batch, run coverage verification and save a short regeneration report.
- Do not build neighbor features during raw cache regeneration.

## Phase C: Neighbor Selection Comparison

Compare the five requested methods from the same reusable raw cache:

- `knn_angular_k30`
- `kernel_knn_k30`
- `radius_ball`
- `knn_cosine_k30`
- `knn_euclidean_k30`

The comparison should use `src/mind/geometry/local_neighborhood.py` and `src/mind/geometry/gpu_distances.py` for shared selection logic. Feature builders should read raw eval/reference shards, compute query-local neighbors on demand, write only compact feature outputs, and record the method name plus parameters in report metadata.

Completion: a later implementation can prove that unchanged raw shards are reused, stale shards regenerate narrowly, and all five methods can be compared without storing pairwise matrices.
