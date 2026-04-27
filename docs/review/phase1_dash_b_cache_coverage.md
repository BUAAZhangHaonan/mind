# Phase 1 DASH-B Cache Coverage

## Scope

Phase 1 regenerated the missing DASH-B eval caches for:

- `internvl3.5-8b`
- `llava-onevision-7b`

The job ran in `tmux` session `mind_phase1_cache` with `CUDA_VISIBLE_DEVICES=0`.

## Inputs

- DASH-B normalized records: `outputs/round2_2026_04/normalized/dash-b/main.jsonl`
- Row count: `2682`
- DASH-B reference candidates: `outputs/round2_2026_04/reference_candidates/dash_b_positive_reference_candidates.json`
- Candidate count: `1341`

## Output Structure

| model | eval shards | eval entries | layer count | vector dim | reference cache shards | reference cache entries | bank objects | bank rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-vl-8b | 21 | 2682 | 16 | 4096 | existing | existing | 70 | 1120 |
| internvl3.5-8b | 21 | 2682 | 16 | 4096 | 11 | 1341 | 70 | 1120 |
| llava-onevision-7b | 21 | 2682 | 14 | 3584 | 11 | 1341 | 70 | 980 |
| molmo-7b-d-0924 | 21 | 2682 | 14 | 3584 | existing | existing | 70 | 980 |

InternVL uses the same selected layers and vector dimension as Qwen. LLaVA uses the same selected layers and vector dimension as Molmo.

The LLaVA raw reference-state cache was regenerated on GPU 0. Its generated answer parsing did not produce usable `parsed_answer == 1` rows, so the DASH-B LLaVA object-bank metadata was rebuilt from the saved object-bank tensors that were already on disk. That preserves the cleaned object bank used by the existing reports while still leaving the regenerated raw reference cache available for inspection.

## Coverage Verification

### qwen3-vl-8b

```text
total_records: 2682
cached_entries: 2682
missing_count: 0
duplicate_count: 0
selected_layers: [[9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]]
selected_layers_consistent: true
vector_dims: [4096]
layer_dim_consistent: true
missing_ids:
```

### internvl3.5-8b

```text
total_records: 2682
cached_entries: 2682
missing_count: 0
duplicate_count: 0
selected_layers: [[9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]]
selected_layers_consistent: true
vector_dims: [4096]
layer_dim_consistent: true
missing_ids:
```

### llava-onevision-7b

```text
total_records: 2682
cached_entries: 2682
missing_count: 0
duplicate_count: 0
selected_layers: [[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
selected_layers_consistent: true
vector_dims: [3584]
layer_dim_consistent: true
missing_ids:
```

### molmo-7b-d-0924

```text
total_records: 2682
cached_entries: 2682
missing_count: 0
duplicate_count: 0
selected_layers: [[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
selected_layers_consistent: true
vector_dims: [3584]
layer_dim_consistent: true
missing_ids:
```

## Runtime Notes

InternVL extraction completed at batch size 4. LLaVA DASH-B eval extraction initially exceeded GPU memory at batch size 4, so it was rerun at batch size 1 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The completed run used GPU 0 only and produced all expected shards.
