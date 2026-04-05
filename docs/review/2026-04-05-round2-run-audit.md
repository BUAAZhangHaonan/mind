# Round-Two Run Audit

## Conclusion

Round two is only half-finished on disk.

- The new-model cache work completed cleanly.
- The paper-facing report layer is still missing most of the real outputs.
- One runtime default still points at Haar even though the paper freeze now says simple stats.

## Verified State

- Branch: `master`
- Working tree: clean
- Active long jobs: none

### Completed caches

- `outputs/round2_2026_04/cache/llava-onevision-7b/pope/popular`
  - `24` shards, `shard-00000.pt` to `shard-00023.pt`
- `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/popular`
  - `24` shards, `shard-00000.pt` to `shard-00023.pt`
- `outputs/round2_2026_04/cache/llava-onevision-7b/pope-reference/train`
  - `40` shards, `shard-00000.pt` to `shard-00039.pt`
- `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope-reference/train`
  - `40` shards, `shard-00000.pt` to `shard-00039.pt`

### Cache payload spot-check

LLaVA-OneVision and Molmo both look sane on the current round-two cache path.

- cache entries use `layer_vectors`
- `selected_layers` length is `14`
- hidden size is `3584`
- first-token logits are present
- parsed answers are present

## Incomplete or Missing Outputs

### Partial popular reports

These two report directories are incomplete and should not be treated as valid final outputs.

- `outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular`
- `outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular`

Each one currently contains only:

- `variant_results/full.csv`
- `variant_results/drift_only.csv`
- `variant_results/no_manifold.csv`
- `variant_results/linear_probe.csv`

Each one is still missing:

- `variant_results/output_p_yes.csv`
- `variant_results/output_logit_margin.csv`
- `variant_results/output_chosen_answer_confidence.csv`
- `baselines.json`
- `ablations.csv`
- `split_sensitivity.csv`

### Missing round-two downstream trees

These non-smoke round-two directories do not exist yet.

- `outputs/round2_2026_04/features`
- `outputs/round2_2026_04/readouts`
- `outputs/round2_2026_04/reference_banks`
- `outputs/round2_2026_04/reference_banks_shared`
- `outputs/round2_2026_04/reference_banks_shuffled`

That means the main execution work is still ahead of us.

## Paper And Export Gaps

- `docs/paper_outline.md` is aligned with the phase-one freeze.
- `docs/results_summary.md` is behind the actual run state:
  - it still says the LLaVA and Molmo reference caches are pending
- `scripts/export_paper_package.py` still reads the old two-model closeout layout and cannot produce the final round-two package as-is

## Runtime Mismatch To Fix Before Reruns

The frozen method choice is now simple stats, but the runtime default still points at Haar.

- current default in `src/mind/evaluation/baselines.py`: `raw_plus_calibrated_haar`
- that default flows into:
  - `scripts/compute_baselines.py`
  - `scripts/train_detector.py`
  - `scripts/run_experiment.py`

This needs to change before any more paper-facing reruns.

## Decision

The next clean step is:

1. fix the default full variant to `raw_plus_calibrated_simple`
2. rebuild the popular report outputs cleanly
3. generate every later table from round-two-local artifacts only
