# Round-Two Run Audit

Date: 2026-04-05

## Conclusion

The round-two cache layer is partly complete, but the paper-facing report layer is not.

- Qwen and InternVL `POPE popular` report directories are partial and cannot be used as final paper sources.
- LLaVA-OneVision and Molmo `POPE popular` eval caches and COCO reference caches are complete and look structurally valid.
- The round-two exporter and summary docs still describe the old closeout world rather than the current run state.

## Verified State

### Repo and job state

- branch: `master`
- working tree: clean
- active long-running experiment jobs: none

### Round-two reports

Present under `outputs/round2_2026_04/reports/`:

- `round2-qwen3-vl-8b-popular/variant_results/full.csv`
- `round2-qwen3-vl-8b-popular/variant_results/drift_only.csv`
- `round2-qwen3-vl-8b-popular/variant_results/no_manifold.csv`
- `round2-qwen3-vl-8b-popular/variant_results/linear_probe.csv`
- `round2-internvl3.5-8b-popular/variant_results/full.csv`
- `round2-internvl3.5-8b-popular/variant_results/drift_only.csv`
- `round2-internvl3.5-8b-popular/variant_results/no_manifold.csv`
- `round2-internvl3.5-8b-popular/variant_results/linear_probe.csv`

Missing from both popular report directories:

- `baselines.json`
- `ablations.csv`
- `split_sensitivity.csv`
- output-side baseline result CSVs

### New-model popular caches

Under `outputs/round2_2026_04/cache/`:

- `llava-onevision-7b/pope/popular`: `24` shards, `shard-00000.pt` to `shard-00023.pt`
- `molmo-7b-d-0924/pope/popular`: `24` shards, `shard-00000.pt` to `shard-00023.pt`
- `llava-onevision-7b/pope-reference/train`: `40` shards, `shard-00000.pt` to `shard-00039.pt`
- `molmo-7b-d-0924/pope-reference/train`: `40` shards, `shard-00000.pt` to `shard-00039.pt`

Spot-check on the first shard for both models:

- cache schema uses `layer_vectors`
- `selected_layers` length is `14`
- hidden state shape is `(14, 3584)`
- first-token logit vectors are present
- `parsed_answer` is present for both eval and reference entries

### Downstream round-two artifacts

Not present yet outside smoke/debug paths:

- non-smoke `features/`
- non-smoke `readouts/`
- non-smoke `reference_banks/`
- non-smoke `plots/`

## Paper-facing mismatches

### Exporter

`scripts/export_paper_package.py` still assumes the old two-model correction-phase package.

- experiment names are hard-coded to `correction-*`
- only Qwen and InternVL appear in `MODEL_LABELS`
- table builders read `metrics.json` and old closeout report names
- no support for HALP or GLSim outputs
- no support for tracked round-two tables under `docs/tables/round2/`

### Results summary

`docs/results_summary.md` is partly outdated.

- it still says LLaVA-OneVision reference cache is pending
- it still says Molmo reference cache is pending
- those cache trees are now complete

## Execution Implication

The next valid step is to rebuild the paper-facing round-two artifacts from the current cache layer, starting with clean `POPE popular` reruns for Qwen and InternVL and full popular completion for LLaVA-OneVision and Molmo.
