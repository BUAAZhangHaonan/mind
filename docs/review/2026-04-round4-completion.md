# Round-Four CPU Completion Note

Date: 2026-04-07

This note records what the CPU-only finalization pass completed and what still depends on future GPU time.

## Completed In This Session

| Item | Status | Key result | Notes |
| --- | --- | --- | --- |
| DASH-B main table | completed | the four `DASH-B` baseline report rows are now assembled into `docs/tables/round2/table1_dash_b.md` and `.csv` | built from saved round-two report dirs only |
| Feature ablation table | completed with one known gap | `docs/tables/round2/table2_feature_ablation.md` and `.csv` now exist | `POPE popular` feature-ablation rows for LLaVA and Molmo were not saved in the round-two tree, so those cells are marked `not run` |
| Findings summary note | completed | `docs/review/2026-04-round2-findings-summary.md` states the main story cleanly | full MIND beats simple output baselines, linear probe beats full MIND, `no_manifold` beats full MIND on `DASH-B` |
| Results summary | completed | `docs/results_summary.md` now reflects the current round-two main tables | stale pending-cache language removed |
| Paper outline refresh | completed | `docs/paper_outline.md` now matches the saved four-model two-benchmark pattern | the outline now says plainly that the manifold step weakens on `DASH-B` |
| Full test suite | completed | `158 passed, 14 warnings in 54.77s` | run in `mind-py311` after all document and table updates |

## Main Findings Locked By This Session

- The paper now has both completed main tables on disk:
  - `POPE popular`
  - `DASH-B`
- Full MIND beats the simple output baselines on nearly every completed row.
- The strongest completed internal baseline is the linear probe, not full MIND.
- On `DASH-B`, `no_manifold` beats full MIND on all four completed rows.

## Still GPU-Dependent

These planned items were not touched in this CPU-only pass and still need future GPU time:

- HALP readout extraction on `POPE popular`
- HALP readout extraction on `DASH-B`
- HALP runs
- GLSim readout extraction on `POPE popular`
- GLSim readout extraction on `DASH-B`
- GLSim runs
- POPE adversarial report generation
- RePOPE report generation
- transfer and control experiments
- bank-size ablation
- layer-count ablation

## Important Limits

- The tracked main tables are now current.
- The comparator columns are still empty because the shared readout and comparator work has not been completed.
- The feature ablation table is partially incomplete on `POPE popular` for LLaVA and Molmo because those ablation rows were never saved in the round-two artifact tree.
