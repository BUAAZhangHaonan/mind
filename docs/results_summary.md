# MIND Results Summary

## Current Status

The repo now has one clean paper decision from the live round-two rerun:

- keep the paper on object hallucination only
- drop Haar from the default method
- use `raw + calibrated simple stats` as the default full MIND feature set
- treat the older March popular numbers as stale

The round-two source of truth is the current code path plus the new rerun artifacts under `outputs/round2_2026_04/`.

## What Changed

The old closeout package is no longer enough for the paper.

- The older correction-phase popular tables do not reproduce under the current evaluation path.
- The current `train_detector.py` and current `compute_baselines.py` agree with each other.
- So the live round-two rerun is the paper-facing result, and the older March package is now audit-only.

## Phase-One Decision

Tracked table:

- `docs/tables/round2/phase_one_popular_decision.md`

Decision:

- default full MIND:
  - `raw + calibrated simple stats`
- reason:
  - Haar is not the best current variant on either model family
  - simple stats beats Haar on both current popular reruns
  - the confidence intervals overlap, so the simpler choice is the right one

Paper implication:

- remove `Multi-scale` from the title
- remove Haar from the default method description
- keep Haar only as an ablation

## Live Popular Rerun

These are the current `POPE popular`, `image_grouped` rerun values on the live code path.

### Qwen3-VL-8B

| Variant | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| raw only | 0.8462 | 0.1159 |
| raw + simple stats | 0.8908 | 0.1741 |
| raw + full curve | 0.9145 | 0.2596 |
| raw + Haar | 0.8690 | 0.1470 |
| logit margin | 0.5955 | 0.0422 |

Readout:

- simple stats is clearly better than Haar
- full curve is strongest, but not cleanly enough to justify the larger default feature set
- logit margin is far below MIND

### InternVL3.5-8B

| Variant | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| raw only | 0.8764 | 0.4284 |
| raw + simple stats | 0.8978 | 0.5092 |
| raw + full curve | 0.9119 | 0.5333 |
| raw + Haar | 0.8929 | 0.4854 |
| logit margin | 0.5454 | 0.0861 |

Readout:

- simple stats again beats Haar
- full curve again looks strongest, but the gap over simple stats is not clean enough to make it the default
- logit margin is again far below MIND

## Framing Decision

The logit-margin truth check came out cleanly in MIND’s favor.

- The paper does not need to retreat to “geometry only because confidence already wins.”
- The safer paper line is:
  - MIND clearly beats simple output-confidence baselines
  - MIND still has to face richer baselines such as a linear probe, HALP, and GLSim

## What Is Still Pending

Main-table reruns still in progress or not started yet:

- Qwen and InternVL:
  - full current-code reruns for `POPE adversarial`
  - full current-code reruns for `RePOPE`
  - full current-code reruns for `DASH-B`
- LLaVA-OneVision:
  - reference cache
  - popular full pipeline
  - adversarial full pipeline
  - DASH-B full pipeline
- Molmo:
  - reference cache
  - popular full pipeline
  - adversarial full pipeline
  - DASH-B full pipeline
- comparators:
  - HALP
  - GLSim

## What Not To Reuse

Do not reuse these as current paper numbers:

- the March correction-phase popular headline table
- the older paper title with `Multi-scale`
- any argument that Haar is scientifically central

Those artifacts still matter for debugging, but they are no longer the paper-facing result set.
