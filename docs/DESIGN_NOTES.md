# MIND Design Notes

## Fixed Design Points

- Exploration pipeline.
- Sequential Stage A-E.
- Full-layer hidden states primary input for Stage A.
- Stage A tests representation hypotheses only.
- Stage A does not validate the final MIND detector.
- Stage 0 completion requires both primary models across POPE popular/random/adversarial, RePOPE popular/random/adversarial, and DASH-B all.
- RePOPE and DASH-B are required, not optional.
- Layer sampling/16-layer controls deferred.
- Shared bank primary hypothesis with object/semantic branches future.
- No dependency on the old drift/manifold/wavelet path.

## Working Shape

Stage 0 prepares audited records, grouped splits, and full-layer caches under `outputs/stage0`. Stage A starts from those caches, writes to `outputs/stageA`, and explores the trajectory signal directly. It may conclude that only multi-layer aggregation is useful while layer order remains unproven. Later stages can add tests and controls in order, but Stage B has not started.
