# MIND v2 Design Notes

## Fixed Design Points

- Exploration pipeline.
- Sequential Stage A-E.
- Full-layer hidden states primary input for Stage A.
- Stage A is not started.
- Stage 0 completion requires both primary models across POPE popular/random/adversarial, RePOPE popular/random/adversarial, and DASH-B all.
- RePOPE and DASH-B are required, not optional.
- Layer sampling/16-layer controls deferred.
- Shared bank primary hypothesis with object/semantic branches future.
- No v2 dependency on v1 drift/manifold/wavelet.

## Working Shape

Stage 0 only prepares audited records, grouped splits, and full-layer caches. It is not complete until `configs/v2/stage0/stage0_complete.yaml` drives a full run for both models over POPE, RePOPE, and DASH-B. Stage A will start from those caches and explore the trajectory signal directly, but it has not started yet. Later stages can add tests and controls in order, but the v2 path should not import the v1 drift, manifold, or wavelet stack as a hidden dependency.
