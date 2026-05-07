# MIND v2 Design Notes

## Fixed Design Points

- Exploration pipeline.
- Sequential Stage A-E.
- Full-layer hidden states primary input for Stage A.
- Layer sampling/16-layer controls deferred.
- Shared bank primary hypothesis with object/semantic branches future.
- No v2 dependency on v1 drift/manifold/wavelet.

## Working Shape

Stage 0 only prepares audited records, grouped splits, and full-layer caches. Stage A starts from those caches and explores the trajectory signal directly. Later stages can add tests and controls in order, but the v2 path should not import the v1 drift, manifold, or wavelet stack as a hidden dependency.
