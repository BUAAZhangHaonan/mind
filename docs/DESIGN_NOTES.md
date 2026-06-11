# MIND Design Notes

## Fixed Design Points

- Exploration pipeline.
- Sequential Stage A-E.
- Full-layer hidden states primary input for Stage A.
- Stage A tests representation hypotheses only and is closed by the Raw-Traj-LSTM full-panel closeout.
- Stage A does not validate the final MIND detector.
- The frozen theory note in `README.md` is the reference for later-stage method boundaries.
- Stage 0 completion requires both primary models across POPE popular/random/adversarial, RePOPE popular/random/adversarial, and DASH-B all.
- RePOPE and DASH-B are required, not optional.
- Layer sampling/16-layer controls deferred.
- Shared bank primary hypothesis with object/semantic branches future.
- No dependency on the old drift/manifold/wavelet path.

## Working Shape

Stage 0 prepares audited records and grouped splits. Experiment 2 builds the full-cache manifest under `outputs/full_cache`. Stage A closeout consumes that manifest, writes to `outputs/stageA_closeout`, and closes the representation pretest by comparing Raw-Traj-LSTM and Sphere-Traj-LSTM. Stage B1 consumes the same unified full-cache manifest and compares BCE, SupCon, and Proxy Anchor objectives on the frozen Sphere-Traj-LSTM representation. Stage B2 freezes Proxy Anchor and varies only the hard-hallucination negative budget. Stage B does not choose the final detector, and Stage C has not started.
