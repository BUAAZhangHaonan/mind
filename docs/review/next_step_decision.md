# Next Step Decision

## Decision

Updated Conclusion B: curvature exists, object conditioning is not reliable, and query-local banks are the right replacement to test next.

This is not Conclusion A because object-conditioned banks still do not beat shuffled banks consistently. This is not Conclusion C because query-local kNN geometry beats no_manifold on every DASH-B model and on 8/8 total PR-AUC comparisons.

## What Changed In The GPU Round

1. The DASH-B coverage gap is closed. InternVL3.5-8B and LLaVA-OneVision-7B now have DASH-B eval caches, and the v2 bank table has all 72 rows.
2. The complete bank identity control still weakens the old object-conditioned story. object_conditioned beats shared on 7/8 full_curve comparisons, but only beats shuffled_object on 4/8.
3. Query-local selection changes the path forward. `query_local_k30` beats object_cond on 7/8 PR-AUC comparisons, shared on 8/8, shuffled on 5/8, and no_manifold on 8/8.

## Current Verdict

The query-local bank rescues the geometric approach enough to keep working on it. It does not rescue object-conditioned MIND.

The useful idea is not "one object label defines one manifold." The useful idea is closer to this: each query should choose a local grounded neighborhood in hidden-state space, then the detector should measure how the query sits relative to that neighborhood.

The project should continue as a method paper only if the next round shows that this query-local result is stable. The method paper should be about query-local grounded neighborhoods, not object-conditioned banks.

## Best Evidence

- Experiment 1: `knn_angular_k10` has the best mean PR-AUC rank among the four distance curves on the 12 complete comparisons.
- Experiment 2 v2: object identity remains mixed; full_curve beats no_manifold on only 4/12 DASH-B bank comparisons.
- Experiment 3: query-local kNN beats no_manifold on all four DASH-B models. Mean DASH-B PR-AUC is 0.9369 for query_local_k30, 0.7237 for no_manifold, and 0.9812 for linear_probe.

## Exact Next Experiment

Run a query-local robustness round.

- Sweep `k` over 10, 30, 50, and 100.
- Compare angular kNN, Euclidean kNN, and hidden-state cosine kNN.
- Ablate query-local features: centroid angular only, local PCA residual only, neighbor mean/std only, and all features.
- Extend from POPE popular and DASH-B to POPE adversarial and RePOPE.
- Keep the same image_grouped split where image identity can leak, and use row splits only where that is the existing benchmark protocol.
- Compare against no_manifold, shared, shuffled_object, object_cond, and linear_probe on the same splits.

Expected timeline: one GPU day for the full sweep if cached hidden states remain usable. The result that would change the decision is clear: if query-local geometry fails to beat no_manifold on at least 3/4 hard benchmark-model groups, or if it only works for one architecture, then the method-paper direction should stop.

## If It Fails

Pivot to compression or cross-architecture analysis.

Reusable code:

- cache loaders
- GPU distance functions
- GPU detector training
- bootstrap CI reporting
- table/report generators

New code needed:

- representation compression probes
- cross-architecture alignment metrics
- model-family comparison reports

## If It Holds

Build the method around dynamic local grounded neighborhoods.

The next paper version should frame object names as metadata, not bank keys. It should show that query-local geometry closes much of the gap to linear_probe while using a smaller and more interpretable feature set.
