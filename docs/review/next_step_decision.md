# Next Step Decision

## Decision

Conclusion B: curvature exists, but object conditioning has no reliable value in the current design.

This is not Conclusion A because the bank identity control fails. object_conditioned does not consistently beat shared, and it loses to shuffled_object on most completed full_curve comparisons. This is not Conclusion C because Experiment 1 still shows useful angular signal: angular variants beat no_manifold on 11/12 completed comparisons and beat local PCA on 8/12 completed comparisons.

## What The Data Says

1. Curvature is worth keeping in view. Angular distances are not a universal win, but they help often enough that the hidden-state geometry does not look purely linear.
2. Object labels are the weak link. Wrong-label banks often match or beat the object-conditioned bank, so the current object-conditioned manifold is not the useful unit.
3. DASH-B is the hard test. On completed DASH-B bank comparisons, full_curve beats no_manifold only 1/6 times, so the current MIND bank design does not justify a method paper by itself.

## Bank Change

Replace object-conditioned banks with reference selection based on the hidden states themselves.

The next bank should be a pooled grounded bank with query-local neighbor selection. It should choose references by activation proximity, angular proximity, or a learned retrieval score, then run the same detector on that local neighborhood. Object names can remain metadata for analysis, but they should not define the bank unless a future control shows that labels beat shared and shuffled banks.

The clean next variant is:

- Build one grounded reference pool per model and layer.
- Select query-local references by angular kNN or activation kNN.
- Compute angular distance and local residual features on that selected neighborhood.
- Keep the prompt-defined full-curve recipe fixed: raw 16-layer values plus calibrated mean, max, final, slope, and variance.
- Compare against no_manifold, shared, shuffled_object, and the existing local-PCA MIND rows with the same splits.

## Remaining Evidence To Collect

Regenerate or locate the missing DASH-B eval caches for InternVL3.5-8B and LLaVA-OneVision-7B, then rerun Experiment 2. The current conclusion is still strong enough to stop investing in object-conditioned banks, but those missing rows would make the DASH-B bank-control result complete across all four models.
