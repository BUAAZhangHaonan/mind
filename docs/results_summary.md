# Results Summary

## Geometry Value Experiments, 2026-04-27

The new results point to Conclusion B: curvature has signal, but object-conditioned banks are not the stable source of value.

Experiment 1 tested local-PCA residuals against angular and centroid distances across four models and four benchmarks. Angular-distance variants beat local PCA on 8/12 completed comparisons and beat the no_manifold baseline on 11/12 completed comparisons. The strongest angular result was kNN angular distance, with a mean rank of 2.17 across its 12 completed comparisons. On the same 12 comparisons, local PCA residual has mean rank 2.67. Across all 16 model-benchmark comparisons, local PCA residual has mean rank 2.25. The missing rows are all explicit `missing_cache` rows, mainly from DASH-B or POPE adversarial caches that were not on disk.

Experiment 2 tested whether object identity matters in the reference bank. It does not hold up. object_conditioned beats shared on 4/6 completed full_curve comparisons, but it beats shuffled_object on only 2/6. shared beats shuffled_object on only 1/6. On DASH-B, full_curve beats no_manifold on 1/6 completed bank comparisons. This fails the main test for the object-conditioned manifold story.

The useful signal is not well explained by the current object-label bank. The next design should separate two questions: whether angular geometry helps, and how the reference bank should be chosen. The object name should not be treated as the primary bank identity until it beats both shared and shuffled banks consistently.

Primary outputs:

- `docs/tables/experiment_curvature_verification.md`
- `docs/tables/experiment_curvature_verification.csv`
- `docs/review/experiment1_curvature_analysis.md`
- `docs/tables/experiment_bank_identity.md`
- `docs/tables/experiment_bank_identity.csv`
- `docs/review/experiment2_bank_analysis.md`
