# Results Summary

## Geometry Value Experiments — GPU Round (2026-04)

The GPU round changes the practical decision. Curvature still looks useful, object-conditioned banks still do not look reliable, and query-local kNN banks are now the best geometric path to test further.

Phase 1 filled the DASH-B cache gap for InternVL3.5-8B and LLaVA-OneVision-7B, then verified full DASH-B cache coverage for all four models. Phase 2 reran the bank identity control with all 72 rows present. Phase 3 tested the new query-local kNN bank on all four models and both POPE popular and DASH-B.

Experiment 1 still supports the curvature premise. Across the 12 complete model-benchmark comparisons where all four distance curves were available, `knn_angular_k10` has the best mean PR-AUC rank and the most PR wins. The saved full MIND row remains strong, but the distance-only comparison says angular neighborhoods carry signal that Euclidean local PCA can miss.

| distance type | comparisons | mean PR rank | PR wins | mean PR-AUC | mean ROC rank | ROC wins | mean ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| knn_angular_k10 | 12 | 2.17 | 6 | 0.3457 | 2.17 | 5 | 0.8781 |
| centroid_euclidean | 12 | 2.33 | 1 | 0.3565 | 2.17 | 4 | 0.8717 |
| euclidean_pca_residual | 12 | 2.67 | 4 | 0.3324 | 2.83 | 3 | 0.8606 |
| centroid_angular | 12 | 2.83 | 1 | 0.3540 | 2.83 | 0 | 0.8688 |

Experiment 2 still says object identity is not the stable source of value. In the complete v2 table, object_conditioned beats shared on 7/8 full_curve PR-AUC comparisons, but it beats shuffled_object on only 4/8. On DASH-B, full_curve beats no_manifold on only 4/12 bank comparisons. That is better coverage than the CPU round, but it does not justify treating the object label as the bank identity.

Experiment 3 is the decisive new result. `query_local_k30` beats object_cond on 7/8 PR-AUC comparisons, shared on 8/8, shuffled on 5/8, and no_manifold on 8/8. On DASH-B, it beats no_manifold on 4/4 models, with mean PR-AUC 0.9369 versus 0.7237 for no_manifold. It also gets much closer to the linear_probe ceiling, though linear_probe still wins every comparison.

| bank/method | comparisons | mean PR rank | PR wins | mean PR-AUC | mean ROC rank | ROC wins | mean ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_probe | 8 | 1.00 | 8 | 0.7471 | 1.00 | 8 | 0.9672 |
| query_local_k30 | 8 | 2.50 | 0 | 0.6345 | 2.25 | 0 | 0.9386 |
| object_cond | 8 | 3.88 | 0 | 0.4850 | 4.00 | 0 | 0.8604 |
| shuffled | 8 | 3.88 | 0 | 0.4805 | 4.25 | 0 | 0.8304 |
| no_manifold | 8 | 4.62 | 0 | 0.4525 | 4.50 | 0 | 0.8398 |
| shared | 8 | 5.12 | 0 | 0.4377 | 5.00 | 0 | 0.8431 |

The updated conclusion is a stronger version of Conclusion B: curvature exists, object-conditioned banks are the wrong design, and query-local selection is worth one more focused method round. The next round should test whether query-local geometry remains strong when k, retrieval metric, reference pool, and benchmark are varied.

Primary GPU-round outputs:

- `docs/review/phase1_dash_b_cache_coverage.md`
- `docs/tables/experiment_bank_identity_v2.md`
- `docs/tables/experiment_bank_identity_v2.csv`
- `docs/review/experiment2_bank_analysis_v2.md`
- `docs/tables/experiment_query_local_bank.md`
- `docs/tables/experiment_query_local_bank.csv`
- `docs/review/experiment3_query_local_analysis.md`
- `docs/review/next_step_decision.md`

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
