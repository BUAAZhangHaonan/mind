# Experiment 1 Curvature Analysis

## Answers

1. Angular-distance variants beat local PCA on 8/12 completed comparisons.
2. Angular-distance variants beat no_manifold on 11/12 completed comparisons.
3. Angular distance helps more on the harder benchmark group than on POPE popular.
4. Rank ordering by PR-AUC, then ROC-AUC:
- euclidean_pca_residual: mean rank 2.25 across 16 comparisons.
- centroid_angular: mean rank 2.83 across 12 comparisons.
- knn_angular_k10: mean rank 2.17 across 12 comparisons.
- centroid_euclidean: mean rank 2.33 across 12 comparisons.
5. The evidence supports more Riemannian geometry work.

## Missing Coverage

- internvl3.5-8b / DASH-B / centroid_angular: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / knn_angular_k10: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / centroid_euclidean: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / centroid_angular: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / knn_angular_k10: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / centroid_euclidean: missing_cache missing eval cache
- llava-onevision-7b / POPE adversarial / centroid_angular: missing_cache missing eval cache
- llava-onevision-7b / POPE adversarial / knn_angular_k10: missing_cache missing eval cache
- llava-onevision-7b / POPE adversarial / centroid_euclidean: missing_cache missing eval cache
- molmo-7b-d-0924 / POPE adversarial / centroid_angular: missing_cache missing eval cache
- molmo-7b-d-0924 / POPE adversarial / knn_angular_k10: missing_cache missing eval cache
- molmo-7b-d-0924 / POPE adversarial / centroid_euclidean: missing_cache missing eval cache
