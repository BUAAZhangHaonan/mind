# Subtask 1 Neighbor Selection Analysis

## Decision Rule

Select the method with the best mean PR-AUC across the 8 model-benchmark settings. Use PR-AUC as the primary metric because Sub-task 2 depends on positive-class retrieval quality. Use wins across the 8 settings as the consistency check, and use ROC-AUC only as a secondary check.

## Conclusion

Use `radius_ball` for Sub-task 2. It has the best mean PR-AUC, 0.5708, and wins 6/8 settings. The two PR-AUC exceptions are Qwen DASH-B, where `kernel_knn_k30` wins, and Molmo POPE, where `kernel_knn_k30` wins.

## Decision Questions

### Does any method consistently beat `knn_angular_k30` on PR-AUC across both benchmarks?

Yes. `radius_ball` is the only method that clearly and consistently beats `knn_angular_k30`. Its mean PR-AUC is 0.5708 versus about 0.5262 for `knn_angular_k30`, and it wins 6/8 settings.

### Does kernel weighting help?

Only a little. `kernel_knn_k30` has mean PR-AUC 0.5271, which is slightly above angular and cosine kNN at about 0.5262. It wins Qwen DASH-B and Molmo POPE, but it does not match `radius_ball` overall.

### Does radius-ball help?

Yes. `radius_ball` helps the most. It has mean PR-AUC 0.5708 and wins 6/8 settings, with the main exceptions being Qwen DASH-B and Molmo POPE.

### Does cosine vs angular make any difference?

No meaningful difference appears. `knn_cosine_k30` and `knn_angular_k30` both have mean PR-AUC around 0.5262, and their per-setting values are effectively the same.

### Does raw Euclidean kNN perform worse?

Yes, but only slightly. `knn_euclidean_k30` has mean PR-AUC 0.5250, below angular and cosine at about 0.5262 and below kernel weighting at 0.5271.

### What is the recommended neighbor selection method for Sub-task 2?

Use `radius_ball`. It is the strongest choice by the primary decision rule: highest mean PR-AUC, 0.5708, and best consistency, with 6/8 PR-AUC wins.
