# Experiment 2 Bank Identity Analysis

## Answers

1. No. object_conditioned beats shared on 4/6 completed full_curve comparisons. That is helpful in some settings, but it is not consistent.
2. No. object_conditioned beats shuffled_object on 2/6 completed full_curve comparisons. Wrong-label banks often match or beat the object bank, so the object label is not carrying a stable signal.
3. No. shared beats shuffled_object on 1/6 completed full_curve comparisons. Pooling grounded states is not enough to explain the wins either.
4. No. On DASH-B, full_curve beats no_manifold on 1/6 completed bank comparisons. This fails the acid test for the manifold story in the completed DASH-B coverage.
5. The completed controls do not support object-conditioned manifold geometry as the useful part. The object label is not reliable, shared pooling is not reliably better than shuffled labels, and DASH-B usually favors no_manifold over full_curve.

## Missing Coverage

- internvl3.5-8b / DASH-B / object_conditioned / no_manifold: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / shared / drift_only: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / shared / no_manifold: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / shared / full_curve: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / shuffled_object / drift_only: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / shuffled_object / no_manifold: missing_cache missing eval cache
- internvl3.5-8b / DASH-B / shuffled_object / full_curve: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / object_conditioned / no_manifold: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / shared / drift_only: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / shared / no_manifold: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / shared / full_curve: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / shuffled_object / drift_only: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / shuffled_object / no_manifold: missing_cache missing eval cache
- llava-onevision-7b / DASH-B / shuffled_object / full_curve: missing_cache missing eval cache
