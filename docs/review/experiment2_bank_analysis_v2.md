# Experiment 2 Bank Identity Analysis v2

## Answers

1. object_conditioned beats shared on 7/8 full_curve comparisons. This is the direct test of object-specific geometry.
2. object_conditioned beats shuffled_object on 4/8 full_curve comparisons. This checks whether the object label itself matters.
3. shared beats shuffled_object on 2/8 full_curve comparisons. This checks whether pooled grounded states help more than wrong labels.
4. On DASH-B, full_curve beats no_manifold on 4/12 bank comparisons. This is the acid test for the manifold story.
5. The complete v2 table does not rescue object-conditioned banks. The useful signal is still not tied reliably to the object label.

## Best Full-Curve Bank by Model and Benchmark

- internvl3.5-8b / DASH-B: object_conditioned (PR-AUC 0.7126, ROC-AUC 0.8586)
- internvl3.5-8b / POPE popular: object_conditioned (PR-AUC 0.5072, ROC-AUC 0.8966)
- llava-onevision-7b / DASH-B: object_conditioned (PR-AUC 0.7260, ROC-AUC 0.8423)
- llava-onevision-7b / POPE popular: shuffled_object (PR-AUC 0.2079, ROC-AUC 0.8647)
- molmo-7b-d-0924 / DASH-B: object_conditioned (PR-AUC 0.5496, ROC-AUC 0.7812)
- molmo-7b-d-0924 / POPE popular: shuffled_object (PR-AUC 0.3876, ROC-AUC 0.8932)
- qwen3-vl-8b / DASH-B: shuffled_object (PR-AUC 0.7344, ROC-AUC 0.9086)
- qwen3-vl-8b / POPE popular: shuffled_object (PR-AUC 0.3601, ROC-AUC 0.9034)

## CPU-vs-GPU ROC-AUC Discrepancies

Compared against `docs/tables/experiment_bank_identity.csv`.

- None under the >0.02 ROC-AUC and non-overlapping-CI rule.
