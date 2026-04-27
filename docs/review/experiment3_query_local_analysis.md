# Experiment 3 Query-Local Bank Analysis

## Answers

1. query_local_k30 beats object_cond on 7/8 PR-AUC comparisons.
2. query_local_k30 beats shared on 8/8 comparisons and shuffled on 5/8 comparisons.
3. On DASH-B, query_local_k30 beats no_manifold on 4/4 comparisons.
4. query_local_k30 is closer to linear_probe than object_cond on 7/8 PR-AUC comparisons.
5. Best method by model and benchmark:
- internvl3.5-8b / DASH-B: linear_probe (PR-AUC 0.9822, ROC-AUC 0.9920)
- internvl3.5-8b / POPE popular: linear_probe (PR-AUC 0.6785, ROC-AUC 0.9552)
- llava-onevision-7b / DASH-B: linear_probe (PR-AUC 0.9909, ROC-AUC 0.9943)
- llava-onevision-7b / POPE popular: linear_probe (PR-AUC 0.3240, ROC-AUC 0.9249)
- molmo-7b-d-0924 / DASH-B: linear_probe (PR-AUC 0.9635, ROC-AUC 0.9810)
- molmo-7b-d-0924 / POPE popular: linear_probe (PR-AUC 0.5946, ROC-AUC 0.9484)
- qwen3-vl-8b / DASH-B: linear_probe (PR-AUC 0.9880, ROC-AUC 0.9957)
- qwen3-vl-8b / POPE popular: linear_probe (PR-AUC 0.4552, ROC-AUC 0.9460)
6. Rank-order variation by model:
- internvl3.5-8b: 2 distinct PR-AUC rank orders across the two benchmarks.
- llava-onevision-7b: 2 distinct PR-AUC rank orders across the two benchmarks.
- molmo-7b-d-0924: 2 distinct PR-AUC rank orders across the two benchmarks.
- qwen3-vl-8b: 2 distinct PR-AUC rank orders across the two benchmarks.

Overall verdict: Query-local selection improves the geometry story, especially if the DASH-B wins are stable.
