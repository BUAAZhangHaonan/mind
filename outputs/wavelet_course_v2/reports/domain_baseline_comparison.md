# Domain Baseline Comparison

这些结果使用小波课程 v2 的同一份 RePOPE primary population 和同一份 grouped split。

- model: qwen3-vl-8b
- dataset: repope
- primary_population: 7986
- hard_hallucinations: 278
- train: 4795 samples, 192 positives
- validation: 1579 samples, 48 positives
- test: 1612 samples, 38 positives
- success_rows: 3
- failure_rows: 0
- official_halp_cache: outputs/wavelet_course_v2/halp_cache/qwen3-vl-8b/repope/primary
- official_halp_policy: train on train split, choose probe and threshold on validation split, report test metrics.
- included_domain_methods: official HALP and linear probe only; MIND and HALP-like are not included.

## Baselines

- Official HALP: MLP probe over `vision_only`, `vision_token_layer_*`, and `query_token_layer_*` features.
- Linear probe: balanced logistic regression on final hidden state and mean-layer hidden state.

## Best Rows

- best_halp_official: halp_official_mlp PR-AUC=0.538175 F1=0.520000 AP=0.553644
- best_linear_probe: linear_probe_final_hidden_logreg PR-AUC=0.541455 F1=0.500000 AP=0.550401
- best_domain_overall: linear_probe_final_hidden_logreg PR-AUC=0.541455 F1=0.500000 AP=0.550401

## Official HALP Status

- success: selected_probe=query_token_layer_27, candidates=11, layer_indices=0,9,18,27,35

## All Successful Rows

- linear_probe / linear_probe_final_hidden_logreg: PR-AUC=0.541455, F1=0.500000, ROC-AUC=0.962834
- halp_official / halp_official_mlp: PR-AUC=0.538175, F1=0.520000, ROC-AUC=0.978483
- linear_probe / linear_probe_mean_layer_hidden_logreg: PR-AUC=0.503735, F1=0.607595, ROC-AUC=0.959189
