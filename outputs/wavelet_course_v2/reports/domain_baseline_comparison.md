# Domain Baseline Comparison

这些结果使用小波课程 v2 的同一份 RePOPE readout cache。HALP 的两个协议分开报告：course-grouped 行使用课程 image_id grouped train/validation/test split；official-row 行使用 HALP 旧 row-stratified train/eval split。

- model: qwen3-vl-8b
- dataset: repope
- primary_population: 7986
- hard_hallucinations: 278
- train: 4795 samples, 192 positives
- validation: 1579 samples, 48 positives
- test: 1612 samples, 38 positives
- success_rows: 4
- failure_rows: 0

## Baselines

- `halp_official_mlp`: course-grouped HALP; validation selects probe and threshold; test reports metrics.
- `halp_official_row_protocol`: old official row protocol; eval selects probe and reports metrics; threshold=0.5.
- Linear probe: balanced logistic regression on final hidden state and mean-layer hidden state.

## Best Rows

- best_halp_official_mlp: halp_official_mlp PR-AUC=0.541768 F1=0.542056 AP=0.559306
- best_halp_official_row_protocol: halp_official_row_protocol PR-AUC=0.886468 F1=0.800000 AP=0.887175
- best_linear_probe: linear_probe_final_hidden_logreg PR-AUC=0.552712 F1=0.565217 AP=0.561763
- best_domain_overall: halp_official_row_protocol PR-AUC=0.886468 F1=0.800000 AP=0.887175

## Official HALP Status

- halp_official_mlp: protocol=course-grouped, selected_probe=query_token_layer_27, threshold=0.384702, selection_metric=validation_roc_auc_then_pr_auc, candidates=11, layer_indices=0,9,18,27,35, PR-AUC=0.541768
- halp_official_row_protocol: protocol=official-row, selected_probe=query_token_layer_35, threshold=0.500000, selection_metric=eval_roc_auc_then_pr_auc, candidates=11, layer_indices=0,9,18,27,35, PR-AUC=0.886468

## All Successful Rows

- halp_official / halp_official_row_protocol: PR-AUC=0.886468, F1=0.800000, ROC-AUC=0.988744
- linear_probe / linear_probe_final_hidden_logreg: PR-AUC=0.552712, F1=0.565217, ROC-AUC=0.959640
- halp_official / halp_official_mlp: PR-AUC=0.541768, F1=0.542056, ROC-AUC=0.982947
- linear_probe / linear_probe_mean_layer_hidden_logreg: PR-AUC=0.506175, F1=0.564706, ROC-AUC=0.958269
