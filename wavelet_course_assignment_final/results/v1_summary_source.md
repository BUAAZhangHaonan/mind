# Wavelet Course Summary

- model: qwen3-vl-8b
- dataset: repope
- quick_run: false
- ours_token_id_source: local_tokenizer_json
- cache_accepted: true
- primary_population: 7986
- hard_hallucinations: 278
- metrics_csv: outputs/wavelet_course/reports/metrics.csv
- best_configs_csv: outputs/wavelet_course/reports/best_configs.csv

## Experiment Completion

- full_run: true
- quick_run: false
- cache_accepted: true
- metrics_rows: 13
- success_count: 9
- failure_count: 4

## Configuration Details

### Teacher-Bagua
- teacher_bagua_haar_l1_lstm: wavelet=haar level=1 threshold=universal_soft
- teacher_bagua_db2_l1_lstm: wavelet=db2 level=1 threshold=universal_soft
- teacher_bagua_db4_l1_lstm: wavelet=db4 level=1 threshold=universal_soft
- LSTM hidden_dim=64 epochs=10 batch_size=16 lr=0.001 patience=3 input_shape=9x114688

### Ours-Wavelet
- ours_db2_swt_l2: transform=swt wavelet=db2 SWT level=2
- ours_db2_swt_l3: transform=swt wavelet=db2 SWT level=3
- ours_sym4_swt_l2: transform=swt wavelet=sym4 SWT level=2
- classifier_variants=logreg,xgb
- trace_list=norm_trace, delta_norm_trace, cos_prev_trace, cos_final_trace, yes_no_margin_trace, yes_no_entropy_trace
- final_broadcast=yes token_source=local_tokenizer_json

### HALP-like/MIND Baselines
- final_hidden_logreg: feature=final-layer hidden vector classifier=logreg
- mean_layer_hidden_logreg: feature=mean-pooled hidden vector across layers classifier=logreg
- norm_traj_logreg: feature=36-point hidden-norm trajectory classifier=logreg
- sphere_traj_meanpool_logreg: feature=mean-pooled unit-sphere layer trajectory classifier=logreg
- logreg=max_iter=5000 class_weight=balanced

## Full Metrics

| family | config | status | PR-AUC | F1 | feature_seconds | train_eval_seconds | total_seconds | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher_bagua | teacher_bagua_db2_l1_lstm | success | 0.023663235446484562 | 0.03986710963455149 | 690.3514017178677 | 243.58495989697985 | 933.9383473568596 |  |
| teacher_bagua | teacher_bagua_db4_l1_lstm | success | 0.02216748943992889 | 0.035856573705179286 | 742.1002384559251 | 191.91824095905758 | 934.0206553239841 |  |
| teacher_bagua | teacher_bagua_haar_l1_lstm | success | 0.02911711167836548 | 0.04977079240340537 | 715.9063899221364 | 224.34840798890218 | 940.2566479649395 |  |
| ours_wavelet | ours_db2_swt_l2_logreg | success | 0.39757361369212124 | 0.4634146341463415 | 57.40194187615998 | 3.217387185897678 | 60.61934060114436 |  |
| ours_wavelet | ours_db2_swt_l2_xgb | failure |  |  | 35.76483379304409 | 0.005018276162445545 | 35.76985559798777 | xgboost_not_installed |
| ours_wavelet | ours_db2_swt_l3_logreg | failure |  |  | 0.7162761420477182 |  | 0.7162801758386195 | wavelet_config_error: SWT level 3 is not feasible for trace length 36; max level is 2 |
| ours_wavelet | ours_db2_swt_l3_xgb | failure |  |  | 0.8042269188445061 |  | 0.8042316071223468 | wavelet_config_error: SWT level 3 is not feasible for trace length 36; max level is 2 |
| ours_wavelet | ours_sym4_swt_l2_logreg | success | 0.312771975990232 | 0.3956043956043956 | 38.62033454491757 | 3.8349312499631196 | 42.45526911015622 |  |
| ours_wavelet | ours_sym4_swt_l2_xgb | failure |  |  | 35.034572649979964 | 0.0031019761227071285 | 35.037680325098336 | xgboost_not_installed |
| mind_baseline | final_hidden_logreg | success | 0.5414553636477906 | 0.5 | 5.695526979863644 | 4.963882399024442 | 10.65941341011785 |  |
| mind_baseline | mean_layer_hidden_logreg | success | 0.5037352006418447 | 0.6075949367088608 | 3.3697746118996292 | 3.1834637520369142 | 6.553243682021275 |  |
| mind_baseline | norm_traj_logreg | success | 0.33741134949148316 | 0.3963963963963964 | 3.606419188901782 | 0.07068051910027862 | 3.6771034840494394 |  |
| mind_baseline | sphere_traj_meanpool_logreg | success | 0.4141539998979404 | 0.44155844155844154 | 4.776504378998652 | 2.9593938048928976 | 7.735901952954009 |  |

## Course Narrative

Traditional industrial fault detection templates assume two things. They assume the signal is a fixed sensor trace with stable axes, and they assume wavelet scales map cleanly onto local time-frequency events. VLM hidden dimensions do not satisfy those assumptions, because one hidden coordinate is not a stable physical channel and its order is not a sensor axis.

The 36 transformer layers are the ordered computation depth here. Teacher-Bagua is intentionally strict transfer: it asks whether a standard temporal readout trained on layer trajectories can move into this hallucination setting without changing the problem to fit it.

Ours puts the wavelet transform on layer-wise semantic traces. That keeps the ordered axis tied to computation depth instead of pretending hidden dimensions are physical sensors.

## Results

- best_teacher_bagua: teacher_bagua_haar_l1_lstm PR-AUC=0.02911711167836548 F1=0.04977079240340537
- best_ours_wavelet: ours_db2_swt_l2_logreg PR-AUC=0.39757361369212124 F1=0.4634146341463415
- best_baseline: final_hidden_logreg PR-AUC=0.5414553636477906 F1=0.5
- overall_best: final_hidden_logreg PR-AUC=0.5414553636477906 F1=0.5

## Timing

- mind_baseline: configs=4 feature_seconds=17.448225 train_eval_seconds=11.177420 total_seconds=28.625663
- ours_wavelet: configs=6 feature_seconds=168.342186 train_eval_seconds=7.060439 total_seconds=175.402657
- teacher_bagua: configs=3 feature_seconds=2148.358030 train_eval_seconds=659.851609 total_seconds=2808.215651

- Teacher-Bagua total_seconds=2808.215651 avg_total_seconds=936.071884
- Ours-Wavelet total_seconds=175.402657 avg_total_seconds=29.233776

## Failures

- ours_db2_swt_l3_logreg: wavelet_config_error: SWT level 3 is not feasible for trace length 36; max level is 2
- ours_db2_swt_l2_xgb: xgboost_not_installed
- ours_db2_swt_l3_xgb: wavelet_config_error: SWT level 3 is not feasible for trace length 36; max level is 2
- ours_sym4_swt_l2_xgb: xgboost_not_installed

## Conclusion

The best observed test PR-AUC is from final_hidden_logreg. This result should be read as a diagnostic comparison, not as proof that hidden dimensions form sensor-like wavelet channels. 4 configurations failed and remain listed in metrics.csv.
