# Paired Wavelet V2 Summary

## Experiment Overview

- narrative: v2 paired extension; v1 not overwritten.
- extension: paired Teacher/Ours wavelet-course v2
- v1_preservation: v1_wavelet_population
- expected_sources: Teacher, Ours
- description: Paired Teacher/Ours wavelet-course v2 experiment on one fixed RePOPE primary population.
- outputs: v2 paired reports are written separately from the v1 output root.

## Task and Population

- task: model_name=qwen3-vl-8b dataset_name=repope subsets=popular,random,adversarial
- cache_accepted: true
- cache_entries: 8185
- task_subset_counts: adversarial=2684, popular=2727, random=2774
- primary_population: 7986
- hard_hallucinations: 278
- correct: 7708
- split_source: constructed:split_manifest_not_train_validation_test
- split_train: pos=192 neg=4603
- split_validation: pos=48 neg=1531
- split_test: pos=38 neg=1574
- paired_grid_path: outputs/wavelet_course_v2/audit/paired_grid.json
- paired_grid_rows: 94
- paired_grid_pair_ids: 47
- paired_grid_blocks: A, B, C, D, E
- sample_grid_path: outputs/wavelet_course_v2/audit/selected_sample_grid.csv
- sample_grid_rows: 7986
- sample_grid_row_order_hash: c9075b79b2c9daab3fa0acae1d4977495ef33c0acfac4ba778984f5be2cad0ca
- metrics_ledger.csv: outputs/wavelet_course_v2/reports/metrics_ledger.csv

## Why Paired Comparison

- comparison: paired Teacher/Ours rows
- long_rows: 114
- paired_rows: 57
- sources: Teacher, Ours
- completeness: every pair key has one Teacher row and one Ours row
- failed rows: preserved in metrics_long.csv, metrics_wide_paired.csv, failure_report.csv, and this summary

## Method Definitions

- Teacher: hidden-dimension signal rows from the teacher-side wavelet comparison.
- Ours: semantic trace rows from the layer-ordered wavelet summary comparison.
- Both sources share the same block, pair_id, classifier, split, seed, model, and dataset before they are compared.

## Wavelet Selection Rationale

- wavelet rationale: hidden dimensions are unordered coordinates, but layers have a stable computation order.
- Hidden dimension axis: a hidden coordinate order is not a physical sensor axis, so it should not be read as stable time or frequency.
- Layer axis: layer order is meaningful computation depth, so Ours applies wavelet summaries to semantic traces over layers.
- Paired grid: Teacher and Ours are compared only when they share the same block, pair_id, classifier, and run context.

## Paired Results

- metrics_long_rows: 114
- metrics_wide_paired_rows: 57
- success_rows: 94
- non_success_rows: 20
- paired_completeness: passed

### Output Files

- metrics_ledger.csv: outputs/wavelet_course_v2/reports/metrics_ledger.csv
- metrics_long.csv: outputs/wavelet_course_v2/reports/metrics_long.csv
- metrics_wide_paired.csv: outputs/wavelet_course_v2/reports/metrics_wide_paired.csv
- best_by_block.csv: outputs/wavelet_course_v2/reports/best_by_block.csv
- pairwise_winrate.csv: outputs/wavelet_course_v2/reports/pairwise_winrate.csv
- failure_report.csv: outputs/wavelet_course_v2/reports/failure_report.csv
- summary.md: outputs/wavelet_course_v2/reports/summary.md

### Paired Metric Rows

| block | pair_id | classifier | teacher_status | ours_status | teacher_value | ours_value | delta | winner | paired_failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A_cwt_mexh_scales_1_16 | logreg | success | success | 0.22611928537853082 | 0.289580168940283 | 0.06346088356175217 | Ours |  |
| A | A_cwt_morl_scales_1_16 | logreg | success | success | 0.28188976971523166 | 0.36729456520778553 | 0.08540479549255386 | Ours |  |
| A | A_dwt_bior2.2_l2 | logreg | success | success | 0.2873950618914342 | 0.3284441386837717 | 0.04104907679233749 | Ours |  |
| A | A_dwt_bior4.4_l2 | logreg | success | success | 0.2858731937392934 | 0.2912242671723846 | 0.0053510734330912135 | Ours |  |
| A | A_dwt_coif1_l2 | logreg | success | success | 0.2522108751125052 | 0.2285523521657024 | -0.023658522946802762 | Teacher |  |
| A | A_dwt_coif2_l2 | logreg | failure | failure |  |  |  |  | Teacher: DWT level 2 is not feasible for length 36 and wavelet 'coif2'; max level is 1; Ours: DWT level 2 is not feasible for length 36 and wavelet 'coif2'; max level is 1 |
| A | A_dwt_db2_l2 | logreg | success | success | 0.27801640913824 | 0.27586818965765014 | -0.0021482194805898525 | Teacher |  |
| A | A_dwt_db4_l2 | logreg | success | success | 0.19900546042404535 | 0.28939700486588793 | 0.09039154444184258 | Ours |  |
| A | A_dwt_db6_l2 | logreg | failure | failure |  |  |  |  | Teacher: DWT level 2 is not feasible for length 36 and wavelet 'db6'; max level is 1; Ours: DWT level 2 is not feasible for length 36 and wavelet 'db6'; max level is 1 |
| A | A_dwt_haar_l1 | logreg | success | success | 0.3343796372871526 | 0.29471780471033315 | -0.039661832576819434 | Teacher |  |
| A | A_dwt_sym2_l2 | logreg | success | success | 0.27717703452180775 | 0.28415846735783823 | 0.006981432836030477 | Ours |  |
| A | A_dwt_sym4_l2 | logreg | success | success | 0.20836796434175425 | 0.32965014033410095 | 0.1212821759923467 | Ours |  |
| A | A_dwt_sym6_l2 | logreg | failure | failure |  |  |  |  | Teacher: DWT level 2 is not feasible for length 36 and wavelet 'sym6'; max level is 1; Ours: DWT level 2 is not feasible for length 36 and wavelet 'sym6'; max level is 1 |
| A | A_none | logreg | success | success | 0.2114638845378483 | 0.23491627462307155 | 0.023452390085223235 | Ours |  |
| A | A_swt_db2_l2 | logreg | success | success | 0.17404798072585237 | 0.2963076129189292 | 0.12225963219307684 | Ours |  |
| A | A_swt_haar_l2 | logreg | success | success | 0.3072043155369921 | 0.25565460927669426 | -0.05154970626029787 | Teacher |  |
| A | A_swt_sym4_l2 | logreg | success | success | 0.17622722180053743 | 0.33050401909128346 | 0.15427679729074603 | Ours |  |
| A | A_wpt_db2_l2 | logreg | success | success | 0.33296220372753726 | 0.3617494642777053 | 0.02878726055016806 | Ours |  |
| A | A_wpt_sym4_l2 | logreg | success | success | 0.23202631246935382 | 0.3649087303287217 | 0.13288241785936786 | Ours |  |
| B | B_direct_raw_sequence__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | success | success | 0.5997333988677356 | 0.4104745767152178 | -0.18925882215251777 | Teacher |  |
| B | B_direct_raw_sequence__lstm_projected_lr0p001 | lstm_projected_lr0p001 | success | success | 0.5368614511918353 | 0.375090047681559 | -0.16177140351027625 | Teacher |  |
| B | B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | success | success | 0.5487612982437247 | 0.21035032248502128 | -0.3384109757587034 | Teacher |  |
| B | B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | lstm_projected_lr0p001 | success | success | 0.6685401150358344 | 0.18288447937884317 | -0.48565563565699127 | Teacher |  |
| B | B_win12_s6_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | success | success | 0.48769811443391764 | 0.2246381710891663 | -0.26305994334475136 | Teacher |  |
| B | B_win12_s6_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | lstm_projected_lr0p001 | success | success | 0.48717142002492636 | 0.27087779804831613 | -0.21629362197661023 | Teacher |  |
| B | B_win4_s4_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | success | success | 0.43720530699913285 | 0.12993841727361174 | -0.3072668897255211 | Teacher |  |
| B | B_win4_s4_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | lstm_projected_lr0p001 | success | success | 0.3824895508512774 | 0.14125108248547327 | -0.24123846836580412 | Teacher |  |
| B | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | failure | failure |  |  |  |  | Teacher: SWT level 2 is not feasible for length 6; max level is 1; Ours: SWT level 2 is not feasible for length 6; max level is 1 |
| B | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | lstm_projected_lr0p001 | failure | failure |  |  |  |  | Teacher: SWT level 2 is not feasible for length 6; max level is 1; Ours: SWT level 2 is not feasible for length 6; max level is 1 |
| B | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | failure | failure |  |  |  |  | Teacher: SWT level 2 is not feasible for length 9; max level is 0; Ours: SWT level 2 is not feasible for length 9; max level is 0 |
| B | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | lstm_projected_lr0p001 | failure | failure |  |  |  |  | Teacher: SWT level 2 is not feasible for length 9; max level is 0; Ours: SWT level 2 is not feasible for length 9; max level is 0 |
| C | C_raw_sequence_cnn1d__cnn1d_lr0p0003 | cnn1d_lr0p0003 | success | success | 0.4738000852437315 | 0.45173392419209946 | -0.02206616105163206 | Teacher |  |
| C | C_raw_sequence_cnn1d__cnn1d_lr0p001 | cnn1d_lr0p001 | success | success | 0.46981525626302284 | 0.35835710063013904 | -0.1114581556328838 | Teacher |  |
| C | C_raw_sequence_gru_projected__gru_projected_lr0p0003 | gru_projected_lr0p0003 | success | success | 0.6038209099200795 | 0.3331138514335141 | -0.27070705848656546 | Teacher |  |
| C | C_raw_sequence_gru_projected__gru_projected_lr0p001 | gru_projected_lr0p001 | success | success | 0.570503203969436 | 0.45292891179770706 | -0.11757429217172893 | Teacher |  |
| C | C_raw_sequence_lstm_projected__lstm_projected_lr0p0003 | lstm_projected_lr0p0003 | success | success | 0.5694338580805832 | 0.37967020927879674 | -0.18976364880178648 | Teacher |  |
| C | C_raw_sequence_lstm_projected__lstm_projected_lr0p001 | lstm_projected_lr0p001 | success | success | 0.5454295010198779 | 0.40176651212977205 | -0.1436629888901058 | Teacher |  |
| C | C_raw_sequence_tcn__tcn_lr0p0003 | tcn_lr0p0003 | success | success | 0.5303392004043577 | 0.39827119467173605 | -0.13206800573262162 | Teacher |  |
| C | C_raw_sequence_tcn__tcn_lr0p001 | tcn_lr0p001 | success | success | 0.5289734696285864 | 0.3638582924120133 | -0.1651151772165731 | Teacher |  |
| C | C_wavelet_summary_static_pooled_extra_trees | extra_trees | success | success | 0.1805621967936478 | 0.34972295836760803 | 0.16916076157396023 | Ours |  |
| C | C_wavelet_summary_static_pooled_linear_svm | linear_svm | success | success | 0.1757884485064049 | 0.2515280926551907 | 0.07573964414878581 | Ours |  |
| C | C_wavelet_summary_static_pooled_logreg | logreg | success | success | 0.17404798072585237 | 0.2963076129189292 | 0.12225963219307684 | Ours |  |
| C | C_wavelet_summary_static_pooled_rf | rf | success | success | 0.1745783123213036 | 0.3891652735813241 | 0.21458696126002047 | Ours |  |
| C | C_wavelet_summary_static_pooled_xgboost | xgboost | success | success | 0.15973239959837646 | 0.38108412109512213 | 0.22135172149674567 | Ours |  |
| D | D_dwt_db2_l2_threshold_none | logreg | success | success | 0.345321406241323 | 0.28777579430764116 | -0.05754561193368185 | Teacher |  |
| D | D_dwt_db2_l2_threshold_sure_soft | logreg | success | success | 0.25517589599705714 | 0.24970731299166923 | -0.005468583005387911 | Teacher |  |
| D | D_dwt_db2_l2_threshold_universal_hard | logreg | success | success | 0.33597772532880565 | 0.3042154877210812 | -0.03176223760772445 | Teacher |  |
| D | D_dwt_db2_l2_threshold_universal_soft | logreg | success | success | 0.27801640913824 | 0.27586818965765014 | -0.0021482194805898525 | Teacher |  |
| E | E_global_window_stat28_static_pooled_logreg | logreg | success | success | 0.3249821895711813 | 0.2943661723818191 | -0.030616017189362188 | Teacher |  |
| E | E_global_window_stat28_static_pooled_rf | rf | success | success | 0.20305753405645222 | 0.5271265539573191 | 0.3240690199008669 | Ours |  |
| E | E_global_window_stat28_static_pooled_xgboost | xgboost | success | success | 0.1980861751662915 | 0.3713342787528023 | 0.17324810358651077 | Ours |  |
| E | E_win4_s4_window_stat28_static_pooled_logreg | logreg | failure | failure |  |  |  |  | Teacher: DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0; Ours: DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_rf | rf | failure | failure |  |  |  |  | Teacher: DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0; Ours: DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_xgboost | xgboost | failure | failure |  |  |  |  | Teacher: DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0; Ours: DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win9_s9_window_stat28_static_pooled_logreg | logreg | success | success | 0.22111509936663223 | 0.37063573732478045 | 0.1495206379581482 | Ours |  |
| E | E_win9_s9_window_stat28_static_pooled_rf | rf | success | success | 0.3215977173690863 | 0.5494370394400006 | 0.22783932207091423 | Ours |  |
| E | E_win9_s9_window_stat28_static_pooled_xgboost | xgboost | success | success | 0.29955116933285886 | 0.4797623591354507 | 0.18021118980259182 | Ours |  |

The tables below report block winners and pairwise win rate from the same paired wide rows.

### Best By Block

| block | selection_scope | comparable_pairs | not_comparable_pairs | best_source | best_pair_id | best_config_name | best_value | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | comparable_both_success | 16 | 3 | Ours | A_cwt_morl_scales_1_16 | A_cwt_morl_scales_1_16::Ours::logreg | 0.36729456520778553 | 6 |
| B | comparable_both_success | 8 | 4 | Teacher | B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001::Teacher::lstm_projected_lr0p001 | 0.6685401150358344 | 8 |
| C | comparable_both_success | 13 | 0 | Teacher | C_raw_sequence_gru_projected__gru_projected_lr0p0003 | C_raw_sequence_gru_projected__gru_projected_lr0p0003::Teacher::gru_projected_lr0p0003 | 0.6038209099200795 | 0 |
| D | comparable_both_success | 4 | 0 | Teacher | D_dwt_db2_l2_threshold_none | D_dwt_db2_l2_threshold_none::Teacher::logreg | 0.345321406241323 | 0 |
| E | comparable_both_success | 6 | 3 | Ours | E_win9_s9_window_stat28_static_pooled_rf | E_win9_s9_window_stat28_static_pooled_rf::Ours::rf | 0.5494370394400006 | 6 |

### Pairwise Win Rate

| block | comparable_pairs | ours_wins | teacher_wins | ties | not_comparable_pairs | ours_winrate |
| --- | --- | --- | --- | --- | --- | --- |
| A | 16 | 12 | 4 | 0 | 3 | 0.750000 |
| B | 8 | 0 | 8 | 0 | 4 | 0.000000 |
| C | 13 | 5 | 8 | 0 | 0 | 0.384615 |
| D | 4 | 0 | 4 | 0 | 0 | 0.000000 |
| E | 6 | 5 | 1 | 0 | 3 | 0.833333 |
| overall | 47 | 22 | 25 | 0 | 10 | 0.468085 |

### Failures

| block | pair_id | source | config_name | status | failure_reason |
| --- | --- | --- | --- | --- | --- |
| A | A_dwt_db6_l2 | Teacher | A_dwt_db6_l2::Teacher::logreg | failure | DWT level 2 is not feasible for length 36 and wavelet 'db6'; max level is 1 |
| A | A_dwt_db6_l2 | Ours | A_dwt_db6_l2::Ours::logreg | failure | DWT level 2 is not feasible for length 36 and wavelet 'db6'; max level is 1 |
| A | A_dwt_sym6_l2 | Teacher | A_dwt_sym6_l2::Teacher::logreg | failure | DWT level 2 is not feasible for length 36 and wavelet 'sym6'; max level is 1 |
| A | A_dwt_sym6_l2 | Ours | A_dwt_sym6_l2::Ours::logreg | failure | DWT level 2 is not feasible for length 36 and wavelet 'sym6'; max level is 1 |
| A | A_dwt_coif2_l2 | Teacher | A_dwt_coif2_l2::Teacher::logreg | failure | DWT level 2 is not feasible for length 36 and wavelet 'coif2'; max level is 1 |
| A | A_dwt_coif2_l2 | Ours | A_dwt_coif2_l2::Ours::logreg | failure | DWT level 2 is not feasible for length 36 and wavelet 'coif2'; max level is 1 |
| B | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | Teacher | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001::Teacher::lstm_projected_lr0p001 | failure | SWT level 2 is not feasible for length 6; max level is 1 |
| B | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | Teacher | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003::Teacher::lstm_projected_lr0p0003 | failure | SWT level 2 is not feasible for length 6; max level is 1 |
| B | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | Ours | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001::Ours::lstm_projected_lr0p001 | failure | SWT level 2 is not feasible for length 6; max level is 1 |
| B | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | Ours | B_win6_s3_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003::Ours::lstm_projected_lr0p0003 | failure | SWT level 2 is not feasible for length 6; max level is 1 |
| B | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | Teacher | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001::Teacher::lstm_projected_lr0p001 | failure | SWT level 2 is not feasible for length 9; max level is 0 |
| B | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | Teacher | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003::Teacher::lstm_projected_lr0p0003 | failure | SWT level 2 is not feasible for length 9; max level is 0 |
| B | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001 | Ours | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001::Ours::lstm_projected_lr0p001 | failure | SWT level 2 is not feasible for length 9; max level is 0 |
| B | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003 | Ours | B_win9_s9_window_stat28_sequence_lstm_projected__lstm_projected_lr0p0003::Ours::lstm_projected_lr0p0003 | failure | SWT level 2 is not feasible for length 9; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_logreg | Teacher | E_win4_s4_window_stat28_static_pooled_logreg::Teacher::logreg | failure | DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_logreg | Ours | E_win4_s4_window_stat28_static_pooled_logreg::Ours::logreg | failure | DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_rf | Teacher | E_win4_s4_window_stat28_static_pooled_rf::Teacher::rf | failure | DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_rf | Ours | E_win4_s4_window_stat28_static_pooled_rf::Ours::rf | failure | DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_xgboost | Teacher | E_win4_s4_window_stat28_static_pooled_xgboost::Teacher::xgboost | failure | DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |
| E | E_win4_s4_window_stat28_static_pooled_xgboost | Ours | E_win4_s4_window_stat28_static_pooled_xgboost::Ours::xgboost | failure | DWT level 1 is not feasible for length 4 and wavelet 'db2'; max level is 0 |

## Interpretation

- paired best: block B, source Teacher, pair B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001, value 0.6685401150358344.
- winrate: Ours wins 22, Teacher wins 25, ties 0, comparable pairs 47, not comparable 10, Ours winrate 0.468085.
- failure counts: 20 non-success config rows are retained.

## Limitations

- limitations: these counts describe comparable paired rows, not an unpaired sweep.
- paired comparisons use only exact Teacher/Ours grid matches.
- failed configs remain in metrics_long.csv, metrics_wide_paired.csv, and failure_report.csv.
- non_comparable_paired_rows: 10
- failed_config_rows: 20

## Conclusion

The strongest block-level pr_auc row is block B, Teacher on B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001. 20 non-success rows remain part of the paired report.

<!-- domain-baseline-comparison:start -->
## Domain Baselines on the Same RePOPE Split

这些补充结果只写在小波课程 v2 输出目录中。

- domain_baselines_csv: outputs/wavelet_course_v2/reports/domain_baselines.csv
- domain_baseline_summary: outputs/wavelet_course_v2/reports/domain_baseline_comparison.md
- official_halp_cache: outputs/wavelet_course_v2/halp_cache/qwen3-vl-8b/repope/primary
- official_halp_policy: train on train split, choose probe and threshold on validation split, report test metrics.
- included_domain_methods: official HALP and linear probe only; MIND and HALP-like are not included.
- best_halp_official: halp_official_mlp, PR-AUC=0.538175, F1=0.520000
- best_linear_probe: linear_probe_final_hidden_logreg, PR-AUC=0.541455, F1=0.500000

### Current Wavelet V2 Best Rows

- best_teacher_bagua: B_global_window_stat28_sequence_lstm_projected__lstm_projected_lr0p001::Teacher::lstm_projected_lr0p001, PR-AUC=0.668540, F1=0.571429
- best_ours_wavelet: E_win9_s9_window_stat28_static_pooled_rf::Ours::rf, PR-AUC=0.549437, F1=0.390244

<!-- domain-baseline-comparison:end -->
