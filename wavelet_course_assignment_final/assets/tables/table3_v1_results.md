| Method | Best Config | PR-AUC | F1 | Main Interpretation |
| --- | --- | --- | --- | --- |
| Teacher-Bagua | teacher_bagua_haar_l1_lstm | 0.0291 | 0.0498 | 严格模板迁移效果接近随机先验，说明方法假设和 hidden states 结构失配。 |
| Ours-Wavelet | ours_db2_swt_l2_logreg | 0.3976 | 0.4634 | 语义层间轨迹上的小波分析明显优于严格模板迁移。 |
