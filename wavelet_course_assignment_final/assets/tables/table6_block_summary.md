| Block | Question | Main Result | Interpretation |
| --- | --- | --- | --- |
| A | 小波函数与小波变换是否改变结论 | Ours 12/16 胜；最佳为 CWT Morlet Ours，PR-AUC 0.3673。 | 语义层间轨迹更适合小波摘要分析。 |
| B | 不同窗口策略对两类信号的影响 | Teacher 8/8 胜；global + 28 统计特征 + projected LSTM 达到 PR-AUC 0.6685。 | 高维原始 hidden states 含有强判别信息，大容量模型可以利用它。 |
| C | 不同分类器和时序模型的影响 | Teacher 在 raw sequence 时序模型中更强；Ours 在 wavelet summary static pooled 中更强。 | 两类信号适合不同建模路线，不能用单一胜负概括。 |
| D | 小波去噪策略是否稳定有效 | Teacher 4/4 胜；最佳为 no threshold。 | hidden states 的高频变化不能直接视为噪声。 |
| E | 28 个传统统计特征应该作用在哪类信号上 | Ours 5/6 胜；关键配对中 Ours 比 Teacher 提升 0.3241 PR-AUC。 | 28 特征不是无效，而是更适合语义轨迹。 |
