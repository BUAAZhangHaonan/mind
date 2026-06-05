| Insight | Evidence |
| --- | --- |
| 严格模板迁移失败 | v1 Teacher PR-AUC 0.0291，F1 0.0498。 |
| Ours 更适合小波摘要和 28 特征工程 | Block A Ours 12/16 胜，Block E Ours 5/6 胜。 |
| Teacher 在高容量序列模型下可以更强 | Block B Teacher best PR-AUC 0.6685。 |
| Teacher best 高分来自高维统计特征和大容量模型 | 输入 114688 维，再投影到 256 维，投影层约 2936 万参数。 |
| 小波去噪不能直接照搬 | Block D 最佳阈值为 none，高频变化本身包含判别信号。 |
| 预测性能和方法解释性必须区分 | 高维 Teacher 可以预测更强，但不说明 hidden dimension 是物理传感器。 |
