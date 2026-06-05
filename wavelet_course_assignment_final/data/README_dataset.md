# 数据集与缓存说明

## 基本设置

- 数据集：RePOPE。
- 模型：Qwen3-VL-8B-Instruct。
- 子集：popular、random、adversarial。
- 主任务：hard object hallucination 检测。
- 主指标：PR-AUC。正类比例只有 3.48%，所以 accuracy 不适合作为主指标。

## 样本定义

- 负类：模型回答正确。
- 正类：hard hallucination，即图像中不存在对象，但模型回答对象存在。
- 排除项：false negative、parsed none、invalid label。

## Hidden-state cache

实验复用 Stage 0 预提取 hidden-state cache，不重新下载数据，也不重新提取模型状态。每条样本包含 36 层 hidden states，每层 hidden state 为 4096 维。因此原始内部状态张量形状为 `(36, 4096)`。

完整 cache 位于：

```text
outputs/stage0/cache/qwen3-vl-8b/repope/<subset>/
```

本归档包只保存 `cache_manifest_excerpt.json` 和 `dataset_manifest.csv`，不复制完整 hidden-state cache。

## 数据划分

样本按 `image_id` 分组划分 train / validation / test，防止同一张图片同时出现在训练集和测试集。当前 primary population 为 7986，其中正类 278，负类 7708，正类比例约 3.48%。

划分统计：

- train：正类 192，负类 4603。
- validation：正类 48，负类 1531。
- test：正类 38，负类 1574。
