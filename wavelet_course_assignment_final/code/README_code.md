# 课程大作业代码说明

本目录保存小波课程实验相关代码副本，不包含 Stage 0 大缓存。

## 脚本

- `scripts/wavelet_course_run.py`：v1 实验入口。它运行严格模板迁移版 Teacher-Bagua、Ours-Wavelet 和线性探针类 baseline。
- `scripts/wavelet_course_v2_run.py`：v2 配对控制实验入口。它生成 Teacher/Ours 成对配置，并输出 long/wide 结果表。
- `scripts/wavelet_course_spatial_run.py`：逐层 4096 维空间小波补充实验入口。
- `scripts/wavelet_course_make_ppt_figures.py`：从 v2 结果生成展示图。
- `scripts/wavelet_course_domain_baselines.py`：领域内 hidden-state probe 参照实验入口。
- `scripts/wavelet_course_extract_halp_cache.py`：HALP 所需轻量缓存提取入口。

## 配置

- `configs/wavelet_course/repope_qwen3_vl_8b.yaml`：v1 实验配置。
- `configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml`：v2 配对实验配置。

## 模块入口

- Teacher-Bagua v1 特征：`src/mind/wavelet_course/teacher_bagua_features.py`。
- Ours-Wavelet v1 特征：`src/mind/wavelet_course/ours_wavelet_features.py`。
- v2 信号构造：`src/mind/wavelet_course/signal_builders.py`。
- v2 小波、窗口、特征协议：`common_wavelet.py`、`common_windowing.py`、`common_feature_protocols.py`。
- v2 分类器和时序模型：`common_classifiers.py`、`common_sequence_models.py`。
- v2 配对网格和报告：`paired_grid.py`、`paired_runner.py`、`paired_reporting.py`。

## 复现实验

先确保 `outputs/stage0` 中已有 Qwen3-VL-8B-Instruct 在 RePOPE 上的 full-layer hidden-state cache。

v1：

```bash
conda run --no-capture-output -n mind-py311 python scripts/wavelet_course_run.py \
  --config configs/wavelet_course/repope_qwen3_vl_8b.yaml --device cuda:0
```

v2：

```bash
conda run --no-capture-output -n mind-py311 python scripts/wavelet_course_v2_run.py \
  --config configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml --device cuda:0
```

测试：

```bash
conda run --no-capture-output -n mind-py311 pytest tests/wavelet_course -q
```
