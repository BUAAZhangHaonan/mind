# Sub-task 2 Anisotropic Analysis

## Decision

Converge the method here. The best anisotropic variant, `lowrank_maha`, does not improve mean PR-AUC over the locked `radius_ball_isotropic` baseline.

- Mean PR-AUC over 8 image-grouped settings: `radius_ball_isotropic` = 0.4795, `lowrank_maha` = 0.4787.
- The mean linear-probe gap is not narrowed: isotropic gap = 0.2676, lowrank gap = 0.2684.
- On DASH-B, the gap gets larger: isotropic gap = 0.1865, lowrank gap = 0.2209.

The expert decision gate lands on: the signal is already captured by isotropic local geometry. Local anisotropic covariance does not add enough value to continue the method paper as a stronger detector claim.

## Radius Setup

Sub-task 1 locked the neighbor selection method to `radius_ball`. Sub-task 2 kept that choice and used fixed angular radii tuned from the existing caches to target 30 neighbors per query-layer on average. Each query-layer used the radius-ball set, with a support floor of 2 nearest angular neighbors only when the radius returned fewer than 2 neighbors. This floor made covariance scores defined without widening every radius.

| model | benchmark | angular radii by selected layer |
| --- | --- | --- |
| internvl3.5-8b | DASH-B | 0.221008, 0.221086, 0.236631, 0.243659, 0.236494, 0.244913, 0.224932, 0.227157, 0.225545, 0.273238, 0.282911, 0.286254, 0.319502, 0.324295, 0.320799, 0.319225 |
| internvl3.5-8b | POPE popular | 0.169474, 0.181651, 0.185352, 0.198432, 0.234952, 0.249760, 0.245718, 0.261495, 0.257373, 0.311600, 0.325627, 0.323543, 0.321814, 0.315703, 0.301586, 0.294712 |
| llava-onevision-7b | DASH-B | 0.276936, 0.319617, 0.300379, 0.303991, 0.326443, 0.309687, 0.312350, 0.275232, 0.298659, 0.306311, 0.312740, 0.344730, 0.338754, 0.310790 |
| llava-onevision-7b | POPE popular | 0.180429, 0.192473, 0.192023, 0.190949, 0.215231, 0.219900, 0.232057, 0.207383, 0.241621, 0.244702, 0.251164, 0.291190, 0.297666, 0.277670 |
| molmo-7b-d-0924 | DASH-B | 0.190725, 0.206806, 0.228425, 0.217542, 0.225821, 0.208414, 0.203026, 0.183203, 0.190976, 0.190093, 0.194291, 0.234193, 0.261004, 0.255936 |
| molmo-7b-d-0924 | POPE popular | 0.198316, 0.205834, 0.219246, 0.210065, 0.225676, 0.213895, 0.218314, 0.208438, 0.222625, 0.226251, 0.235686, 0.275625, 0.296834, 0.291209 |
| qwen3-vl-8b | DASH-B | 0.101848, 0.095242, 0.099003, 0.114894, 0.109580, 0.112134, 0.107063, 0.112365, 0.112306, 0.142616, 0.162157, 0.172486, 0.173181, 0.164998, 0.152521, 0.145858 |
| qwen3-vl-8b | POPE popular | 0.087233, 0.090730, 0.088422, 0.097698, 0.098792, 0.105817, 0.109030, 0.127289, 0.132957, 0.172997, 0.195055, 0.214689, 0.210920, 0.201402, 0.180175, 0.171815 |

## Main Comparison

`lowrank_maha` is the best anisotropic variant by mean DASH-B PR-AUC, but it still trails isotropic radius-ball on DASH-B and on the full 8-row mean.

| method | mean PR-AUC over 8 | mean DASH-B PR-AUC | mean POPE popular PR-AUC |
| --- | ---: | ---: | ---: |
| radius_ball_isotropic | 0.4795 | 0.7947 | 0.1644 |
| diag_maha | 0.4074 | 0.6376 | 0.1773 |
| lowrank_maha | 0.4787 | 0.7603 | 0.1970 |
| full_maha_shrink | 0.4504 | 0.7281 | 0.1726 |
| no_manifold | 0.4525 | 0.7237 | 0.1813 |
| linear_probe | 0.7471 | 0.9812 | 0.5122 |
| query_local_k30 | 0.6345 | 0.9369 | 0.3320 |

## Answers

1. No anisotropic variant beats `radius_ball_isotropic` on mean PR-AUC across the 8 settings. `lowrank_maha` is closest, but it is lower by 0.0009 PR-AUC.

2. `lowrank_maha` does not narrow the mean gap to `linear_probe`. The per-row PR-AUC gaps are:

| model | benchmark | lowrank_maha PR-AUC | linear_probe PR-AUC | gap |
| --- | --- | ---: | ---: | ---: |
| internvl3.5-8b | DASH-B | 0.6532 | 0.9822 | 0.3290 |
| internvl3.5-8b | POPE popular | 0.3081 | 0.6785 | 0.3703 |
| llava-onevision-7b | DASH-B | 0.7916 | 0.9909 | 0.1993 |
| llava-onevision-7b | POPE popular | 0.1223 | 0.3240 | 0.2017 |
| molmo-7b-d-0924 | DASH-B | 0.7707 | 0.9635 | 0.1928 |
| molmo-7b-d-0924 | POPE popular | 0.2176 | 0.5946 | 0.3770 |
| qwen3-vl-8b | DASH-B | 0.8257 | 0.9880 | 0.1623 |
| qwen3-vl-8b | POPE popular | 0.1401 | 0.4552 | 0.3151 |

3. On DASH-B, `lowrank_maha` beats `no_manifold` on 3 of 4 models, not all four.

| model | lowrank_maha PR-AUC | no_manifold PR-AUC | delta |
| --- | ---: | ---: | ---: |
| internvl3.5-8b | 0.6532 | 0.7031 | -0.0498 |
| llava-onevision-7b | 0.7916 | 0.7790 | +0.0126 |
| molmo-7b-d-0924 | 0.7707 | 0.6736 | +0.0971 |
| qwen3-vl-8b | 0.8257 | 0.7392 | +0.0865 |

4. The anisotropic advantage is larger on POPE popular than on DASH-B. Against isotropic, `lowrank_maha` gains +0.0326 mean PR-AUC on POPE popular, but loses -0.0344 on DASH-B.

5. On `object_heldout`, `lowrank_maha` beats `linear_probe` on 4 of 4 models in PR-AUC and beats `no_manifold` on 3 of 4 models. It does not beat the old `query_local_k30` heldout results: mean heldout PR-AUC is 0.1166 for `lowrank_maha` and 0.1370 for `query_local_k30`.

| model | lowrank_maha PR-AUC | query_local_k30 PR-AUC | linear_probe PR-AUC |
| --- | ---: | ---: | ---: |
| internvl3.5-8b | 0.2499 | 0.2846 | 0.2206 |
| llava-onevision-7b | 0.0705 | 0.0749 | 0.0443 |
| molmo-7b-d-0924 | 0.0670 | 0.0771 | 0.0487 |
| qwen3-vl-8b | 0.0789 | 0.1114 | 0.0671 |

6. `lowrank_maha` works best overall among anisotropic variants. That supports a weak low-dimensional covariance signal, but the signal is not stable enough to improve the detector over mean angular distance. Full shrinkage helps Qwen and InternVL on DASH-B, which suggests some covariance structure exists, but it is not a universal gain.

7. There is evidence that local covariance structure is model-specific. Anisotropic winners vary: `lowrank_maha` wins 4 of 8 rows, `full_maha_shrink` wins 3 of 8, and `diag_maha` wins 1 of 8.

| model | benchmark | best anisotropic variant |
| --- | --- | --- |
| internvl3.5-8b | DASH-B | full_maha_shrink |
| internvl3.5-8b | POPE popular | lowrank_maha |
| llava-onevision-7b | DASH-B | lowrank_maha |
| llava-onevision-7b | POPE popular | diag_maha |
| molmo-7b-d-0924 | DASH-B | lowrank_maha |
| molmo-7b-d-0924 | POPE popular | lowrank_maha |
| qwen3-vl-8b | DASH-B | full_maha_shrink |
| qwen3-vl-8b | POPE popular | full_maha_shrink |

## Final Call

The best anisotropic variant gives marginal change over isotropic radius-ball, below the 0.02 PR-AUC threshold and slightly negative on average. The method should converge at isotropic radius-ball geometry for this pipeline.
