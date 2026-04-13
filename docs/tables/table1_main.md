| model | benchmark | protocol | method | roc_auc | roc_auc_ci | pr_auc | pr_auc_ci |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | POPE popular | image_grouped | p_yes | 0.6342 | [0.6152, 0.6549] | 0.0451 | [0.0375, 0.0528] |
| Qwen3-VL-8B | POPE popular | image_grouped | logit_margin | 0.5955 | [0.5819, 0.6104] | 0.0422 | [0.0350, 0.0491] |
| Qwen3-VL-8B | POPE popular | image_grouped | chosen_confidence | 0.7947 | [0.7437, 0.8387] | 0.1462 | [0.1066, 0.1965] |
| Qwen3-VL-8B | POPE popular | image_grouped | drift_only | 0.8497 | [0.8262, 0.8741] | 0.1253 | [0.0985, 0.1605] |
| Qwen3-VL-8B | POPE popular | image_grouped | no_manifold | 0.8385 | [0.7958, 0.8783] | 0.1983 | [0.1385, 0.2657] |
| Qwen3-VL-8B | POPE popular | image_grouped | full MIND | 0.8908 | [0.8694, 0.9105] | 0.1741 | [0.1375, 0.2169] |
| Qwen3-VL-8B | POPE popular | image_grouped | linear_probe | 0.9161 | [0.8868, 0.9414] | 0.3803 | [0.2892, 0.4728] |
| Qwen3-VL-8B | DASH-B | image_grouped | p_yes | 0.4764 | [0.4335, 0.5219] | 0.2193 | [0.1834, 0.2612] |
| Qwen3-VL-8B | DASH-B | image_grouped | logit_margin | 0.6160 | [0.5697, 0.6613] | 0.2801 | [0.2598, 0.3004] |
| Qwen3-VL-8B | DASH-B | image_grouped | chosen_confidence | 0.6856 | [0.6559, 0.7153] | 0.4057 | [0.3678, 0.4420] |
| Qwen3-VL-8B | DASH-B | image_grouped | drift_only | 0.9185 | [0.9044, 0.9324] | 0.7371 | [0.6989, 0.7729] |
| Qwen3-VL-8B | DASH-B | image_grouped | no_manifold | 0.9290 | [0.9162, 0.9415] | 0.7784 | [0.7433, 0.8118] |
| Qwen3-VL-8B | DASH-B | image_grouped | full MIND | 0.9193 | [0.9049, 0.9332] | 0.7374 | [0.6965, 0.7755] |
| Qwen3-VL-8B | DASH-B | image_grouped | linear_probe | 0.9909 | [0.9874, 0.9938] | 0.9779 | [0.9714, 0.9841] |
| InternVL3.5-8B | POPE popular | image_grouped | p_yes | 0.5601 | [0.5471, 0.5740] | 0.0878 | [0.0783, 0.0969] |
| InternVL3.5-8B | POPE popular | image_grouped | logit_margin | 0.5454 | [0.5335, 0.5588] | 0.0861 | [0.0767, 0.0949] |
| InternVL3.5-8B | POPE popular | image_grouped | chosen_confidence | 0.8039 | [0.7745, 0.8342] | 0.2637 | [0.2198, 0.3142] |
| InternVL3.5-8B | POPE popular | image_grouped | drift_only | 0.8802 | [0.8622, 0.8982] | 0.4270 | [0.3680, 0.4888] |
| InternVL3.5-8B | POPE popular | image_grouped | no_manifold | 0.8559 | [0.8351, 0.8761] | 0.4033 | [0.3393, 0.4618] |
| InternVL3.5-8B | POPE popular | image_grouped | full MIND | 0.8978 | [0.8810, 0.9140] | 0.5092 | [0.4528, 0.5669] |
| InternVL3.5-8B | POPE popular | image_grouped | linear_probe | 0.9366 | [0.9192, 0.9518] | 0.6550 | [0.5881, 0.7133] |
| InternVL3.5-8B | DASH-B | image_grouped | p_yes | 0.4790 | [0.4367, 0.5228] | 0.2558 | [0.2159, 0.3004] |
| InternVL3.5-8B | DASH-B | image_grouped | logit_margin | 0.5584 | [0.5150, 0.5993] | 0.2893 | [0.2665, 0.3116] |
| InternVL3.5-8B | DASH-B | image_grouped | chosen_confidence | 0.6475 | [0.6179, 0.6748] | 0.3860 | [0.3507, 0.4218] |
| InternVL3.5-8B | DASH-B | image_grouped | drift_only | 0.8593 | [0.8426, 0.8747] | 0.7054 | [0.6636, 0.7409] |
| InternVL3.5-8B | DASH-B | image_grouped | no_manifold | 0.8769 | [0.8621, 0.8918] | 0.7288 | [0.6907, 0.7630] |
| InternVL3.5-8B | DASH-B | image_grouped | full MIND | 0.8574 | [0.8404, 0.8728] | 0.7084 | [0.6669, 0.7437] |
| InternVL3.5-8B | DASH-B | image_grouped | linear_probe | 0.9858 | [0.9821, 0.9893] | 0.9699 | [0.9617, 0.9768] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | p_yes | 0.6200 | [0.6044, 0.6366] | 0.0357 | [0.0289, 0.0429] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | logit_margin | 0.6095 | [0.5933, 0.6266] | 0.0347 | [0.0280, 0.0415] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | chosen_confidence | 0.8277 | [0.7772, 0.8742] | 0.1195 | [0.0877, 0.1585] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | drift_only | 0.8030 | [0.7708, 0.8364] | 0.0941 | [0.0683, 0.1332] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | no_manifold | 0.8078 | [0.7618, 0.8537] | 0.1282 | [0.0886, 0.1947] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | full MIND | 0.8085 | [0.7809, 0.8405] | 0.0874 | [0.0642, 0.1225] |
| LLaVA-OneVision-7B | POPE popular | image_grouped | linear_probe | 0.8833 | [0.8403, 0.9228] | 0.3238 | [0.2347, 0.4311] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | p_yes | 0.4497 | [0.4009, 0.5000] | 0.3231 | [0.2726, 0.3801] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | logit_margin | 0.6602 | [0.6198, 0.6949] | 0.4297 | [0.4019, 0.4562] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | chosen_confidence | 0.7046 | [0.6754, 0.7331] | 0.4938 | [0.4596, 0.5265] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | drift_only | 0.8664 | [0.8488, 0.8830] | 0.7431 | [0.7105, 0.7736] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | no_manifold | 0.8996 | [0.8839, 0.9130] | 0.7883 | [0.7549, 0.8201] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | full MIND | 0.8404 | [0.8208, 0.8599] | 0.7234 | [0.6893, 0.7553] |
| LLaVA-OneVision-7B | DASH-B | image_grouped | linear_probe | 0.9923 | [0.9896, 0.9944] | 0.9883 | [0.9845, 0.9916] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | p_yes | 0.5658 | [0.5412, 0.5908] | 0.0512 | [0.0430, 0.0591] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | logit_margin | 0.5810 | [0.5658, 0.5964] | 0.0541 | [0.0462, 0.0618] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | chosen_confidence | 0.6522 | [0.6200, 0.6863] | 0.0687 | [0.0556, 0.0834] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | drift_only | 0.8346 | [0.8099, 0.8578] | 0.1651 | [0.1324, 0.2014] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | no_manifold | 0.8256 | [0.7956, 0.8557] | 0.1857 | [0.1459, 0.2341] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | full MIND | 0.8839 | [0.8608, 0.9060] | 0.2992 | [0.2327, 0.3691] |
| Molmo-7B-D-0924 | POPE popular | image_grouped | linear_probe | 0.9209 | [0.8988, 0.9424] | 0.5606 | [0.4703, 0.6409] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | p_yes | 0.6170 | [0.5772, 0.6528] | 0.3289 | [0.3022, 0.3538] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | logit_margin | 0.5369 | [0.4955, 0.5745] | 0.2878 | [0.2634, 0.3099] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | chosen_confidence | 0.6369 | [0.5992, 0.6718] | 0.3425 | [0.3164, 0.3673] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | drift_only | 0.7967 | [0.7775, 0.8149] | 0.5611 | [0.5185, 0.6002] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | no_manifold | 0.8655 | [0.8479, 0.8819] | 0.6861 | [0.6479, 0.7191] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | full MIND | 0.7795 | [0.7589, 0.7978] | 0.5422 | [0.5010, 0.5794] |
| Molmo-7B-D-0924 | DASH-B | image_grouped | linear_probe | 0.9775 | [0.9722, 0.9827] | 0.9561 | [0.9450, 0.9655] |