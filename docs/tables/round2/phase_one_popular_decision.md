# Phase-One Popular Decision

Current source:

- `outputs/round2_2026_04/decision_gate/pope_popular_phase_one_decision.json`

Decision:

- default full MIND feature set:
  - `raw + calibrated simple stats`
- headline paper framing:
  - keep the detector claim modest
  - drop `Multi-scale`
  - keep Haar only as an ablation

## Qwen3-VL-8B

| Variant | ROC-AUC | 95% CI | PR-AUC | 95% CI |
| --- | ---: | --- | ---: | --- |
| raw only | 0.8462 | [0.8228, 0.8696] | 0.1159 | [0.0928, 0.1450] |
| raw + simple stats | 0.8908 | [0.8694, 0.9105] | 0.1741 | [0.1375, 0.2168] |
| raw + full curve | 0.9145 | [0.8946, 0.9331] | 0.2596 | [0.1997, 0.3340] |
| raw + Haar | 0.8690 | [0.8445, 0.8918] | 0.1470 | [0.1153, 0.1842] |
| logit margin | 0.5955 | [0.5819, 0.6104] | 0.0422 | [0.0350, 0.0491] |

Takeaways:

- simple stats beats Haar on both metrics
- full curve is strongest, but not by a clean enough margin to justify the larger default feature set
- logit margin is far below every geometry-aware calibrated variant

## InternVL3.5-8B

| Variant | ROC-AUC | 95% CI | PR-AUC | 95% CI |
| --- | ---: | --- | ---: | --- |
| raw only | 0.8764 | [0.8594, 0.8943] | 0.4284 | [0.3685, 0.4939] |
| raw + simple stats | 0.8978 | [0.8810, 0.9140] | 0.5092 | [0.4529, 0.5669] |
| raw + full curve | 0.9119 | [0.8973, 0.9260] | 0.5333 | [0.4740, 0.5936] |
| raw + Haar | 0.8929 | [0.8764, 0.9092] | 0.4854 | [0.4274, 0.5466] |
| logit margin | 0.5454 | [0.5335, 0.5588] | 0.0861 | [0.0767, 0.0949] |

Takeaways:

- simple stats again beats Haar on both metrics
- full curve again looks strongest, but the gap over simple stats still overlaps heavily
- logit margin is again far below the geometry-based variants

## Paper Gate

- Haar does not win on both model families.
- The confidence intervals for simple stats and Haar overlap on both families.
- By the pre-registered Phase-One rule, the project should freeze `raw + calibrated simple stats` as the default full MIND feature set.
- The logit-margin fallback did not happen.
- So the paper can still say:
  - MIND beats simple output baselines
  - MIND does not yet beat richer upper baselines by default
