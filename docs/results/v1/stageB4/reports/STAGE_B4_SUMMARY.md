# Stage B4 Summary

Stage B4 evaluates Proxy Anchor support families at a fixed 0.5 ratio.
Stage C has not started.

- stage_c_started: false
- detector_selected: false
- objective: proxy_anchor
- negative_budget_ratio: 0.5
- verdict: parametric_support_preferred

## Outputs

- metrics_long: outputs/stageB4/reports/stageB4_metrics_long.csv
- repope_support_family_knn: outputs/stageB4/reports/repope_support_family_knn.csv
- repope_support_family_single_vmf: outputs/stageB4/reports/repope_support_family_single_vmf.csv
- repope_support_family_mixture_vmf: outputs/stageB4/reports/repope_support_family_mixture_vmf.csv
- knn_scale_grid: outputs/stageB4/reports/knn_scale_grid.csv
- knn_stability_band: outputs/stageB4/reports/knn_stability_band.csv
- vmf_stability_band: outputs/stageB4/reports/vmf_stability_band.csv
- classifier_control: outputs/stageB4/reports/classifier_control.csv

## Excluded Models

- glm-4.6v-flash: answer format incompatible with frozen yes/no population rule
