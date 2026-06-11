# Stage B3 Summary

Stage B3 evaluates Proxy Anchor kNN scale stability at a fixed 0.5 ratio.
Stage C has not started.

- stage_c_started: false
- objective: proxy_anchor
- negative_budget_ratio: 0.5
- verdict: scale_sensitive_panel

## Outputs

- metrics_long: outputs/stageB3/reports/stageB3_metrics_long.csv
- knn_scale_grid: outputs/stageB3/reports/knn_scale_grid.csv
- knn_stability_band: outputs/stageB3/reports/knn_stability_band.csv
- classifier_control: outputs/stageB3/reports/classifier_control.csv
- vmf_probe_summary: outputs/stageB3/reports/vmf_probe_summary.csv

## Excluded Models

- glm-4.6v-flash: answer format incompatible with frozen yes/no population rule
