# Stage C Summary

Stage C compares detector families on frozen Proxy Anchor embeddings.
Stage D has not started.

- stage_d_started: false
- objective: proxy_anchor
- negative_budget_ratio: 0.5
- support_winner: radius_ball
- comparator_status: beats_supervised
- panel_verdict: nonparametric_winner

## Outputs

- metrics_long: outputs/stageC/reports/stageC_metrics_long.csv
- repope_main_table: outputs/stageC/reports/repope_main_table.csv
- knn_selected_k: outputs/stageC/reports/knn_selected_k.csv
- radius_ball_selected_rho: outputs/stageC/reports/radius_ball_selected_rho.csv
- vmf_selected_k: outputs/stageC/reports/vmf_selected_k.csv
- logistic_selected_c: outputs/stageC/reports/logistic_selected_c.csv

## Excluded Models

- glm-4.6v-flash: answer format incompatible with frozen yes/no population rule
