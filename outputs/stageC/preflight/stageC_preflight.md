# Stage C Preflight

Stage C compares support estimators on frozen Proxy Anchor embeddings.

- total_panel_models: 16
- evaluable_models: 15
- cache_root_readiness: ready
- split_readiness: ready
- fixed_objective: proxy_anchor
- fixed_encoder_family: Sphere-Traj-LSTM
- fixed_negative_budget_ratio: 0.5

## Excluded Models

- glm-4.6v-flash: answer format incompatible with frozen yes/no population rule
