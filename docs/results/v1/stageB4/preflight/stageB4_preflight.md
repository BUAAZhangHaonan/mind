# Stage B4 Preflight

Stage B4 evaluates Proxy Anchor support families at a fixed 0.5 ratio.

- total_panel_models: 16
- evaluable_models: 15
- cache_root_readiness: ready
- split_readiness: ready
- fixed_objective: proxy_anchor
- fixed_encoder_family: Sphere-Traj-LSTM
- fixed_negative_budget_ratio: 0.5

## Excluded Models

- glm-4.6v-flash: answer format incompatible with frozen yes/no population rule
