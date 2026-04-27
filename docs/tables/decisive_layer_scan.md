# Decisive Layer-Count Sensitivity

| setting | model | layer_count | bank_scope | method | pr_auc | roc_auc | best_layer_for_method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popular-object-heldout | qwen3-vl-8b | 8 | object | full | 0.0673 | 0.7434 | no |
| popular-object-heldout | qwen3-vl-8b | 8 | object | linear_probe | 0.1150 | 0.7282 | no |
| popular-object-heldout | qwen3-vl-8b | 8 | object | no_manifold | 0.0691 | 0.6479 | no |
| popular-object-heldout | qwen3-vl-8b | 12 | object | full | 0.0730 | 0.7599 | yes |
| popular-object-heldout | qwen3-vl-8b | 12 | object | linear_probe | 0.1097 | 0.7328 | no |
| popular-object-heldout | qwen3-vl-8b | 12 | object | no_manifold | 0.0648 | 0.6234 | no |
| popular-object-heldout | qwen3-vl-8b | 16 | object | full | 0.0710 | 0.7446 | no |
| popular-object-heldout | qwen3-vl-8b | 16 | object | linear_probe | 0.1182 | 0.7346 | yes |
| popular-object-heldout | qwen3-vl-8b | 16 | object | no_manifold | 0.0849 | 0.6622 | yes |
| dash-b | qwen3-vl-8b | 8 | object | full | 0.6939 | 0.8997 | no |
| dash-b | qwen3-vl-8b | 8 | object | linear_probe | 0.9796 | 0.9914 | yes |
| dash-b | qwen3-vl-8b | 8 | object | no_manifold | 0.5682 | 0.8599 | no |
| dash-b | qwen3-vl-8b | 12 | object | full | 0.7347 | 0.9145 | no |
| dash-b | qwen3-vl-8b | 12 | object | linear_probe | 0.9775 | 0.9905 | no |
| dash-b | qwen3-vl-8b | 12 | object | no_manifold | 0.5940 | 0.8585 | no |
| dash-b | qwen3-vl-8b | 16 | object | full | 0.7471 | 0.9251 | yes |
| dash-b | qwen3-vl-8b | 16 | object | linear_probe | 0.9780 | 0.9909 | no |
| dash-b | qwen3-vl-8b | 16 | object | no_manifold | 0.7628 | 0.9257 | yes |
| popular-object-heldout | molmo-7b-d-0924 | 8 | object | full | 0.0991 | 0.7755 | no |
| popular-object-heldout | molmo-7b-d-0924 | 8 | object | linear_probe | 0.0553 | 0.5519 | no |
| popular-object-heldout | molmo-7b-d-0924 | 8 | object | no_manifold | 0.0374 | 0.4007 | no |
| popular-object-heldout | molmo-7b-d-0924 | 12 | object | full | 0.1023 | 0.7814 | yes |
| popular-object-heldout | molmo-7b-d-0924 | 12 | object | linear_probe | 0.0565 | 0.5458 | yes |
| popular-object-heldout | molmo-7b-d-0924 | 12 | object | no_manifold | 0.0400 | 0.4395 | yes |
| dash-b | molmo-7b-d-0924 | 8 | object | full | 0.4592 | 0.7107 | no |
| dash-b | molmo-7b-d-0924 | 8 | object | linear_probe | 0.9577 | 0.9786 | yes |
| dash-b | molmo-7b-d-0924 | 8 | object | no_manifold | 0.6725 | 0.8600 | no |
| dash-b | molmo-7b-d-0924 | 12 | object | full | 0.5472 | 0.7727 | yes |
| dash-b | molmo-7b-d-0924 | 12 | object | linear_probe | 0.9574 | 0.9783 | no |
| dash-b | molmo-7b-d-0924 | 12 | object | no_manifold | 0.6791 | 0.8613 | yes |

## Generated Interpretation

- qwen3-vl-8b / popular-object-heldout: full MIND best at 12 layers by PR-AUC; 16-layer default is present but not best.
- qwen3-vl-8b / dash-b: full MIND best at 16 layers by PR-AUC; 16-layer default is best.
- molmo-7b-d-0924 / popular-object-heldout: full MIND best at 12 layers by PR-AUC; 16-layer default was not evaluated.
- molmo-7b-d-0924 / dash-b: full MIND best at 12 layers by PR-AUC; 16-layer default was not evaluated.
