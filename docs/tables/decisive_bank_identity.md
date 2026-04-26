# Decisive Bank-Identity Controls

| setting | model | bank_scope | pr_auc | roc_auc | rank_by_pr_auc |
| --- | --- | --- | --- | --- | --- |
| popular-object-heldout | qwen3-vl-8b | object | 0.0611 | 0.7211 | 3 |
| popular-object-heldout | qwen3-vl-8b | shared | 0.0820 | 0.8074 | 1 |
| popular-object-heldout | qwen3-vl-8b | shuffled_object | 0.0658 | 0.7381 | 2 |
| dash-b | qwen3-vl-8b | object | 0.7505 | 0.9260 | 2 |
| dash-b | qwen3-vl-8b | shared | 0.6865 | 0.8954 | 3 |
| dash-b | qwen3-vl-8b | shuffled_object | 0.7613 | 0.9189 | 1 |
| popular-object-heldout | molmo-7b-d-0924 | object | 0.1043 | 0.7766 | 2 |
| popular-object-heldout | molmo-7b-d-0924 | shared | 0.0775 | 0.7089 | 3 |
| popular-object-heldout | molmo-7b-d-0924 | shuffled_object | 0.1052 | 0.7550 | 1 |
| dash-b | molmo-7b-d-0924 | object | 0.5897 | 0.7995 | 1 |
| dash-b | molmo-7b-d-0924 | shared | 0.5376 | 0.7530 | 2 |
| dash-b | molmo-7b-d-0924 | shuffled_object | 0.5368 | 0.7103 | 3 |

## Generated Interpretation

- qwen3-vl-8b / popular-object-heldout: object-conditioned did not rank first by PR-AUC.
- qwen3-vl-8b / dash-b: object-conditioned did not rank first by PR-AUC.
- molmo-7b-d-0924 / popular-object-heldout: object-conditioned did not rank first by PR-AUC.
- molmo-7b-d-0924 / dash-b: object-conditioned ranked first by PR-AUC.
