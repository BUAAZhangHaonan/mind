# MIND Results Summary

## Correction Phase

This file now treats the 2026-03-30 correction phase as the main result set for the paper decision.

What changed in this phase:

- removed per-sample drift-curve normalization
- kept raw manifold magnitude features
- calibrated drift curves from cleaned reference-bank statistics
- applied Haar only to calibrated curves
- cleaned reference banks with `parsed_answer == 1`
- switched the primary evaluation protocol from row-wise splitting to `image_grouped`

All corrected outputs live under:

- `outputs/correction_phase/reference_banks/`
- `outputs/correction_phase/features/`
- `outputs/correction_phase/reports/`
- `outputs/correction_phase/plots/`

## Closeout Follow-up: 2026-04-01

Closeout scope in this phase:

- keep `image_grouped` as the only primary protocol
- add the shared-bank control
- relabel the same corrected popular predictions with RePOPE
- add corrected adversarial reruns
- export a script-generated paper package

Already completed in the closeout phase:

- RePOPE relabel on the corrected popular predictions
  - Qwen popular + RePOPE:
    - `ROC-AUC 0.888700`
    - `PR-AUC 0.257797`
    - `TPR@1%FPR 0.130081`
  - InternVL popular + RePOPE:
    - `ROC-AUC 0.882635`
    - `PR-AUC 0.488681`
    - `TPR@1%FPR 0.208904`
- corrected Qwen adversarial rerun under `image_grouped`
  - `ROC-AUC 0.870756`
  - `PR-AUC 0.265281`
  - `TPR@1%FPR 0.070175`
- corrected InternVL adversarial rerun under `image_grouped`
  - `ROC-AUC 0.859557`
  - `PR-AUC 0.443024`
  - `TPR@1%FPR 0.142857`
- shared-bank control on corrected popular and corrected `object_heldout`
  - Qwen popular + shared bank:
    - `ROC-AUC 0.897876`
    - `PR-AUC 0.198559`
    - `TPR@1%FPR 0.117117`
  - InternVL popular + shared bank:
    - `ROC-AUC 0.866674`
    - `PR-AUC 0.340905`
    - `TPR@1%FPR 0.101562`
  - Qwen `object_heldout` + shared bank:
    - `ROC-AUC 0.862392`
    - `PR-AUC 0.131936`
    - `TPR@1%FPR 0.036036`
  - InternVL `object_heldout` + shared bank:
    - `ROC-AUC 0.830655`
    - `PR-AUC 0.254406`
    - `TPR@1%FPR 0.050781`
- script-generated paper package export
  - `artifacts/paper_closeout/tables/`
  - `artifacts/paper_closeout/figures/`
  - `artifacts/paper_closeout/figure_manifest.json`

Closeout completion:

- the A100 environment was verified healthy before the rerun:
  - `2 x NVIDIA A100 80GB PCIe`
  - `torch.cuda.is_available() == True`
  - `make test` passed with `93` tests
- the final missing InternVL adversarial cache completed with:
  - `24` shards
  - `3000` cached entries
- the full closeout export is no longer blocked

## Shared-Bank Control

### Popular, `image_grouped`

| Model | Object Bank ROC-AUC | Shared Bank ROC-AUC | Object Bank PR-AUC | Shared Bank PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 0.917113 | 0.897876 | 0.283927 | 0.198559 |
| InternVL | 0.914178 | 0.866674 | 0.543810 | 0.340905 |

Takeaways:

- On the primary grouped protocol, the shared bank hurts both model families.
- The object-conditioned bank is therefore carrying useful signal on popular, not just memorized structure.

### Popular, `object_heldout`

| Model | Object Bank ROC-AUC | Shared Bank ROC-AUC | Object Bank PR-AUC | Shared Bank PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 0.724419 | 0.862392 | 0.063808 | 0.131936 |
| InternVL | 0.839763 | 0.830655 | 0.409716 | 0.254406 |

Takeaways:

- On Qwen, the shared bank improves held-out object transfer substantially.
- On InternVL, the shared bank still hurts both ROC-AUC and PR-AUC.
- The safer paper reading is that object conditioning buys popular accuracy for both models, but it weakens transfer much more sharply for Qwen than for InternVL.

## Primary Protocol: `image_grouped`

### Qwen Popular

Model: `Qwen/Qwen3-VL-8B-Instruct`

| Variant | ROC-AUC | PR-AUC | TPR@1%FPR | F1 |
| --- | ---: | ---: | ---: | ---: |
| full MIND | 0.917113 | 0.283927 | 0.144144 | 0.000000 |
| drift-only | 0.849675 | 0.125340 | 0.036036 | 0.000000 |
| no-manifold | 0.838508 | 0.198308 | 0.081081 | 0.000000 |
| linear probe | 0.916075 | 0.380287 | 0.252252 | 0.417476 |

Takeaways:

- The corrected full MIND signal clearly beats corrected drift-only and corrected no-manifold.
- The ROC-AUC gap to the linear probe is effectively closed on Qwen popular.
- The PR-AUC gap is still large enough that MIND should not be written up as the strongest detector.

### InternVL Popular

Model: `OpenGVLab/InternVL3_5-8B-HF`

| Variant | ROC-AUC | PR-AUC | TPR@1%FPR | F1 |
| --- | ---: | ---: | ---: | ---: |
| full MIND | 0.914178 | 0.543810 | 0.253906 | 0.373938 |
| drift-only | 0.880153 | 0.427028 | 0.167969 | 0.113879 |
| no-manifold | 0.855884 | 0.403316 | 0.164062 | 0.247678 |
| linear probe | 0.936664 | 0.655071 | 0.320312 | 0.636054 |

Takeaways:

- The corrected full MIND signal again beats corrected drift-only and corrected no-manifold.
- InternVL remains stronger than Qwen on the same corrected popular split in PR-AUC.
- The linear probe still leads on the primary grouped protocol.

## Adversarial Check: `image_grouped`

| Model | ROC-AUC | PR-AUC | TPR@1%FPR | F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen adversarial | 0.870756 | 0.265281 | 0.070175 | 0.000000 |
| InternVL adversarial | 0.859557 | 0.443024 | 0.142857 | 0.230216 |

Takeaways:

- The adversarial split is harder than popular for both families.
- InternVL keeps a large PR-AUC advantage on adversarial even though Qwen keeps a slightly higher ROC-AUC.
- The final paper package now has all six grouped closeout rows.

## Legacy Comparison: `row`

The row split is now legacy mode only. It stays here for continuity with older repo outputs.

| Model | Variant | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: |
| Qwen | full MIND | 0.884625 | 0.315199 |
| Qwen | drift-only | 0.815491 | 0.118222 |
| Qwen | no-manifold | 0.797421 | 0.158706 |
| Qwen | linear probe | 0.924120 | 0.479551 |
| InternVL | full MIND | 0.904515 | 0.548230 |
| InternVL | drift-only | 0.879535 | 0.422068 |
| InternVL | no-manifold | 0.849947 | 0.423549 |
| InternVL | linear probe | 0.908870 | 0.625761 |

Takeaways:

- The corrected grouped results do not collapse to random.
- Qwen actually improves in ROC-AUC under `image_grouped` relative to corrected `row`, which supports the claim that the earlier weakness came from the signal definition more than from the stricter protocol itself.
- InternVL changes little between corrected `row` and corrected `image_grouped`, which is a good sign for stability.

## Secondary Check: `object_heldout`

`object_heldout` at `5` folds was not class-feasible on these corrected popular labels. The largest shared valid setting across both model families was `2` folds, so that is the protocol used here.

### Qwen Popular, `object_heldout`

| Variant | ROC-AUC | PR-AUC | TPR@1%FPR |
| --- | ---: | ---: | ---: |
| full MIND | 0.724419 | 0.063808 | 0.000000 |
| drift-only | 0.780095 | 0.080777 | 0.027027 |
| no-manifold | 0.656304 | 0.072021 | 0.036036 |
| linear probe | 0.743235 | 0.123253 | 0.036036 |

Takeaway:

- Qwen does not hold up well under object-held-out transfer.

### InternVL Popular, `object_heldout`

| Variant | ROC-AUC | PR-AUC | TPR@1%FPR |
| --- | ---: | ---: | ---: |
| full MIND | 0.839763 | 0.409716 | 0.218750 |
| drift-only | 0.834840 | 0.305181 | 0.082031 |
| no-manifold | 0.776749 | 0.266311 | 0.078125 |
| linear probe | 0.782977 | 0.270688 | 0.082031 |

Takeaway:

- InternVL holds up much better than Qwen under object-held-out transfer.
- On this secondary protocol, full MIND even beats the InternVL linear probe.

## Reference-Bank Cleaning Result

Cleaning was strict:

- kept only `parsed_answer == 1`
- dropped all `parsed_answer == 0`
- dropped unparsed rows

Corrected reference support at `k_neighbors = 32`:

| Model | Kept reference rows | Supported object-layer rows | Low-support rows |
| --- | ---: | ---: | ---: |
| Qwen | 4652 / 5056 | 1264 / 1264 | 0 |
| InternVL | 4842 / 5056 | 1264 / 1264 | 0 |

That means the cleaned-bank correction did not create a support cliff on this dataset.

## Paper Decision Gate

Current decision:

- keep the project alive
- stop short of claiming strongest overall detector
- center the paper on low-dimensional geometry-aware early warning

Reason:

- The corrected signal clearly beats corrected drift-only and corrected no-manifold on the primary grouped protocol for both model families.
- The linear probe still keeps a clear PR-AUC advantage on the primary grouped protocol for both model families.
- The signal is therefore real and useful, but the best paper position is interpretability, compression, calibration, and cross-model behavior rather than pure leaderboard superiority.

## Historical Note

Older Qwen runs on POPE `random`, `popular`, and `adversarial`, plus the earlier layer-range ablation, remain in the repo and journal as historical context. They are still useful for understanding the late-layer pattern, but the correction-phase popular runs above are the current source of truth for the paper decision.
