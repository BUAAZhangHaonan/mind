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
