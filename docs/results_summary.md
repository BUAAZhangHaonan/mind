# MIND Results Summary

## Runtime State

- Machine used for the completed runs: `3 x RTX 3090 24GB`
- One fourth card remains faulty and is still not visible to `nvidia-smi` or PyTorch.
- Canonical environment: `mind-py311`
- Hugging Face mirror used when needed: `HF_ENDPOINT=https://hf-mirror.com`

## Completed Qwen Runs

### Smoke Stage

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Data: `200` examples from POPE popular
- Purpose: verify the full path from data loading to plots

Smoke result:

- detector ROC-AUC: `0.4237`
- note: the held-out fold contained only `1` hallucination positive, so this run only validates the pipeline and does not support a meaningful detector claim

### Main Qwen Summary

Model: `Qwen/Qwen3-VL-8B-Instruct`

| Subset | MIND ROC-AUC | RePOPE ROC-AUC | Raw yes/no accuracy | Linear probe ROC-AUC | Drift-only ROC-AUC | No-manifold ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| popular | 0.673692 | 0.644327 | 0.889000 | 0.924120 | 0.665059 | 0.611303 |
| random | 0.581979 | 0.640958 | 0.916667 | 0.814462 | 0.569647 | 0.544002 |
| adversarial | 0.707938 | 0.704809 | 0.869000 | 0.783459 | 0.696552 | 0.671147 |

Observations:

- MIND produced a usable ranking signal on all three POPE subsets.
- RePOPE lowered the popular ROC-AUC but raised the random ROC-AUC relative to the original POPE labels.
- In this implementation, the direct hidden-state linear probe was stronger than MIND on all finished Qwen runs.

### Popular Ablations

Model: `Qwen/Qwen3-VL-8B-Instruct`

| Variant | ROC-AUC |
| --- | ---: |
| early layers | 0.569641 |
| middle layers | 0.673692 |
| late layers | 0.861207 |

Interpretation:

- The corrected popular run supports `late > middle > early`, not the earlier middle-layer story.
- Wavelet features helped slightly over raw drift on popular.
- The manifold term helped on popular relative to the no-manifold variant.

## Cross-Family InternVL Result

Model: `OpenGVLab/InternVL3_5-8B-HF`

| Setting | MIND ROC-AUC | Accuracy | F1 | Raw yes/no accuracy | Linear probe ROC-AUC | Drift-only ROC-AUC | No-manifold ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POPE popular | 0.836676 | 0.912222 | 0.024691 | 0.873000 | 0.908870 | 0.835256 | 0.723454 |
| RePOPE popular | 0.811205 | 0.898889 | 0.021505 | - | - | - | - |

Observations:

- InternVL was clearly stronger than Qwen on the popular split in ROC-AUC.
- Within the InternVL run, MIND only slightly improved over drift-only features.
- The direct hidden-state linear probe was still stronger than MIND, so the current evidence supports MIND more as an interpretable geometry-aware detector than as the strongest raw predictor in this codebase.

Output roots:

- `outputs/reports/cross-internvl3.5-8b-popular/`
- `outputs/reports/cross-internvl3.5-8b-popular-repope/`
- `outputs/plots/cross-internvl3.5-8b-popular/`

## H-POPE Status

- Loader and config surface are present in the repo.
- Public benchmark assets were not found in a directly usable downloadable package during this execution window.
- H-POPE remains documented as blocked until a public asset release is actually reachable.
