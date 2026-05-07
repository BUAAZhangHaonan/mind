# MIND

`master` is the MIND v2 Stage 0 line.

MIND v2 starts from audited multimodal records, deterministic grouped splits, and full-layer hidden-state cache shards. The old v1 research path is preserved on the local branch `v1` and tag `v1-freeze-before-v2` at commit `81e3444`; it is not the main path on `master`.

## Scope

- Stage 0 data audit.
- Stage 0 grouped split manifests.
- Stage 0 full-layer cache extraction and validation.
- Later Stage A-E work starts from the Stage 0 cache contract.

## Kept Surface

```text
configs/models/
configs/v2/
docs/v2/
scripts/v2/
scripts/verify_env.py
src/mind/cache/
src/mind/config/
src/mind/data/
src/mind/evaluation/
src/mind/extractors/
src/mind/models/
src/mind/trajectory/
src/mind/utils/
tests/v2/
```

## Environment

The project environment name is `mind-py311`. The documented command path uses the shipped `Makefile`.

```bash
make env
make verify-env
make verify-model MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
make verify-model MODEL_ID=OpenGVLab/InternVL3_5-8B-HF
make test
```

If Hugging Face access is slow, export `HF_ENDPOINT=https://hf-mirror.com`.

## Stage 0

Run the Stage 0 smoke dry run:

```bash
make plan-smoke
```

Run Stage 0 directly:

```bash
conda run --no-capture-output -n mind-py311 python scripts/v2/stage0_run.py \
  --models qwen3-vl-8b \
  --datasets pope \
  --subsets popular \
  --smoke-limit 8
```

Stage 0 writes under `outputs/v2_stage0` by default. Existing output artifacts are retained as artifacts, not as active master code.
