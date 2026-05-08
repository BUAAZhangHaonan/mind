# MIND

`master` contains the completed Stage 0 line and the Stage A representation acid test.

MIND starts from audited multimodal records, deterministic grouped splits, and full-layer hidden-state cache shards. Stage A uses those cached tensors to test representation hypotheses only. It does not validate the final MIND detector.

## Scope

- Stage 0 data audit.
- Stage 0 grouped split manifests.
- Stage 0 full-layer cache extraction and validation.
- Stage A representation-space diagnostics from `outputs/stage0` to `outputs/stageA`.
- Stage B has not started.

## Kept Surface

```text
configs/models/
configs/stage0/
configs/stageA/
docs/
scripts/
scripts/verify_env.py
src/mind/cache/
src/mind/config/
src/mind/data/
src/mind/evaluation/
src/mind/extractors/
src/mind/models/
src/mind/trajectory/
src/mind/utils/
tests/stage0/
tests/stage_a/
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
conda run --no-capture-output -n mind-py311 python scripts/stage0_run.py \
  --models qwen3-vl-8b \
  --datasets pope \
  --subsets popular \
  --smoke-limit 8
```

Stage 0 writes under `outputs/stage0` by default. Existing output artifacts are retained as artifacts, not as active master code.

## Stage A

Run a Stage A dry run:

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_a_run.py \
  --stage0-root outputs/stage0 \
  --output-root outputs/stageA \
  --models qwen3-vl-8b \
  --subsets popular random adversarial \
  --device cuda:0 \
  --dry-run
```

Stage A may conclude that only multi-layer aggregation is useful while layer order remains unproven. DASH-B and RePOPE are checked for Stage 0 completion but are not Stage A primary inputs.
