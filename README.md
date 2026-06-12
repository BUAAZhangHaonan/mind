# MIND

`master` contains the completed Stage 0 line, the Experiment 2 full-cache asset surface, and the frozen Stage A closeout line.

For local model asset registry and smoke validation details, see `docs/ASSET_REGISTRY.md`.

MIND starts from audited multimodal records, deterministic grouped splits, and full-layer hidden-state cache shards. Stage A uses those cached tensors to test representation hypotheses only. It does not validate the final MIND detector.

## Frozen Theory Note

### MIND Frozen Note v1: Hyperspherical Sequential Metric Learning for Pre-generation Support Estimation

MIND does not study how to build a stronger object hallucination classifier. It studies how to model VLM pre-generation hidden states as an interpretable, comparable representation space that supports inlier support estimation. Object hallucination detection is then written as support estimation on that space, not as ordinary classification.

Given one sample `x`, MIND observes the final prefill token hidden states from all layers:

```text
H(x) = (h_1, ..., h_L), h_l in R^d
```

Each layer vector is L2-normalized:

```text
u_l = h_l / ||h_l||_2, u_l in S^(d-1)
```

The basic object is the pre-generation semantic trajectory:

```text
T(x) = (u_1, ..., u_L) in (S^(d-1))^L
```

So MIND is built around a product-of-spheres trajectory, not a single hidden-state point. This matches the full-layer prefill cache surface built in Experiment 2.

The default representation map is:

```text
z = f_theta(T(x)), z in S^(m-1)
```

The current engineering default can use an LSTM sequence encoder, but the method is not the LSTM itself. Stage A showed that classifier-friendly embeddings can be strong while bank geometry can remain weak. This means BCE or CE can expose signal without learning an embedding that is well aligned with support estimation. BCE and CE are therefore baselines only.

The only main objective families after Stage A are supervised contrastive learning, Proxy Anchor, and angular-margin objectives. The detector head must be described as support estimation on hyperspherical embeddings. A non-parametric head can use a geodesic radius ball, and a parametric head can use vMF support. `radius_ball` is one support estimator, not the core contribution.

Frozen kNN is a signal probe, not a one-class method. Deep one-class methods need their own compactness or descriptiveness objective, and MIND must not rename frozen kNN as one-class learning.

Mainline methods must use pre-generation full-layer trajectories, hyperspherical or manifold-aware representation space, metric-aligned representation learning, and support estimation heads. Euclidean classifiers, BCE-only sequence encoders, frozen kNN, raw static probes, HALP, and linear probes remain baselines.

The frozen stage order is:

- Stage A: close representation pretests by comparing `Sphere-Traj-LSTM` with `Raw-Traj-LSTM`.
- Stage B: test metric-aligned objectives, measure negative-budget efficiency on the selected objective, check kNN scale robustness, and compare parametric hyperspherical support diagnostics.
- Stage C: compare support estimators on frozen hyperspherical embeddings.
- Stage D: test cross-domain and domain-expansion behavior.

Stage A is closed after Raw-Traj-LSTM is added and the closeout summary is written. Later stages must not reopen Stage A except if this frozen theory note is explicitly revised.

Background references: Hyperspherical Prototype Networks [1], ArcFace [2], Hyperspherical VAE [3], Supervised Contrastive Learning [4], Proxy Anchor [5], probabilistic hyperspherical contrastive learning [6], Deep One-Class [7], Deep SVDD [8], DROCC [9], HALP [10], and Riemannian adaptive optimization [11].

## Scope

- Stage 0 data audit.
- Stage 0 grouped split manifests.
- Stage 0 full-layer cache extraction and validation.
- Experiment 2 full-cache unified manifest under `outputs/full_cache/manifests/`.
- Stage A closeout diagnostics from `outputs/full_cache` to `outputs/stageA_closeout`.
- Stage B1 metric-objective diagnostics from `outputs/full_cache` to `outputs/stageB`.
- Stage B2 Proxy Anchor negative-budget diagnostics from `outputs/full_cache` to `outputs/stageB2`.
- Stage B3 Proxy Anchor kNN scale-robustness diagnostics from `outputs/full_cache` to `outputs/stageB3`.
- Stage B4 Proxy Anchor vMF support-family diagnostics from `outputs/full_cache` to `outputs/stageB4`.

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
tests/full_cache/
tests/stage_a_closeout/
tests/stage_b/
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

Legacy Stage A v1 used `outputs/stageA`. The closeout run uses the Experiment 2 unified full-cache manifest and writes to `outputs/stageA_closeout`.

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_a_closeout_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageA_closeout \
  --bootstrap 1000 \
  --device cuda:0 \
  --lstm-epochs 10
```

Stage A closeout evaluates RePOPE as the primary closeout dataset, with POPE and DASH-B as secondary readouts. Stage B1 starts after this closeout and does not reopen Stage A.

## Stage B1

Stage B compares metric-aligned objectives on the frozen hyperspherical trajectory representation. It does not choose the final detector.

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B1 keeps the object, space, and encoder family fixed. It compares `bce`, `supcon`, and `proxy_anchor` with auto-tuned geodesic kNN as the primary geometry diagnostic, vMF prototype as a secondary diagnostic, and classifier readout as a control. Stage C has not started.

## Stage B2

Stage B2 freezes `Proxy Anchor` and varies only the hard-hallucination negative budget.

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b2_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB2 \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B2 uses ratios `1.00`, `0.50`, `0.25`, and `0.10`, with seeds `20260506`, `20260507`, and `20260508`. Auto-tuned geodesic kNN is the primary geometry diagnostic. The classifier readout is a control, and the single-vMF prototype is a tertiary diagnostic. Stage B2 does not choose the final detector, and Stage C has not started.

## Stage B3

Stage B3 freezes `Sphere-Traj-LSTM + Proxy Anchor` at the 50% hard-negative budget and checks whether the geodesic kNN signal is stable across local neighborhood scales.

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b3_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB3 \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B3 evaluates the fixed k grid `{1, 2, 4, 8, 16, 32, 64}` after bank-size clipping. It reports selected `k`, scale-grid curves, stability bands, classifier control, and a single-vMF probe. It does not choose the final detector, and Stage C has not started.

## Stage B4

Stage B4 freezes `Sphere-Traj-LSTM + Proxy Anchor` at the 50% hard-negative budget and compares support-family diagnostics on the frozen hyperspherical embedding.

```bash
conda run --no-capture-output -n mind-py311 python scripts/stage_b4_run.py \
  --full-cache-root outputs/full_cache \
  --output-root outputs/stageB4 \
  --bootstrap 1000 \
  --epochs 20 \
  --device cuda:0
```

Stage B4 keeps kNN as the nonparametric reference and evaluates single-vMF plus mixture-vMF as parametric hyperspherical support diagnostics. Mixture `K` is selected on RePOPE calibration rows only from `{1, 2, 4, 8}`. The classifier readout is a control. Stage B4 does not choose the final detector, and Stage C has not started.

[1]: https://arxiv.org/abs/1901.10514
[2]: https://arxiv.org/abs/1801.07698
[3]: https://arxiv.org/abs/1804.00891
[4]: https://arxiv.org/abs/2004.11362
[5]: https://arxiv.org/abs/2003.13911
[6]: https://arxiv.org/abs/2405.16460
[7]: https://arxiv.org/abs/1801.05365
[8]: https://arxiv.org/abs/2001.08873
[9]: https://arxiv.org/abs/2002.12718
[10]: https://arxiv.org/abs/2603.05465
[11]: https://arxiv.org/abs/1810.00760
