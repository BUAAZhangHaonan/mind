# Sub-task 2 Anisotropic Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test local anisotropic MIND scoring with locked radius-ball neighbors, then decide whether local covariance geometry adds real value over the Sub-task 1 isotropic baseline.

**Architecture:** Reuse existing float16 eval caches and pooled reference banks. Tune the locked Sub-task 1 radius-ball rule once per model, benchmark, and layer, then score each query with radius_ball_isotropic, diag_maha, lowrank_maha, and full_maha_shrink. Store features and metrics under `outputs/subtask2_anisotropic/`, and write the final report tables in `docs/tables/` plus the decision memo in `docs/review/`.

**Tech Stack:** PyTorch CUDA on GPU 0, pandas/parquet, existing `train_gpu_detector.evaluate_frame`, existing `mind.geometry.gpu_distances`, existing Sub-task 1 neighbor/radius helpers, tmux session `mind_subtask2_anisotropic`.

---

## Locked Decisions

- `radius_ball` from Sub-task 1 is locked. Do not rerun the five-method neighbor-selection comparison.
- Radius tuning uses the same rule as Sub-task 1: `target_count=30`, `reference_chunk_size=16384`, `binary_steps=32`, and membership margin `2e-5`.
- The radius is fixed per `(model, benchmark, layer)`. It is not per query, not one global radius, and not one radius per model.
- All long runs use `CUDA_VISIBLE_DEVICES=0` and `--device cuda:0`.
- Existing caches only. Do not regenerate eval caches or reference caches.
- Stay on `master`. Do not create a branch or use `.worktree`.
- Commit and push after each implementation milestone. Stage only intended files.

## Known Inputs

Use these eval caches and banks:

| model | benchmark | eval cache | pooled bank |
|---|---|---|---|
| `qwen3-vl-8b` | `popular` | `outputs/round2_2026_04/cache/qwen3-vl-8b/pope/popular` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/popular/qwen3-vl-8b` |
| `qwen3-vl-8b` | `dash-b` | `outputs/decisive_round_2026_04/cache/qwen3-vl-8b/dash-b/main` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/dash-b/qwen3-vl-8b` |
| `internvl3.5-8b` | `popular` | `outputs/round2_2026_04/cache/internvl3.5-8b/pope/popular` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/popular/internvl3.5-8b` |
| `internvl3.5-8b` | `dash-b` | `outputs/round2_2026_04/cache/internvl3.5-8b/dash-b/main` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/dash-b/internvl3.5-8b` |
| `llava-onevision-7b` | `popular` | `outputs/round2_2026_04/cache/llava-onevision-7b/pope/popular` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/popular/llava-onevision-7b` |
| `llava-onevision-7b` | `dash-b` | `outputs/round2_2026_04/cache/llava-onevision-7b/dash-b/main` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/dash-b/llava-onevision-7b` |
| `molmo-7b-d-0924` | `popular` | `outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/popular` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/popular/molmo-7b-d-0924` |
| `molmo-7b-d-0924` | `dash-b` | `outputs/decisive_round_2026_04/cache/molmo-7b-d-0924/dash-b/main` | `outputs/gpu_round_2026_04/query_local_bank/pooled_banks/dash-b/molmo-7b-d-0924` |

Reference baseline sources:

- Main image-grouped baselines: `docs/tables/experiment_query_local_bank.csv`
- Direct JSON baselines:
  - `outputs/gpu_round_2026_04/bank_identity/metrics/{model}/{benchmark}/object_conditioned/no_manifold.json`
  - `outputs/gpu_round_2026_04/query_local_bank/metrics/{model}/{benchmark}/linear_probe.json`
  - `outputs/gpu_round_2026_04/query_local_bank/metrics/{model}/{benchmark}/query_local_k30.json`
- Heldout linear-probe table: `docs/tables/table3_transfer_controls.md`
- Heldout feature files:
  - `outputs/round2_2026_04/features/round2-qwen3-vl-8b-popular-object-heldout-refresh/popular.parquet`
  - `outputs/round2_2026_04/features/round2-internvl3.5-8b-popular-object-heldout-refresh/popular.parquet`
  - `outputs/round2_2026_04/features/round2-llava-onevision-7b-popular-object-heldout-refresh/popular.parquet`
  - `outputs/round2_2026_04/features/round2-molmo-7b-d-0924-popular-object-heldout-refresh/popular.parquet`

## Output Contract

Create these files:

- `src/mind/geometry/gpu_anisotropic.py`
- `tests/unit/test_gpu_anisotropic.py`
- `scripts/experiments/anisotropic_scoring_comparison.py`
- `docs/review/subtask2_anisotropic_analysis.md`
- `docs/tables/subtask2_anisotropic_comparison.md`
- `docs/tables/subtask2_anisotropic_comparison.csv`
- `docs/tables/subtask2_heldout_transfer.md`
- `docs/tables/subtask2_heldout_transfer.csv`

Write generated artifacts under:

- `outputs/subtask2_anisotropic/radii/`
- `outputs/subtask2_anisotropic/features/`
- `outputs/subtask2_anisotropic/metrics/`
- `outputs/subtask2_anisotropic/predictions/`
- `outputs/subtask2_anisotropic/logs/`

Do not commit generated `outputs/` files unless the repository already tracks the exact requested artifact. Commit source, tests, and docs only.

## Completion Criteria

- Unit tests pass for GPU anisotropic scoring on GPU 0.
- Smoke evaluation passes on a small row limit for all four methods.
- Full image-grouped evaluation has 8 model-benchmark rows and 4 method columns.
- Heldout transfer evaluation has 4 model rows for the best DASH-B anisotropic variant.
- Every metric cell includes ROC-AUC CI and PR-AUC CI.
- The analysis document answers all seven requested questions and states the decision gate result without hedging.
- Radius values are saved in JSON and summarized in the analysis document.
- `tmux` evidence shows the full run used session `mind_subtask2_anisotropic`.
- Final disk usage remains below 1 TB.
- Final commit `Complete Sub-task 2: local anisotropic metrics with radius-ball neighbors` is pushed to `origin/master`.

## Delegation Map

Use SubAgents in batches of 1-3:

1. Runner/API review agent: checks call sites and confirms imports, CLI shape, and baseline paths.
2. Math/test agent: reviews Mahalanobis formulas and writes CPU reference expectations for tests.
3. Report agent: checks table schemas and analysis-question coverage.

The main executor should synthesize results, make final edits, run validation, commit, and push.

---

### Task 1: Preflight State And Locked Inputs

**Files:**
- Read: `docs/review/subtask1_neighbor_selection_analysis.md`
- Read: `docs/tables/subtask1_neighbor_comparison.csv`
- Read: `scripts/experiments/neighbor_selection_comparison.py`
- Read: `src/mind/geometry/neighbor_selection.py`
- Read: `src/mind/geometry/gpu_distances.py`

**Step 1: Check git state**

Run:

```bash
git status --short --branch
```

Expected:

- Branch is `master`.
- Existing unrelated untracked files, such as `docs/reference.md`, are not staged.

**Step 2: Confirm GPU 0 visibility**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
PY
```

Expected:

- CUDA is available.
- Device count is `1`.
- Visible device is the physical GPU 0 due to `CUDA_VISIBLE_DEVICES=0`.

**Step 3: Verify cache paths exist without regenerating**

Run:

```bash
for path in \
  outputs/round2_2026_04/cache/qwen3-vl-8b/pope/popular \
  outputs/decisive_round_2026_04/cache/qwen3-vl-8b/dash-b/main \
  outputs/round2_2026_04/cache/internvl3.5-8b/pope/popular \
  outputs/round2_2026_04/cache/internvl3.5-8b/dash-b/main \
  outputs/round2_2026_04/cache/llava-onevision-7b/pope/popular \
  outputs/round2_2026_04/cache/llava-onevision-7b/dash-b/main \
  outputs/round2_2026_04/cache/molmo-7b-d-0924/pope/popular \
  outputs/decisive_round_2026_04/cache/molmo-7b-d-0924/dash-b/main \
  outputs/gpu_round_2026_04/query_local_bank/pooled_banks/popular \
  outputs/gpu_round_2026_04/query_local_bank/pooled_banks/dash-b
do
  test -e "$path" || { echo "missing $path"; exit 1; }
done
```

Expected:

- No missing path is printed.

**Step 4: Record disk baseline**

Run:

```bash
du -sh outputs
df -h .
```

Expected:

- `outputs` remains far below 1 TB.

### Task 2: Write Failing Tests For GPU Anisotropic Scoring

**Files:**
- Create: `tests/unit/test_gpu_anisotropic.py`

**Step 1: Add GPU-only test helpers**

Create tests that:

- Skip if CUDA is unavailable.
- Always create CUDA tensors after `CUDA_VISIBLE_DEVICES=0`.
- Use small tensors, for example `k=6`, `d=5`, and `n_query=2`.
- Include CPU reference code inside the tests only.

**Step 2: Add diagonal Mahalanobis test**

Test:

```python
score = score_diag_maha_gpu(query, neighbors, eps=1e-6)
```

Expected:

- It equals `sqrt(sum((query - mean)^2 / (var + eps)))`.
- Variance uses `unbiased=False`.

**Step 3: Add low-rank Mahalanobis test**

Test:

```python
score = score_lowrank_maha_gpu(query, neighbors, rank_cap=3, eps=1e-6)
```

Expected:

- It matches a CPU SVD reference.
- `r = min(rank_cap, k - 1, d)`.
- Residual variance is computed from remaining covariance trace and clamped by `eps`.

**Step 4: Add full shrinkage test**

Test:

```python
score = score_full_maha_shrink_gpu(query, neighbors, eps=1e-6)
```

Expected:

- It matches an explicit small `d x d` shrunk covariance solve:

```python
cov = centered.T @ centered / (k - 1)
tau = torch.trace(cov) / d
shrunk = (1 - alpha) * cov + alpha * tau * torch.eye(d)
solved = torch.linalg.solve(shrunk + eps * torch.eye(d), delta)
expected = torch.sqrt(delta @ solved)
```

- The production function may use Woodbury, but the result must match this explicit reference.

**Step 5: Add radius-ball anisotropic feature test**

Test:

```python
row = compute_anisotropic_feature_row_gpu(...)
```

Expected:

- It returns 16 raw drift columns plus five calibrated columns.
- `radius_ball_isotropic` matches mean angular distance for the selected neighbors.
- All values are finite.

**Step 6: Run tests and confirm failure**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/unit/test_gpu_anisotropic.py
```

Expected:

- Tests fail because `src/mind/geometry/gpu_anisotropic.py` does not exist yet.

### Task 3: Implement GPU Anisotropic Scoring Primitives

**Files:**
- Create: `src/mind/geometry/gpu_anisotropic.py`

**Step 1: Add public constants and validation**

Implement:

```python
ANISOTROPIC_METHODS = (
    "radius_ball_isotropic",
    "diag_maha",
    "lowrank_maha",
    "full_maha_shrink",
)
DEFAULT_EPS = 1e-6
DEFAULT_RANK_CAP = 8
```

Validation rules:

- Production inputs must be CUDA tensors.
- Inputs must be rank-1 query vectors or rank-2 query matrices as documented.
- Neighbor matrices must be floating point and non-empty.
- Query and neighbor dimensions must match.

**Step 2: Implement radius-ball neighbor selection**

Implement:

```python
@dataclass(frozen=True)
class RadiusBallNeighborhood:
    indices: torch.Tensor
    distances: torch.Tensor
    vectors: torch.Tensor

def select_radius_ball_neighbors_gpu(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    radius: torch.Tensor | float,
    reference_chunk_size: int = 16_384,
    radius_margin: float = DEFAULT_RADIUS_MARGIN,
) -> list[RadiusBallNeighborhood]:
    ...
```

Rules:

- Use `mind.geometry.gpu_distances.batch_angular_distance` where possible.
- Keep all distance computation on GPU.
- Process reference in chunks.
- Return ragged CUDA tensors per query row.
- Raise `RuntimeError` if any query gets zero neighbors. Do not fall back to kNN.

**Step 3: Implement `score_diag_maha_gpu`**

Formula:

```python
mu = neighbors.mean(dim=0)
var = neighbors.var(dim=0, unbiased=False)
delta = query - mu
score = torch.sqrt((delta.square() / (var + eps)).sum())
```

**Step 4: Implement `score_lowrank_maha_gpu`**

Formula:

```python
centered = neighbors - neighbors.mean(dim=0)
u, s, vh = torch.linalg.svd(centered, full_matrices=False)
r = min(rank_cap, neighbors.shape[0] - 1, neighbors.shape[1], s.numel())
eig = s[:r].square() / max(neighbors.shape[0] - 1, 1)
proj = delta @ vh[:r].T
in_score = (proj.square() / (eig + eps)).sum()
residual = delta - proj @ vh[:r]
total_var = centered.square().sum() / max(neighbors.shape[0] - 1, 1)
residual_var = (total_var - eig.sum()).clamp_min(0.0) / max(neighbors.shape[1] - r, 1)
score = torch.sqrt(in_score + residual.square().sum() / (residual_var + eps))
```

**Step 5: Implement Ledoit-Wolf shrinkage helper**

Implement:

```python
def ledoit_wolf_shrinkage_alpha_gpu(centered: torch.Tensor) -> torch.Tensor:
    ...
```

Rules:

- Port the standard analytic Ledoit-Wolf shrinkage formula to torch.
- Clamp only to the valid analytic range `[0, 1]`.
- Add tests against an explicit small-tensor reference.

**Step 6: Implement `score_full_maha_shrink_gpu`**

Production formula:

- Define `cov = centered.T @ centered / (k - 1)`.
- Define `tau = trace(cov) / d`.
- Define `lambda = alpha * tau + eps`.
- Use the exact Woodbury form for `(lambda I + c X^T X)^(-1)` with `torch.linalg.solve` on the small `k x k` system.
- Do not materialize a `4096 x 4096` covariance in production.
- Include a private small-test helper that materializes the shrunk covariance for unit tests.

Reason:

- This is mathematically the same full shrunk covariance score.
- It avoids thousands of dense `d x d` solves during the full run.

**Step 7: Implement feature-row builder**

Implement:

```python
def compute_anisotropic_feature_row_gpu(
    *,
    layer_vectors: torch.Tensor,
    selected_layers: Sequence[int],
    reference_layers: Mapping[int, torch.Tensor],
    method: str,
    layer_radii: Mapping[int, torch.Tensor | float],
    reference_chunk_size: int = 16_384,
    eps: float = 1e-6,
    rank_cap: int = 8,
) -> dict[str, float]:
    ...
```

Rules:

- Produce `raw_drift_0` through `raw_drift_15` where the model has 16 selected layers.
- For LLaVA/Molmo, produce raw columns for their selected layer count and let the runner create a stable feature schema based on present raw columns.
- Add calibrated columns:
  - `cal_mean_drift`
  - `cal_max_drift`
  - `cal_final_drift`
  - `cal_drift_slope`
  - `cal_drift_variance`
- Reuse the slope logic from `src/mind/geometry/neighbor_selection.py`.

**Step 8: Run tests**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/unit/test_gpu_anisotropic.py
```

Expected:

- All new tests pass.

**Step 9: Run focused existing tests**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/unit/test_neighbor_selection.py tests/unit/test_gpu_anisotropic.py
```

Expected:

- All tests pass.

**Step 10: Commit and push milestone**

Run:

```bash
git add src/mind/geometry/gpu_anisotropic.py tests/unit/test_gpu_anisotropic.py
git commit -m "Add GPU anisotropic scoring primitives"
git push origin master
```

### Task 4: Implement Sub-task 2 Runner

**Files:**
- Create: `scripts/experiments/anisotropic_scoring_comparison.py`

**Step 1: Add CLI commands**

Implement commands:

```bash
python scripts/experiments/anisotropic_scoring_comparison.py run-one ...
python scripts/experiments/anisotropic_scoring_comparison.py run-suite ...
python scripts/experiments/anisotropic_scoring_comparison.py format ...
```

`run-one` arguments:

- `--cache-path`
- `--pooled-bank-root`
- `--model-name`
- `--benchmark`
- `--split-strategy image_grouped|object_heldout`
- `--methods all|radius_ball_isotropic,diag_maha,lowrank_maha,full_maha_shrink`
- `--device cuda:0`
- `--target-count 30`
- `--reference-chunk-size 16384`
- `--bootstrap-resamples 1000`
- `--num-folds 5`
- `--random-state 13`
- `--max-iter 100`
- `--limit-rows`
- `--output-root outputs/subtask2_anisotropic`

`run-suite` arguments:

- Same eval settings.
- No per-path arguments. It uses the locked path table in this plan.

`format` arguments:

- `--metrics-csv outputs/subtask2_anisotropic/metrics/subtask2_anisotropic_metrics.csv`
- `--heldout-csv outputs/subtask2_anisotropic/metrics/subtask2_heldout_metrics.csv`
- `--output-comparison-md docs/tables/subtask2_anisotropic_comparison.md`
- `--output-comparison-csv docs/tables/subtask2_anisotropic_comparison.csv`
- `--output-heldout-md docs/tables/subtask2_heldout_transfer.md`
- `--output-heldout-csv docs/tables/subtask2_heldout_transfer.csv`
- `--output-analysis docs/review/subtask2_anisotropic_analysis.md`

**Step 2: Add CUDA guard**

Reuse the Sub-task 1 guard:

```python
if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
    raise ValueError("CUDA_VISIBLE_DEVICES must be '0' when set.")
```

Also reject any device except `cuda` or `cuda:0`.

**Step 3: Load caches and pooled banks**

Reuse:

- `mind.evaluation.baselines.load_cache_entries`
- `load_pooled_reference_layers` from `scripts/experiments/neighbor_selection_comparison.py`
- `metadata_row_from_entry`
- `feature_columns`

Keep fields:

```python
{
  "sample_id",
  "image_id",
  "label",
  "parsed_answer",
  "subset",
  "object_name",
  "selected_layers",
  "layer_vectors",
}
```

**Step 4: Tune and save radius manifest**

Reuse `tune_radius_ball_layer_radii_gpu`.

Write:

```text
outputs/subtask2_anisotropic/radii/{model}_{benchmark}_target30.json
```

JSON fields:

- `model`
- `benchmark`
- `target_count`
- `reference_chunk_size`
- `radius_margin`
- `radii_by_layer`
- `mean_neighbor_count_by_layer`
- `min_neighbor_count_by_layer`
- `max_neighbor_count_by_layer`
- `cache_path`
- `pooled_bank_root`

**Step 5: Build feature frames**

For each method:

- Build one parquet:

```text
outputs/subtask2_anisotropic/features/{model}/{benchmark}/{split_strategy}/{method}.parquet
```

- Include metadata columns:
  - `sample_id`
  - `image_id`
  - `ground_truth_label`
  - `answer_label`
  - `label`
  - `subset`
  - `object_name`
- Include raw and calibrated feature columns.

**Step 6: Evaluate feature frames**

Call `train_gpu_detector.evaluate_frame` directly.

For main comparison:

- `split_strategy="image_grouped"`
- 5 folds
- 1000 bootstrap samples

For heldout:

- `split_strategy="object_heldout"`
- Use only the best anisotropic method by mean DASH-B PR-AUC.

**Step 7: Load baseline cells**

For Table 1:

- Load `no_manifold` and `linear_probe` from `docs/tables/experiment_query_local_bank.csv`.
- Keep `query_local_k30` available for analysis text only.

For Table 2:

- Load saved `linear_probe` heldout metrics from `docs/tables/table3_transfer_controls.md` when possible.
- If complete heldout `no_manifold` or `query_local_k30` metrics are not present, re-evaluate existing feature parquets with `split_strategy=object_heldout`. This is evaluation only, not cache regeneration.

**Step 8: Format docs**

Write Table 1:

```text
docs/tables/subtask2_anisotropic_comparison.md
docs/tables/subtask2_anisotropic_comparison.csv
```

Columns:

- `model`
- `benchmark`
- `radius_ball_isotropic`
- `diag_maha`
- `lowrank_maha`
- `full_maha_shrink`
- `no_manifold`
- `linear_probe`

Write Table 2:

```text
docs/tables/subtask2_heldout_transfer.md
docs/tables/subtask2_heldout_transfer.csv
```

Columns:

- `model`
- `best_anisotropic`
- `no_manifold`
- `linear_probe`

Each metric cell:

```text
ROC-AUC 0.xxxx [0.xxxx, 0.xxxx]; PR-AUC 0.xxxx [0.xxxx, 0.xxxx]
```

**Step 9: Write analysis document**

Write:

```text
docs/review/subtask2_anisotropic_analysis.md
```

It must answer:

1. Does any anisotropic variant beat radius_ball isotropic on mean PR-AUC across the 8 model-benchmark settings?
2. Does the best anisotropic variant narrow the gap to linear_probe? Report the absolute PR-AUC gap for each model and benchmark.
3. On DASH-B, does anisotropic scoring beat no_manifold on all four models?
4. Is the anisotropic advantage larger on DASH-B or POPE popular?
5. On object_heldout, does the best anisotropic variant beat linear_probe? If not, does it narrow the gap relative to old query_local_k30 heldout results?
6. Which anisotropic variant works best, and what does this imply about local grounded covariance?
7. Is there model-specific covariance evidence?

Apply the decision gate:

- If best anisotropic narrows the linear-probe gap by at least `0.05` PR-AUC on DASH-B or heldout, state: continue the method paper.
- If gains over isotropic are below `0.02` PR-AUC, state: converge the method here.
- If anisotropic beats isotropic but loses badly to linear_probe on heldout, state: reframe around compact grounded detection.

Do not hedge.

**Step 10: Smoke run**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/anisotropic_scoring_comparison.py run-one \
  --cache-path outputs/round2_2026_04/cache/qwen3-vl-8b/pope/popular \
  --pooled-bank-root outputs/gpu_round_2026_04/query_local_bank/pooled_banks/popular \
  --model-name qwen3-vl-8b \
  --benchmark popular \
  --split-strategy image_grouped \
  --methods all \
  --device cuda:0 \
  --target-count 30 \
  --reference-chunk-size 16384 \
  --bootstrap-resamples 10 \
  --num-folds 2 \
  --limit-rows 64 \
  --output-root outputs/subtask2_anisotropic/smoke
```

Expected:

- Four method rows are written.
- Feature parquets are readable.
- No CPU fallback is used.

**Step 11: Commit and push milestone**

Run:

```bash
git add scripts/experiments/anisotropic_scoring_comparison.py
git commit -m "Add Sub-task 2 anisotropic evaluation runner"
git push origin master
```

### Task 5: Run Full Image-Grouped Evaluation In tmux

**Files:**
- Create generated script: `outputs/subtask2_anisotropic/run_subtask2_image_grouped.sh`
- Write logs: `outputs/subtask2_anisotropic/logs/`

**Step 1: Create run script**

Create a shell script under `outputs/subtask2_anisotropic/` with:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/team/zhanghaonan/mind
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate mind-py311
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
python scripts/experiments/anisotropic_scoring_comparison.py run-suite \
  --split-strategy image_grouped \
  --methods all \
  --device cuda:0 \
  --target-count 30 \
  --reference-chunk-size 16384 \
  --bootstrap-resamples 1000 \
  --num-folds 5 \
  --random-state 13 \
  --max-iter 100 \
  --output-root outputs/subtask2_anisotropic
```

**Step 2: Start tmux**

Run:

```bash
tmux new-session -d -s mind_subtask2_anisotropic \
  'bash /home/team/zhanghaonan/mind/outputs/subtask2_anisotropic/run_subtask2_image_grouped.sh 2>&1 | tee /home/team/zhanghaonan/mind/outputs/subtask2_anisotropic/logs/image_grouped.log'
```

**Step 3: Monitor without interrupting**

Run periodically:

```bash
tmux capture-pane -pt mind_subtask2_anisotropic -S -80
nvidia-smi
du -sh outputs
```

Expected:

- GPU 0 memory and CPU use stay below 90%.
- GPU 1 is not used.
- Disk stays below 1 TB.

**Step 4: Validate main metrics**

After tmux exits:

```bash
test "$(python - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/subtask2_anisotropic/metrics/subtask2_anisotropic_metrics.csv')
print(len(df))
PY
)" = "32"
```

Expected:

- `32` rows: 8 settings times 4 methods.

### Task 6: Run Heldout Evaluation For Best DASH-B Variant

**Files:**
- Append generated script: `outputs/subtask2_anisotropic/run_subtask2_heldout.sh`
- Write logs: `outputs/subtask2_anisotropic/logs/heldout.log`

**Step 1: Select best anisotropic variant**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/subtask2_anisotropic/metrics/subtask2_anisotropic_metrics.csv')
ani = df[(df.benchmark == 'dash-b') & (df.method.isin(['diag_maha', 'lowrank_maha', 'full_maha_shrink']))]
print(ani.groupby('method').pr_auc.mean().sort_values(ascending=False))
PY
```

Expected:

- One best method is selected by mean DASH-B PR-AUC.

**Step 2: Run heldout in the same tmux session name**

If the old tmux session ended, reuse the session name:

```bash
tmux new-session -d -s mind_subtask2_anisotropic \
  'bash /home/team/zhanghaonan/mind/outputs/subtask2_anisotropic/run_subtask2_heldout.sh 2>&1 | tee /home/team/zhanghaonan/mind/outputs/subtask2_anisotropic/logs/heldout.log'
```

The heldout script should call:

```bash
python scripts/experiments/anisotropic_scoring_comparison.py run-suite \
  --split-strategy object_heldout \
  --methods BEST_METHOD_FROM_STEP_1 \
  --device cuda:0 \
  --target-count 30 \
  --reference-chunk-size 16384 \
  --bootstrap-resamples 1000 \
  --num-folds 5 \
  --random-state 13 \
  --max-iter 100 \
  --output-root outputs/subtask2_anisotropic
```

**Step 3: Validate heldout metrics**

Run:

```bash
test "$(python - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/subtask2_anisotropic/metrics/subtask2_heldout_metrics.csv')
print(len(df))
PY
)" = "4"
```

Expected:

- `4` rows: one per model.

### Task 7: Format Tables And Analysis

**Files:**
- Create: `docs/tables/subtask2_anisotropic_comparison.md`
- Create: `docs/tables/subtask2_anisotropic_comparison.csv`
- Create: `docs/tables/subtask2_heldout_transfer.md`
- Create: `docs/tables/subtask2_heldout_transfer.csv`
- Create: `docs/review/subtask2_anisotropic_analysis.md`

**Step 1: Run format command**

Run:

```bash
python scripts/experiments/anisotropic_scoring_comparison.py format \
  --metrics-csv outputs/subtask2_anisotropic/metrics/subtask2_anisotropic_metrics.csv \
  --heldout-csv outputs/subtask2_anisotropic/metrics/subtask2_heldout_metrics.csv \
  --output-comparison-md docs/tables/subtask2_anisotropic_comparison.md \
  --output-comparison-csv docs/tables/subtask2_anisotropic_comparison.csv \
  --output-heldout-md docs/tables/subtask2_heldout_transfer.md \
  --output-heldout-csv docs/tables/subtask2_heldout_transfer.csv \
  --output-analysis docs/review/subtask2_anisotropic_analysis.md
```

Expected:

- Markdown and CSV tables are created.
- Analysis file includes explicit radius summary and final decision.

**Step 2: Validate table shape**

Run:

```bash
python - <<'PY'
import pandas as pd
main = pd.read_csv('docs/tables/subtask2_anisotropic_comparison.csv')
held = pd.read_csv('docs/tables/subtask2_heldout_transfer.csv')
assert len(main) == 8, len(main)
assert len(held) == 4, len(held)
for col in ['radius_ball_isotropic', 'diag_maha', 'lowrank_maha', 'full_maha_shrink', 'no_manifold', 'linear_probe']:
    assert col in main.columns, col
for col in ['best_anisotropic', 'no_manifold', 'linear_probe']:
    assert col in held.columns, col
print('ok')
PY
```

Expected:

- Prints `ok`.

**Step 3: Validate analysis questions**

Run:

```bash
for phrase in \
  "1." \
  "2." \
  "3." \
  "4." \
  "5." \
  "6." \
  "7." \
  "Decision"
do
  rg "$phrase" docs/review/subtask2_anisotropic_analysis.md
done
```

Expected:

- Each requested answer and the decision section are present.

### Task 8: Final Validation And Cleanup

**Files:**
- Read: all files from Output Contract

**Step 1: Run focused tests**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/unit/test_neighbor_selection.py tests/unit/test_gpu_anisotropic.py
```

Expected:

- All tests pass.

**Step 2: Run formatting validation without CUDA**

Run:

```bash
python scripts/experiments/anisotropic_scoring_comparison.py format \
  --metrics-csv outputs/subtask2_anisotropic/metrics/subtask2_anisotropic_metrics.csv \
  --heldout-csv outputs/subtask2_anisotropic/metrics/subtask2_heldout_metrics.csv \
  --output-comparison-md docs/tables/subtask2_anisotropic_comparison.md \
  --output-comparison-csv docs/tables/subtask2_anisotropic_comparison.csv \
  --output-heldout-md docs/tables/subtask2_heldout_transfer.md \
  --output-heldout-csv docs/tables/subtask2_heldout_transfer.csv \
  --output-analysis docs/review/subtask2_anisotropic_analysis.md
```

Expected:

- It succeeds without loading GPU tensors.

**Step 3: Check tmux and GPU state**

Run:

```bash
tmux ls || true
nvidia-smi
```

Expected:

- No needed Sub-task 2 job is still running.
- GPU 0 is not left with a stuck process.
- GPU 1 remains untouched by Sub-task 2.

**Step 4: Check disk**

Run:

```bash
du -sh outputs
df -h .
```

Expected:

- Total output use is below 1 TB.

If space is tight, delete only clearly redundant Sub-task 1 bad-cache archives:

```bash
rm -rf outputs/subtask1_neighbor_selection/archive_bad_cache
```

Do not delete current Sub-task 2 features, metrics, logs, or radii before the final report is committed.

**Step 5: Check git diff**

Run:

```bash
git status --short --branch
git diff -- src/mind/geometry/gpu_anisotropic.py tests/unit/test_gpu_anisotropic.py scripts/experiments/anisotropic_scoring_comparison.py docs/review/subtask2_anisotropic_analysis.md docs/tables/subtask2_anisotropic_comparison.md docs/tables/subtask2_anisotropic_comparison.csv docs/tables/subtask2_heldout_transfer.md docs/tables/subtask2_heldout_transfer.csv
```

Expected:

- Only intended source, test, and docs files are staged or unstaged.
- `outputs/` files are not staged.
- Unrelated files remain untouched.

**Step 6: Final commit and push**

Run:

```bash
git add \
  src/mind/geometry/gpu_anisotropic.py \
  tests/unit/test_gpu_anisotropic.py \
  scripts/experiments/anisotropic_scoring_comparison.py \
  docs/review/subtask2_anisotropic_analysis.md \
  docs/tables/subtask2_anisotropic_comparison.md \
  docs/tables/subtask2_anisotropic_comparison.csv \
  docs/tables/subtask2_heldout_transfer.md \
  docs/tables/subtask2_heldout_transfer.csv
git commit -m "Complete Sub-task 2: local anisotropic metrics with radius-ball neighbors"
git push origin master
```

Expected:

- Commit succeeds.
- Push succeeds.

## Pre-Finish Audit

Before final response, answer these internally:

1. The original goal was to test local anisotropic metrics with locked radius-ball neighbors.
2. The subtasks were preflight, tests, scoring module, runner, full image-grouped run, heldout run, report formatting, final validation, commit, and push.
3. Mark each subtask complete only after its validation command passes.
4. If any subtask remains executable, continue before responding.
5. Report final validation, commit hashes, pushed branch, disk use, and residual risks.

