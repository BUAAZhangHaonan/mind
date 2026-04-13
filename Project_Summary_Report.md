# Project Summary Report

## Evidence Note

This report uses tracked repo materials plus retained local round-two outputs available in the workspace at `/home/team/zhanghaonan/mind/outputs/round2_2026_04/`. The report relies on source code, configs, tests, tracked round-two tables under `docs/tables/`, and the retained workspace outputs used as the source of truth for paper-facing results. [Sources: `docs/_archive/review/results_summary.md`; `docs/_archive/review/paper_outline.md`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/`; `docs/tables/`]

## 1. Project Overview and Background

The project is named `MIND`. In `README.md`, `MIND` is expanded as “Multi-scale Internal Normal-residual Drift,” but the current paper outline says the paper should stop expanding the acronym and should treat `MIND` as a method name. That naming mismatch is still present in the repository. [Sources: `README.md`; `docs/_archive/review/paper_outline.md`]

The project studies early detection of multimodal object hallucination in vision-language models. The maintained scope in the current repository is object hallucination detection only. The paper outline states the core question as whether grounded and hallucinated behavior leave a useful pre-answer geometric signal in hidden states before answer generation starts. [Sources: `README.md`; `docs/_archive/review/paper_outline.md`]

The codebase is organized as a research pipeline. Config files define models and datasets, scripts run staged experiments, tests cover method behavior, and export tooling writes paper tables and PNG figure assets. [Sources: `README.md`; `scripts/run_experiment.py`; `scripts/export_paper_package.py`; `tests/integration/test_paper_export.py`]

The intended users are `[Not Found]`. The repository does not contain a direct statement naming the intended users. [Sources: repository contents inspected across `README.md`, `docs/`, `configs/`, `scripts/`, and `tests/`]

The benchmark surface recorded in the configs is POPE, RePOPE, DASH-B, and optional H-POPE. The maintained comparator surface in the live workflow is output-confidence baselines, linear probes, and official HALP row runs. Official GLSim is not wired into the current POPE/DASH-B workflow. [Sources: `configs/data/pope.yaml`; `configs/data/repope.yaml`; `configs/data/dash_b.yaml`; `configs/data/hpope.yaml`; `scripts/run_halp.py`; `scripts/run_glsim.py`; `src/mind/comparators/glsim.py`]

## 2. Application Scenarios

The repository supports research evaluation of pre-answer hallucination signals in large vision-language models. The current staged workflow is: normalize a benchmark into a shared yes/no object-existence format, extract pre-generation hidden states, build reference banks with `build_reference`, compute drift features, score baseline methods, run official HALP row comparisons where available, and export paper tables. [Sources: `scripts/prepare_data.py`; `scripts/extract_eval_states.py`; `scripts/build_manifolds.py`; `scripts/compute_drift.py`; `scripts/compute_baselines.py`; `scripts/run_halp.py`; `scripts/export_paper_package.py`]

The repository also supports controlled comparison across four model configs in the retained round-two lane: Qwen3-VL-8B, InternVL3.5-8B, LLaVA-OneVision-7B, and Molmo-7B-D-0924. The wrapper layer standardizes prompt construction, multimodal batching, generation, and hidden-state extraction across the retained round-two model configs, with model-specific wrapper selection handled by the factory. [Sources: `configs/models/qwen3_vl_8b.yaml`; `configs/models/internvl3_5_8b.yaml`; `configs/models/llava_onevision_7b.yaml`; `configs/models/molmo_7b_d_0924.yaml`; `src/mind/models/factory.py`; `src/mind/models/wrappers.py`]

The export path is part of the maintained workflow. `scripts/export_paper_package.py` reads saved round-two reports and writes markdown and CSV tables into `docs/tables/`. [Sources: `scripts/export_paper_package.py`; `docs/tables/`]

The current repository materials describe experimental runtime constraints rather than product deployment constraints. The active documentation and environment checks reference local benchmark assets, Hugging Face checkpoints, the `mind-py311` environment, and A100 GPUs. [Sources: `README.md`; `scripts/verify_env.py`]

Several failure cases are handled explicitly in code. Unsupported dataset directory layouts raise errors, missing H-POPE files raise `DatasetUnavailableError`, infeasible object-heldout folds raise errors, and missing reference coverage during drift computation raises an error instead of silently continuing. [Sources: `src/mind/data/pope.py`; `src/mind/evaluation/baselines.py`; `scripts/compute_drift.py`]

Privacy, compliance, and end-user safety requirements are `[Not Found]` in the current repository materials. [Sources: repository contents inspected across `README.md`, `docs/`, `configs/`, `scripts/`, and `tests/`]

## 3. Tasks and Objectives

The checked-in code defines a staged experiment pipeline. The table below summarizes the main tasks, their inputs, outputs, and success conditions in the current repository.

| Task | Inputs | Outputs | Success Criteria |
| --- | --- | --- | --- |
| Benchmark normalization | Raw benchmark files and dataset config metadata | Normalized JSONL records with `sample_id`, `image_id`, `image_path`, `question`, `label`, `object_name`, `split`, and `subset` | Records are converted to a common object yes/no format; RePOPE can override labels by `sample_id` |
| Reference candidate construction | COCO annotations, benchmark image ids, object names | Candidate JSON for grounded reference images | Candidate images contain the queried object and exclude evaluation image ids |
| Prefill extraction | Normalized records, model wrapper, processor, model checkpoint | Hidden-state cache shards with selected layers, layer vectors, logits, and parsed answers | Pre-generation hidden states and first-token logits are saved for each example |
| Reference bank building | Cached grounded reference states | Per-object or shared banks plus saved support statistics | Cleaned reference banks exist and support manifold scoring |
| Drift feature generation | Eval cache shards, reference banks, saved stats | Feature parquet with raw drift and calibrated features | Each evaluation example receives valid features; missing reference coverage causes failure |
| Baseline evaluation | Feature frame, cache entries, reference banks, split protocol | `baselines.json`, variant CSVs, and split sensitivity outputs | Full MIND, drift-only, no-manifold, linear probe, and output baselines are scored with metrics and confidence intervals |
| Comparator evaluation | Compact readout caches | `halp.json`, `halp_results.csv`, and `halp_selection.csv` | Official HALP row metrics are saved for the selected model and benchmark |
| Paper export | Round-two report directories | Markdown and CSV tables plus PNG figure assets and a figure manifest | Tracked tables in `docs/tables/` and export artifacts are generated from saved reports |
| Verification | Unit tests, integration tests, environment checks | Passing test output and verified runtime notes | Synthetic pipeline and paper export tests pass; environment and model loading are checked |

[Sources: `src/mind/data/pope.py`; `src/mind/extractors/prefill.py`; `scripts/build_manifolds.py`; `scripts/compute_drift.py`; `scripts/compute_baselines.py`; `scripts/run_halp.py`; `scripts/export_paper_package.py`; `tests/integration/test_synthetic_pipeline.py`; `tests/integration/test_paper_export.py`]

## 4. Methods and Technical Approach

### 4.1 Overall pipeline

The experiment runner is a staged CLI workflow. An experiment config combines model config, dataset config, subset, split, selected layer count, and detector choice. `scripts/run_experiment.py` expands that config into stages such as `prepare`, `build_reference`, `cache_reference`, `extract_eval`, `build_manifolds`, `compute_drift`, `baselines`, `train_detector`, `evaluate`, and `plot`, and `plot` writes PNGs under `plots/<experiment_name>/`. [Sources: `src/mind/config/schema.py`; `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`; `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`; `scripts/run_experiment.py`; `scripts/plot_results.py`]

### 4.2 Data model and normalization

The normalization layer converts POPE-style object-existence tasks into a shared record type. `load_object_yes_no_records` builds `HallucinationRecord` objects with consistent fields, and `apply_repope_labels` reassigns labels by `sample_id` for RePOPE. DASH-B is normalized through a directory-to-row conversion that builds yes/no questions from object names. [Sources: `src/mind/data/pope.py`; `docs/_archive/review/results_summary.md`]

The dataset configs tie the live workflow to their intended sources. POPE and RePOPE point to COCO `val2014`, DASH-B uses its own image root, and H-POPE remains optional because public assets were not available locally. [Sources: `configs/data/pope.yaml`; `configs/data/repope.yaml`; `configs/data/dash_b.yaml`; `configs/data/hpope.yaml`; `src/mind/data/pope.py`]

### 4.3 Model wrappers and extraction

`create_model_wrapper` dispatches to wrapper classes for Qwen, InternVL, LLaVA-OneVision, and Molmo. The wrappers standardize padding, prompt construction, multimodal batching, generation, hidden-state extraction, and model-specific helpers such as vision token span resolution. [Sources: `src/mind/models/factory.py`; `src/mind/models/wrappers.py`]

The extraction path is pre-generation. `extract_prefill_entries` prepares batched inputs, runs generation, recovers prefill hidden states, slices the selected layers, stores those vectors, keeps the first generated-token logits, and parses the answer as `yes`, `no`, or `None`. [Sources: `src/mind/extractors/prefill.py`; `src/mind/models/wrappers.py`]

Comparator extraction is separate from the main MIND cache path. `extract_prefill_readout_entries` stores richer pre-generation tensors, and `compact_prefill_readout_entry` reduces them to the tensors needed by comparator code. [Sources: `src/mind/extractors/readouts.py`; `scripts/extract_readout_states.py`]

### 4.4 Reference banks and local geometry

The reference-bank step uses cleaned grounded examples. `clean_reference_entries` keeps entries whose `parsed_answer == 1`. `build_reference_bank` supports object-conditioned banks, a shared bank under `__shared__`, and shuffled-object banks. [Sources: `src/mind/manifolds/local_pca.py`; `scripts/build_manifolds.py`]

The local geometry step applies local PCA. For each query vector, the method finds nearest neighbors in the reference bank, computes a local PCA basis, and measures the normalized normal residual as off-manifold distance divided by average neighborhood radius. [Sources: `src/mind/manifolds/local_pca.py`]

The code also saves support statistics such as residual mean, residual standard deviation, neighbor residual mean, and neighbor radius quantiles per object and per layer. Those statistics are reused for calibration and support checks. [Sources: `src/mind/manifolds/local_pca.py`; `scripts/build_manifolds.py`]

### 4.5 Drift features and detector variants

`compute_drift_curve` scores each selected layer against the reference bank. `calibrate_drift_curve` then z-scores the raw curve using saved bank statistics. `build_drift_features` keeps raw drift values and raw summary statistics, while Haar wavelet features are extracted only from the calibrated curve. The paper notes freeze the default full feature set as `raw + calibrated simple stats`. [Sources: `src/mind/drift/features.py`; `src/mind/wavelets/features.py`; `docs/tables/phase_one_popular_decision.md`; `docs/_archive/review/paper_outline.md`]

The baseline framework exposes four named feature variants: `raw_curve_only`, `raw_plus_calibrated_simple`, `raw_plus_calibrated_full_curve`, and `raw_plus_calibrated_haar`. The default full variant is `raw_plus_calibrated_simple`. The current baseline lane also evaluates `drift_only`, `no_manifold`, `linear_probe`, `output_p_yes`, `output_logit_margin`, and `output_chosen_answer_confidence`. [Sources: `src/mind/evaluation/baselines.py`; `scripts/compute_baselines.py`; `tests/unit/test_baselines.py`]

### 4.6 Evaluation protocols

The evaluation layer supports three split strategies: `row`, `image_grouped`, and `object_heldout`. Group columns are tied to the selected strategy, and confidence intervals are computed on the corresponding grouping unit. The object-heldout path filters to supported objects and refuses infeasible fold counts. [Sources: `src/mind/evaluation/baselines.py`; `scripts/compute_baselines.py`; `docs/_archive/review/results_summary.md`]

The hallucination label is reconstructed from `ground_truth_label` and `answer_label` as “object absent in ground truth, but answered yes by the model.” [Sources: `src/mind/evaluation/baselines.py`; `scripts/compute_drift.py`]

### 4.7 Comparator methods

HALP is part of the maintained comparator lane. The HALP utilities define the official 11-probe setup: one `vision_only` probe plus query-token and vision-token probes at five layer positions chosen by `resolve_halp_layer_indices`. The tests confirm the five-layer schedule and the probe list. [Sources: `src/mind/comparators/halp.py`; `tests/unit/test_halp.py`]

`scripts/run_halp.py` accepts `--split-strategy` values `row`, `image_grouped`, and `object_heldout`, while the retained official row evidence is limited to the POPE popular and DASH-B row outputs. The script writes `halp.json`, `halp_results.csv`, and `halp_selection.csv`. [Sources: `scripts/run_halp.py`; `docs/_archive/review/results_summary.md`]

Official GLSim is not implemented in the current POPE/DASH-B workflow. `scripts/run_glsim.py` exits with a message that official GLSim is not wired into the current round-two workflow, and `src/mind/comparators/glsim.py` exports no active implementation. [Sources: `scripts/run_glsim.py`; `src/mind/comparators/glsim.py`]

### 4.8 External libraries and tools

The repository uses PyTorch for model inference and probes, Transformers and Accelerate for model loading, Pandas and PyArrow for tabular outputs, scikit-learn for logistic baselines and split utilities, PyWavelets for Haar features, Matplotlib and Seaborn for plots, and Pydantic for typed config validation. [Sources: `requirements.txt`; `pyproject.toml`; `environment.yml`; `src/mind/config/schema.py`; `src/mind/comparators/halp.py`; `src/mind/wavelets/features.py`; `scripts/plot_results.py`]

## 5. Innovations and Contributions

The project does not introduce a new base model. The main contribution in the current repository is a research pipeline for measuring pre-answer geometric drift in VLM hidden states across multiple model families and benchmark datasets. [Sources: `README.md`; `docs/_archive/review/paper_outline.md`; `src/mind/manifolds/local_pca.py`; `src/mind/drift/features.py`]

The central method contribution is object-conditioned manifold drift before answer generation. The pipeline builds grounded reference banks, fits local PCA neighborhoods, and measures normalized off-manifold residuals layer by layer before the answer starts. [Sources: `src/mind/manifolds/local_pca.py`; `src/mind/drift/features.py`; `docs/_archive/review/paper_outline.md`]

The second contribution is calibration. The repository stores reference-bank statistics, calibrates each layer by those statistics, freezes `raw + calibrated simple stats` as the default full feature set, and keeps full-curve and Haar variants as ablations. [Sources: `src/mind/manifolds/local_pca.py`; `src/mind/drift/features.py`; `docs/tables/phase_one_popular_decision.md`; `docs/_archive/review/paper_outline.md`]

The third contribution is a shared normalization and evaluation surface for POPE, RePOPE, and DASH-B. The retained code paths use the same normalization layer and the same baseline machinery across those benchmarks. [Sources: `src/mind/data/pope.py`; `scripts/compute_baselines.py`; `docs/_archive/review/results_summary.md`]

The fourth contribution is reproducibility support through typed configs, staged scripts, synthetic end-to-end tests, paper export tests, and tracked round-two tables. [Sources: `src/mind/config/schema.py`; `tests/integration/test_synthetic_pipeline.py`; `tests/integration/test_paper_export.py`; `docs/tables/`]

The maintained report also makes an explicit boundary between official HALP row evidence and methods that are not part of the current official workflow, such as official GLSim. [Sources: `scripts/run_halp.py`; `scripts/run_glsim.py`; `src/mind/comparators/glsim.py`]

## 6. Experimental Design

### 6.1 Datasets and splits

The retained normalized files available in `/home/team/zhanghaonan/mind/outputs/round2_2026_04/normalized/` are:

| Dataset artifact | Rows |
| --- | ---: |
| `outputs/round2_2026_04/normalized/pope/popular.jsonl` | 3000 |
| `outputs/round2_2026_04/normalized/pope/adversarial.jsonl` | 3000 |
| `outputs/round2_2026_04/normalized/repope/popular.jsonl` | 2727 |
| `outputs/round2_2026_04/normalized/dash-b/main.jsonl` | 2682 |

[Sources: retained normalized files under `/home/team/zhanghaonan/mind/outputs/round2_2026_04/normalized/`]

POPE random and RePOPE adversarial are not part of the current retained inventory. Neither retained round-two normalized subset exists in the workspace, and the older pre-archive normalized tree also does not contain surviving copies, so both subsets remain earlier-plan material outside the current retained scope. [Sources: retained normalized files under `/home/team/zhanghaonan/mind/outputs/round2_2026_04/normalized/`; workspace sweep of the legacy normalized tree before archival]

The dataset configs tie these files to their intended image roots and public sources. POPE and RePOPE use COCO `val2014`, DASH-B uses its own image tree, and H-POPE remains optional because assets are not available locally. [Sources: `configs/data/pope.yaml`; `configs/data/repope.yaml`; `configs/data/dash_b.yaml`; `configs/data/hpope.yaml`; `src/mind/data/pope.py`]

### 6.2 Model set

The retained round-two model set is:

| Model label | Config key | Hugging Face id | Family |
| --- | --- | --- | --- |
| Qwen3-VL-8B | `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | `qwen_vl` |
| InternVL3.5-8B | `internvl3.5-8b` | `OpenGVLab/InternVL3_5-8B-HF` | `internvl` |
| LLaVA-OneVision-7B | `llava-onevision-7b` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | `llava_onevision` |
| Molmo-7B-D-0924 | `molmo-7b-d-0924` | `allenai/Molmo-7B-D-0924` | `molmo` |

[Sources: `configs/models/qwen3_vl_8b.yaml`; `configs/models/internvl3_5_8b.yaml`; `configs/models/llava_onevision_7b.yaml`; `configs/models/molmo_7b_d_0924.yaml`; `src/mind/models/factory.py`]

### 6.3 Default configuration

The main Qwen popular experiment preset uses `selected_layers: 16` and `detector: logistic`. The main all-subset preset uses the same selected-layer count and detector. `RuntimeConfig` also sets `selected_layers` to `16` by default. [Sources: `src/mind/config/schema.py`; `configs/experiments/medium/qwen3_vl_8b_pope_popular.yaml`; `configs/experiments/main/qwen3_vl_8b_pope_all.yaml`]

### 6.4 Evaluation metrics

The current metric surface includes accuracy, precision, recall, F1, false positive rate, ROC-AUC, PR-AUC, and TPR at 1% FPR. [Sources: `scripts/compute_baselines.py`; `src/mind/evaluation/metrics.py`]

### 6.5 Baselines and comparators

The baseline set used in the retained round-two main tables is:

- `p_yes`
- `logit_margin`
- `chosen_confidence`
- `drift_only`
- `no_manifold`
- `full MIND`
- `linear_probe`

The feature-ablation variants are `raw_only`, `raw_plus_simple_stats`, `raw_plus_full_curve`, and `raw_plus_Haar`. The retained comparator evidence is official HALP row outputs for POPE popular and DASH-B. Official GLSim is not part of the current workflow. [Sources: `scripts/export_paper_package.py`; `src/mind/evaluation/baselines.py`; `docs/tables/table1_pope_popular.md`; `docs/tables/table1_dash_b.md`; `docs/tables/table2_feature_ablation.md`; `scripts/run_halp.py`; `scripts/run_glsim.py`]

### 6.6 Hardware and software environment

The current documented environment is the `mind-py311` conda environment, Python 3.11, PyTorch 2.6.0 with CUDA 12.4, and `2 x NVIDIA A100 80GB PCIe`. [Sources: `README.md`; `scripts/verify_env.py`]

## 7. Experimental Results

### 7.1 Final round-two main results: POPE popular

| model | benchmark | p_yes | logit_margin | chosen_confidence | drift_only | no_manifold | full_MIND | linear_probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | POPE popular | ROC 0.6342 [0.6152, 0.6549]; PR 0.0451 [0.0375, 0.0528] | ROC 0.5955 [0.5819, 0.6104]; PR 0.0422 [0.0350, 0.0491] | ROC 0.7947 [0.7437, 0.8387]; PR 0.1462 [0.1066, 0.1965] | ROC 0.8497 [0.8262, 0.8741]; PR 0.1253 [0.0985, 0.1605] | ROC 0.8385 [0.7958, 0.8783]; PR 0.1983 [0.1385, 0.2657] | ROC 0.8908 [0.8694, 0.9105]; PR 0.1741 [0.1375, 0.2169] | ROC 0.9161 [0.8868, 0.9414]; PR 0.3803 [0.2892, 0.4728] |
| InternVL3.5-8B | POPE popular | ROC 0.5601 [0.5471, 0.5740]; PR 0.0878 [0.0783, 0.0969] | ROC 0.5454 [0.5335, 0.5588]; PR 0.0861 [0.0767, 0.0949] | ROC 0.8039 [0.7745, 0.8342]; PR 0.2637 [0.2198, 0.3142] | ROC 0.8802 [0.8622, 0.8982]; PR 0.4270 [0.3680, 0.4888] | ROC 0.8559 [0.8351, 0.8761]; PR 0.4033 [0.3393, 0.4618] | ROC 0.8978 [0.8810, 0.9140]; PR 0.5092 [0.4528, 0.5669] | ROC 0.9366 [0.9192, 0.9518]; PR 0.6550 [0.5881, 0.7133] |
| LLaVA-OneVision-7B | POPE popular | ROC 0.6200 [0.6044, 0.6366]; PR 0.0357 [0.0289, 0.0429] | ROC 0.6095 [0.5933, 0.6266]; PR 0.0347 [0.0280, 0.0415] | ROC 0.8277 [0.7772, 0.8742]; PR 0.1195 [0.0877, 0.1585] | ROC 0.8030 [0.7708, 0.8364]; PR 0.0941 [0.0683, 0.1332] | ROC 0.8078 [0.7618, 0.8537]; PR 0.1282 [0.0886, 0.1947] | ROC 0.8085 [0.7809, 0.8405]; PR 0.0874 [0.0642, 0.1225] | ROC 0.8833 [0.8403, 0.9228]; PR 0.3238 [0.2347, 0.4311] |
| Molmo-7B-D-0924 | POPE popular | ROC 0.5658 [0.5412, 0.5908]; PR 0.0512 [0.0430, 0.0591] | ROC 0.5810 [0.5658, 0.5964]; PR 0.0541 [0.0462, 0.0618] | ROC 0.6522 [0.6200, 0.6863]; PR 0.0687 [0.0556, 0.0834] | ROC 0.8346 [0.8099, 0.8578]; PR 0.1651 [0.1324, 0.2014] | ROC 0.8256 [0.7956, 0.8557]; PR 0.1857 [0.1459, 0.2341] | ROC 0.8839 [0.8608, 0.9060]; PR 0.2992 [0.2327, 0.3691] | ROC 0.9209 [0.8988, 0.9424]; PR 0.5606 [0.4703, 0.6409] |

[Sources: `docs/tables/table1_pope_popular.md`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-final/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular/baselines.json`]

On this table, full MIND is higher than the three output-confidence baselines on all four models. `linear_probe` is higher than full MIND on all four rows. For LLaVA-OneVision-7B, `chosen_confidence` is higher than full MIND. [Sources: `docs/tables/table1_pope_popular.md`; retained round-two `baselines.json` files]

### 7.2 Final round-two main results: DASH-B

| model | benchmark | p_yes | logit_margin | chosen_confidence | drift_only | no_manifold | full_MIND | linear_probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | DASH-B | ROC 0.4764 [0.4335, 0.5219]; PR 0.2193 [0.1834, 0.2612] | ROC 0.6160 [0.5697, 0.6613]; PR 0.2801 [0.2598, 0.3004] | ROC 0.6856 [0.6559, 0.7153]; PR 0.4057 [0.3678, 0.4420] | ROC 0.9185 [0.9044, 0.9324]; PR 0.7371 [0.6989, 0.7729] | ROC 0.9290 [0.9162, 0.9415]; PR 0.7784 [0.7433, 0.8118] | ROC 0.9193 [0.9049, 0.9332]; PR 0.7374 [0.6965, 0.7755] | ROC 0.9909 [0.9874, 0.9938]; PR 0.9779 [0.9714, 0.9841] |
| InternVL3.5-8B | DASH-B | ROC 0.4790 [0.4367, 0.5228]; PR 0.2558 [0.2159, 0.3004] | ROC 0.5584 [0.5150, 0.5993]; PR 0.2893 [0.2665, 0.3116] | ROC 0.6475 [0.6179, 0.6748]; PR 0.3860 [0.3507, 0.4218] | ROC 0.8593 [0.8426, 0.8747]; PR 0.7054 [0.6636, 0.7409] | ROC 0.8769 [0.8621, 0.8918]; PR 0.7288 [0.6907, 0.7630] | ROC 0.8574 [0.8404, 0.8728]; PR 0.7084 [0.6669, 0.7437] | ROC 0.9858 [0.9821, 0.9893]; PR 0.9699 [0.9617, 0.9768] |
| LLaVA-OneVision-7B | DASH-B | ROC 0.4497 [0.4009, 0.5000]; PR 0.3231 [0.2726, 0.3801] | ROC 0.6602 [0.6198, 0.6949]; PR 0.4297 [0.4019, 0.4562] | ROC 0.7046 [0.6754, 0.7331]; PR 0.4938 [0.4596, 0.5265] | ROC 0.8664 [0.8488, 0.8830]; PR 0.7431 [0.7105, 0.7736] | ROC 0.8996 [0.8839, 0.9130]; PR 0.7883 [0.7549, 0.8201] | ROC 0.8404 [0.8208, 0.8599]; PR 0.7234 [0.6893, 0.7553] | ROC 0.9923 [0.9896, 0.9944]; PR 0.9883 [0.9845, 0.9916] |
| Molmo-7B-D-0924 | DASH-B | ROC 0.6170 [0.5772, 0.6528]; PR 0.3289 [0.3022, 0.3538] | ROC 0.5369 [0.4955, 0.5745]; PR 0.2878 [0.2634, 0.3099] | ROC 0.6369 [0.5992, 0.6718]; PR 0.3425 [0.3164, 0.3673] | ROC 0.7967 [0.7775, 0.8149]; PR 0.5611 [0.5185, 0.6002] | ROC 0.8655 [0.8479, 0.8819]; PR 0.6861 [0.6479, 0.7191] | ROC 0.7795 [0.7589, 0.7978]; PR 0.5422 [0.5010, 0.5794] | ROC 0.9775 [0.9722, 0.9827]; PR 0.9561 [0.9450, 0.9655] |

[Sources: `docs/tables/table1_dash_b.md`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-dash-b/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-dash-b/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-dash-b/baselines.json`]

On this table, full MIND is higher than the three output-confidence baselines on all four models. `no_manifold` is higher than full MIND on all four rows, and `linear_probe` is higher than both on all four rows. [Sources: `docs/tables/table1_dash_b.md`; retained round-two `baselines.json` files]

### 7.3 Supplementary benchmark results: POPE adversarial

| model | benchmark | p_yes | logit_margin | chosen_confidence | drift_only | no_manifold | full_MIND | linear_probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | POPE adversarial | ROC 0.6199 [0.6030, 0.6375]; PR 0.0671 [0.0584, 0.0769] | ROC 0.5882 [0.5765, 0.6016]; PR 0.0642 [0.0560, 0.0729] | ROC 0.8089 [0.7694, 0.8419]; PR 0.2053 [0.1607, 0.2496] | ROC 0.8511 [0.8265, 0.8743]; PR 0.2154 [0.1786, 0.2607] | ROC 0.7981 [0.7668, 0.8266]; PR 0.2134 [0.1675, 0.2665] | ROC 0.8668 [0.8446, 0.8892]; PR 0.2450 [0.2017, 0.2977] | ROC 0.8164 [0.7774, 0.8519]; PR 0.3182 [0.2471, 0.3945] |
| InternVL3.5-8B | POPE adversarial | ROC 0.5548 [0.5414, 0.5695]; PR 0.1146 [0.1048, 0.1237] | ROC 0.5271 [0.5154, 0.5396]; PR 0.1094 [0.0997, 0.1185] | ROC 0.7817 [0.7575, 0.8060]; PR 0.2733 [0.2375, 0.3125] | ROC 0.8496 [0.8321, 0.8682]; PR 0.4037 [0.3540, 0.4549] | ROC 0.8013 [0.7797, 0.8232]; PR 0.3423 [0.2946, 0.3899] | ROC 0.8516 [0.8330, 0.8698]; PR 0.4185 [0.3706, 0.4690] | ROC 0.8555 [0.8336, 0.8758]; PR 0.4806 [0.4252, 0.5329] |
| LLaVA-OneVision-7B | POPE adversarial | ROC 0.6135 [0.5981, 0.6284]; PR 0.0579 [0.0493, 0.0664] | ROC 0.6061 [0.5918, 0.6197]; PR 0.0571 [0.0485, 0.0654] | ROC 0.8269 [0.7964, 0.8564]; PR 0.1763 [0.1379, 0.2232] | ROC 0.8050 [0.7758, 0.8328]; PR 0.1485 [0.1166, 0.1875] | ROC 0.6986 [0.6601, 0.7343]; PR 0.0948 [0.0747, 0.1211] | ROC 0.8076 [0.7798, 0.8362]; PR 0.1535 [0.1208, 0.1947] | ROC 0.7933 [0.7472, 0.8339]; PR 0.2638 [0.1960, 0.3391] |
| Molmo-7B-D-0924 | POPE adversarial | ROC 0.5161 [0.4881, 0.5449]; PR 0.0667 [0.0584, 0.0764] | ROC 0.5605 [0.5449, 0.5756]; PR 0.0735 [0.0652, 0.0827] | ROC 0.6659 [0.6327, 0.7001]; PR 0.1096 [0.0923, 0.1299] | ROC 0.8149 [0.7890, 0.8382]; PR 0.2235 [0.1814, 0.2695] | ROC 0.7527 [0.7151, 0.7850]; PR 0.1688 [0.1397, 0.2062] | ROC 0.8442 [0.8190, 0.8681]; PR 0.2850 [0.2342, 0.3426] | ROC 0.8298 [0.7965, 0.8573]; PR 0.3265 [0.2634, 0.4010] |

[Sources: `docs/tables/supp_pope_adversarial.md`; retained adversarial `baselines.json` files under `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/`]

On this benchmark, full MIND has the highest ROC-AUC on Qwen3-VL-8B, LLaVA-OneVision-7B, and Molmo-7B-D-0924, while `linear_probe` has the highest PR-AUC on all four models. [Sources: `docs/tables/supp_pope_adversarial.md`; retained adversarial `baselines.json` files]

### 7.4 Supplementary benchmark results: RePOPE

The retained round-two RePOPE report directories contain saved `baselines.json` files, and the tracked markdown table `docs/tables/supp_repope.md` is populated from those retained results. [Sources: `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-repope/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-repope/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-repope/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-repope/baselines.json`; `docs/tables/supp_repope.md`]

| model | benchmark | p_yes | logit_margin | chosen_confidence | drift_only | no_manifold | full_MIND | linear_probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | RePOPE | ROC 0.6382; PR 0.0504 | ROC 0.5969; PR 0.0472 | ROC 0.7924; PR 0.1555 | ROC 0.8585; PR 0.1352 | ROC 0.7738; PR 0.1187 | ROC 0.8843; PR 0.1804 | ROC 0.8697; PR 0.3419 |
| InternVL3.5-8B | RePOPE | ROC 0.5591; PR 0.0998 | ROC 0.5490; PR 0.0990 | ROC 0.8243; PR 0.3111 | ROC 0.8804; PR 0.4392 | ROC 0.8480; PR 0.4133 | ROC 0.8865; PR 0.4779 | ROC 0.8918; PR 0.5545 |
| LLaVA-OneVision-7B | RePOPE | ROC 0.6182; PR 0.0439 | ROC 0.6108; PR 0.0434 | ROC 0.8637; PR 0.1537 | ROC 0.7922; PR 0.0927 | ROC 0.7213; PR 0.1054 | ROC 0.7963; PR 0.0982 | ROC 0.8360; PR 0.2061 |
| Molmo-7B-D-0924 | RePOPE | ROC 0.5578; PR 0.0574 | ROC 0.5537; PR 0.0561 | ROC 0.6239; PR 0.0720 | ROC 0.8217; PR 0.1665 | ROC 0.7626; PR 0.1333 | ROC 0.8473; PR 0.2310 | ROC 0.8711; PR 0.4441 |

[Sources: retained RePOPE `baselines.json` files under `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/`]

### 7.5 Saved official HALP row results

The retained comparator tree contains official HALP row outputs for POPE popular and DASH-B. The grouped, object-heldout, and RePOPE HALP artifacts are not retained in the current workspace, so only row-split evidence is kept here. [Sources: retained HALP row `halp.json` files under `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/`; `scripts/run_halp.py`; `docs/_archive/review/results_summary.md`]

| model | POPE popular row HALP | DASH-B row HALP |
| --- | --- | --- |
| Qwen3-VL-8B | ROC 0.9527; PR 0.7280 | ROC 0.9982; PR 0.9948 |
| InternVL3.5-8B | ROC 0.9684; PR 0.7399 | ROC 0.9953; PR 0.9908 |
| LLaVA-OneVision-7B | ROC 0.9737; PR 0.6014 | ROC 0.9946; PR 0.9898 |
| Molmo-7B-D-0924 | ROC 0.9684; PR 0.6711 | ROC 0.9826; PR 0.9611 |

[Sources: `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-dash-b-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-dash-b-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-dash-b-halp-row/halp.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-dash-b-halp-row/halp.json`]

Each saved HALP row score is higher than the corresponding full MIND score on the same model and benchmark in the retained POPE popular and DASH-B tables. [Sources: retained HALP row `halp.json` files; `docs/tables/table1_pope_popular.md`; `docs/tables/table1_dash_b.md`]

### 7.6 Saved feature-ablation results

The retained feature-ablation evidence now comes from `docs/tables/table2_feature_ablation.md` for both POPE popular and DASH-B rows, with `docs/tables/phase_one_popular_decision.md` recording the freeze to `raw + calibrated simple stats`. The LLaVA-OneVision-7B and Molmo-7B-D-0924 POPE popular rows are populated in the tracked table. [Sources: `docs/tables/phase_one_popular_decision.md`; `docs/tables/table2_feature_ablation.md`; `docs/tables/table2_feature_ablation.csv`]

| model | benchmark | raw_only | raw_plus_simple_stats | raw_plus_full_curve | raw_plus_Haar |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | POPE popular | ROC 0.8462; PR 0.1159 | ROC 0.8908; PR 0.1741 | ROC 0.9145; PR 0.2593 | ROC 0.8690; PR 0.1470 |
| InternVL3.5-8B | POPE popular | ROC 0.8764; PR 0.4284 | ROC 0.8978; PR 0.5092 | ROC 0.9119; PR 0.5331 | ROC 0.8928; PR 0.4853 |
| LLaVA-OneVision-7B | POPE popular | ROC 0.8023; PR 0.0900 | ROC 0.8085; PR 0.0874 | ROC 0.8890; PR 0.1769 | ROC 0.8194; PR 0.0935 |
| Molmo-7B-D-0924 | POPE popular | ROC 0.8353; PR 0.1519 | ROC 0.8839; PR 0.2992 | ROC 0.9110; PR 0.3851 | ROC 0.8975; PR 0.3187 |
| Qwen3-VL-8B | DASH-B | ROC 0.9208 [0.9067, 0.9345]; PR 0.7429 [0.7038, 0.7795] | ROC 0.9193 [0.9049, 0.9332]; PR 0.7374 [0.6965, 0.7755] | ROC 0.9291 [0.9154, 0.9422]; PR 0.7697 [0.7319, 0.8064] | ROC 0.9179 [0.9029, 0.9330]; PR 0.7360 [0.6972, 0.7748] |
| InternVL3.5-8B | DASH-B | ROC 0.8595 [0.8426, 0.8748]; PR 0.7059 [0.6629, 0.7414] | ROC 0.8574 [0.8404, 0.8728]; PR 0.7084 [0.6669, 0.7437] | ROC 0.8595 [0.8426, 0.8749]; PR 0.7204 [0.6813, 0.7554] | ROC 0.8611 [0.8444, 0.8761]; PR 0.7097 [0.6671, 0.7466] |
| LLaVA-OneVision-7B | DASH-B | ROC 0.8347 [0.8149, 0.8543]; PR 0.7074 [0.6720, 0.7398] | ROC 0.8404 [0.8208, 0.8599]; PR 0.7234 [0.6893, 0.7553] | ROC 0.8505 [0.8314, 0.8691]; PR 0.7236 [0.6869, 0.7557] | ROC 0.8498 [0.8302, 0.8684]; PR 0.7339 [0.7009, 0.7640] |
| Molmo-7B-D-0924 | DASH-B | ROC 0.7753 [0.7565, 0.7929]; PR 0.5089 [0.4695, 0.5474] | ROC 0.7795 [0.7589, 0.7978]; PR 0.5422 [0.5010, 0.5794] | ROC 0.7987 [0.7798, 0.8155]; PR 0.5911 [0.5457, 0.6298] | ROC 0.7788 [0.7578, 0.7973]; PR 0.5421 [0.5021, 0.5778] |

[Sources: `docs/tables/phase_one_popular_decision.md`; `docs/tables/table2_feature_ablation.md`; `docs/tables/table2_feature_ablation.csv`]

The retained ablation evidence is one reason the paper notes freeze `raw + calibrated simple stats` as the default full feature set. [Sources: `docs/tables/phase_one_popular_decision.md`; `docs/_archive/review/paper_outline.md`]

### 7.7 Retained gaps and excluded material

The maintained report excludes historical and correction-phase artifacts because they are not part of the tracked repo tree used by this document. [Sources: `docs/_archive/review/results_summary.md`; `docs/_archive/review/paper_outline.md`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/`]

The tracked transfer-control table is now populated for both `image_grouped` and `object_heldout`. In the fresh `object_heldout` rows, full MIND reports ROC-AUC / PR-AUC of `0.6381 / 0.0461` for Qwen3-VL-8B, `0.8833 / 0.4610` for InternVL3.5-8B, `0.7226 / 0.0475` for LLaVA-OneVision-7B, and `0.7300 / 0.0865` for Molmo-7B-D-0924. `linear_probe` remains higher than full MIND on all four object-heldout rows. [Sources: `docs/tables/table3_transfer_controls.md`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-object-heldout/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular-object-heldout/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular-object-heldout/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular-object-heldout/baselines.json`]

### 7.8 Supplementary table status

| table | status | note |
| --- | --- | --- |
| `docs/tables/supp_dash_b_transfer.md` | populated | Retained DASH-B transfer rows were exported from the round-two dash-B reports in `outputs/round2_2026_04/reports/`. |
| `docs/tables/supp_bank_size.md` | placeholder-not-run | No bank-size-specific retained report artifact was found under `outputs/round2_2026_04/`. |
| `docs/tables/supp_layer_count.md` | placeholder-not-run | No layer-count-specific retained report artifact was found under `outputs/round2_2026_04/`. |

## 8. Conclusion and Future Work

The retained mainline evidence supports a narrow detector claim. Full MIND is higher than the simple output-confidence baselines on the retained POPE popular, DASH-B, and POPE adversarial rows, with one popular-table exception where LLaVA-OneVision-7B `chosen_confidence` is higher than full MIND. [Sources: `docs/tables/table1_pope_popular.md`; `docs/tables/table1_dash_b.md`; `docs/tables/supp_pope_adversarial.md`]

The retained mainline evidence does not support a strongest-detector claim. `linear_probe` is higher than full MIND on every retained POPE popular and DASH-B main-table row, and official HALP row scores are higher than full MIND on every retained popular and DASH-B comparator row. [Sources: `docs/tables/table1_pope_popular.md`; `docs/tables/table1_dash_b.md`; retained HALP row `halp.json` files]

The current maintained limitations are:

1. The naming mismatch remains. `README.md` still expands `MIND` as a multi-scale acronym, while the paper outline says not to do that. [Sources: `README.md`; `docs/_archive/review/paper_outline.md`]
2. H-POPE is configured but remains unavailable. This is a known limitation and is not blocking any current deliverable. [Sources: `configs/data/hpope.yaml`; `src/mind/data/pope.py`]

The retained future work is documentation and evidence closure on the maintained surface:

1. Keep the paper framing aligned with the retained evidence: compact pre-answer geometry versus simple output baselines, with explicit acknowledgment that richer internal baselines remain stronger under both `image_grouped` and `object_heldout` transfer controls. [Sources: `docs/_archive/review/paper_outline.md`; `docs/tables/table3_transfer_controls.md`; retained HALP row `halp.json` files]
2. Keep non-official comparator adaptations out of the maintained official workflow unless they are separately implemented and documented as official methods. [Sources: `scripts/run_glsim.py`; `src/mind/comparators/glsim.py`]

---

## Document Revision Log

| Date | Section | Change | Source |
|------|---------|--------|--------|
| 2026-04-12 | Evidence Note and Sections 1-6 | Swapped stale historical source citations for the archived `docs/_archive/review/results_summary.md` and `docs/_archive/review/paper_outline.md` paths while keeping the verified narrative intact | `docs/_archive/review/results_summary.md`; `docs/_archive/review/paper_outline.md`; `README.md`; `src/mind/manifolds/local_pca.py`; `src/mind/drift/features.py` |
| 2026-04-12 | Sections 6-7 | Re-anchored dataset inventory and result tables to retained normalized files, retained round-two tables, retained `baselines.json` files, and retained HALP row `halp.json` files | `/home/team/zhanghaonan/mind/outputs/round2_2026_04/normalized/`; `docs/tables/table1_pope_popular.md`; `docs/tables/table1_dash_b.md`; `docs/tables/supp_pope_adversarial.md`; retained `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/` |
| 2026-04-12 | Sections 7.4-7.7 | Marked the RePOPE table as populated, filled the LLaVA and Molmo POPE popular ablation rows, and narrowed the remaining tracked gap to blank `object_heldout` cells in `table3_transfer_controls.md` | `docs/tables/supp_repope.md`; `docs/tables/table2_feature_ablation.md`; `docs/tables/table3_transfer_controls.md`; retained `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/` |
| 2026-04-12 | Section 8 | Updated the limitations and future-work notes to match the current tracked-table state and the remaining transfer-control gap | `docs/tables/table3_transfer_controls.md`; `README.md`; `configs/data/hpope.yaml` |
| 2026-04-13 | Sections 7.7-8 | Populated the previously blank `object_heldout` transfer-control rows from fresh Qwen, InternVL, LLaVA, and Molmo held-out reports, and removed the stale “remaining tracked gap” language | `docs/tables/table3_transfer_controls.md`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-qwen3-vl-8b-popular-object-heldout/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-internvl3.5-8b-popular-object-heldout/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-llava-onevision-7b-popular-object-heldout/baselines.json`; `/home/team/zhanghaonan/mind/outputs/round2_2026_04/reports/round2-molmo-7b-d-0924-popular-object-heldout/baselines.json` |
| 2026-04-13 | Sections 1, 3, 4, 7 | Corrected wrapper dispatch wording, named `build_reference` in the staged workflow, aligned HALP CLI wording with the retained official row evidence, and updated the plot artifact contract | `/home/team/zhanghaonan/mind/Project_Summary_Report.md` |
