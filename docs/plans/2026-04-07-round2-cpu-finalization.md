# Round-Two CPU Finalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the CPU-only round-two paper assembly work from already-complete report artifacts.

**Architecture:** Reuse the saved round-two report trees under `outputs/round2_2026_04/reports/` as the single source of truth. Generate the missing paper-facing tables from those reports, then update the tracked summary docs so they reflect the saved artifacts instead of older execution notes.

**Tech Stack:** Python, pandas, existing exporter helpers in `scripts/export_paper_package.py`, Markdown, git

---

### Task 1: Generate The DASH-B Main Table

**Files:**
- Modify: `docs/tables/round2/table1_dash_b.csv`
- Modify: `docs/tables/round2/table1_dash_b.md`
- Create: `docs/plans/2026-04-07-round2-cpu-finalization.md`

**Step 1: Verify the four DASH-B report roots are complete**

Run:

```bash
conda run --no-capture-output -n mind-py311 python - <<'PY'
from pathlib import Path
base = Path("outputs/round2_2026_04/reports")
roots = [
    base / "round2-qwen3-vl-8b-dash-b",
    base / "round2-internvl3.5-8b-dash-b",
    base / "round2-llava-onevision-7b-dash-b",
    base / "round2-molmo-7b-d-0924-dash-b",
]
wanted = {
    "full.csv",
    "drift_only.csv",
    "no_manifold.csv",
    "linear_probe.csv",
    "output_p_yes.csv",
    "output_logit_margin.csv",
    "output_chosen_answer_confidence.csv",
}
for root in roots:
    files = {p.name for p in (root / "variant_results").glob("*.csv")}
    print(root.name, "missing", sorted(wanted - files))
PY
```

Expected: every row prints `missing []`

**Step 2: Generate only the DASH-B wide benchmark table**

Run:

```bash
conda run --no-capture-output -n mind-py311 python - <<'PY'
from pathlib import Path
from scripts.export_paper_package import (
    discover_round_two_reports,
    build_wide_benchmark_table,
    _write_table_bundle,
)
reports = discover_round_two_reports(Path("outputs/round2_2026_04/reports"))
frame = build_wide_benchmark_table(reports, benchmark_key="dash-b")
_write_table_bundle(
    frame,
    export_csv=Path("artifacts/paper_round2/tables/table1_dash_b.csv"),
    export_md=Path("artifacts/paper_round2/tables/table1_dash_b.md"),
    docs_csv=Path("docs/tables/round2/table1_dash_b.csv"),
    docs_md=Path("docs/tables/round2/table1_dash_b.md"),
)
print(frame.to_string(index=False))
PY
```

Expected: four-row table written to both `artifacts/paper_round2/tables/` and `docs/tables/round2/`

**Step 3: Verify the generated files**

Run:

```bash
sed -n '1,80p' docs/tables/round2/table1_dash_b.md
```

Expected: one markdown table with four models and seven populated method columns

**Step 4: Commit**

```bash
git add docs/plans/2026-04-07-round2-cpu-finalization.md docs/tables/round2/table1_dash_b.csv docs/tables/round2/table1_dash_b.md
git commit -m "docs(tables): add DASH-B round2 main table"
```

### Task 2: Generate The Feature Ablation Table

**Files:**
- Modify: `docs/tables/round2/table2_feature_ablation.csv`
- Modify: `docs/tables/round2/table2_feature_ablation.md`

**Step 1: Verify feature-variant entries exist in the saved reports**

Run:

```bash
conda run --no-capture-output -n mind-py311 python - <<'PY'
from pathlib import Path
import json
for path in sorted(Path("outputs/round2_2026_04/reports").glob("round2-*/baselines.json")):
    data = json.loads(path.read_text())
    if "-popular" in path.parent.name or "-dash-b" in path.parent.name:
        print(path.parent.name, all(key in data for key in [
            "raw_curve_only",
            "raw_plus_calibrated_simple",
            "raw_plus_calibrated_full_curve",
            "raw_plus_calibrated_haar",
        ]))
PY
```

Expected: `True` for the eight popular and DASH-B report roots

**Step 2: Generate only the wide feature table**

Run:

```bash
conda run --no-capture-output -n mind-py311 python - <<'PY'
from pathlib import Path
from scripts.export_paper_package import (
    discover_round_two_reports,
    build_wide_feature_table,
    _write_table_bundle,
)
reports = discover_round_two_reports(Path("outputs/round2_2026_04/reports"))
frame = build_wide_feature_table(reports)
_write_table_bundle(
    frame,
    export_csv=Path("artifacts/paper_round2/tables/table2_feature_ablation.csv"),
    export_md=Path("artifacts/paper_round2/tables/table2_feature_ablation.md"),
    docs_csv=Path("docs/tables/round2/table2_feature_ablation.csv"),
    docs_md=Path("docs/tables/round2/table2_feature_ablation.md"),
)
print(frame.to_string(index=False))
PY
```

Expected: eight-row feature table written to `docs/tables/round2/`

**Step 3: Commit**

```bash
git add docs/tables/round2/table2_feature_ablation.csv docs/tables/round2/table2_feature_ablation.md
git commit -m "docs(tables): add round2 feature ablation table"
```

### Task 3: Write The Findings Summary Note

**Files:**
- Create: `docs/review/2026-04-round2-findings-summary.md`

**Step 1: Read the two main tables**

Run:

```bash
sed -n '1,80p' docs/tables/round2/table1_pope_popular.md
sed -n '1,80p' docs/tables/round2/table1_dash_b.md
```

Expected: both tables present and populated

**Step 2: Write the findings note**

Cover:
- MIND versus simple output baselines
- linear probe versus MIND
- `no_manifold` versus full MIND on DASH-B
- model differences
- DASH-B versus POPE popular

**Step 3: Commit**

```bash
git add docs/review/2026-04-round2-findings-summary.md
git commit -m "docs(review): add round2 findings summary"
```

### Task 4: Update The Results Summary

**Files:**
- Modify: `docs/results_summary.md`

**Step 1: Replace stale status text with current round-two state**

Required updates:
- both main tables are now available
- no stale “pending cache” statements
- use current round-two numbers only

**Step 2: Commit**

```bash
git add docs/results_summary.md
git commit -m "docs(results): update round2 summary tables"
```

### Task 5: Update The Paper Outline

**Files:**
- Modify: `docs/paper_outline.md`

**Step 1: Update the results-facing sections**

Required updates:
- mention the four-model two-benchmark pattern
- say plainly that linear probe beats MIND on both benchmarks
- say plainly that `no_manifold` beats full MIND on DASH-B

**Step 2: Commit**

```bash
git add docs/paper_outline.md
git commit -m "docs(paper): update round2 findings and framing"
```

### Task 6: Final Verification And Completion Note

**Files:**
- Create: `docs/review/2026-04-round4-completion.md`

**Step 1: Run the full test suite**

Run:

```bash
conda run --no-capture-output -n mind-py311 pytest -q
```

Expected: passing suite

**Step 2: Write the session completion note**

Record:
- what this CPU-only session finished
- what still depends on future GPU time:
  - HALP
  - GLSim
  - readout extraction
  - adversarial
  - RePOPE
  - transfer controls
  - bank-size ablation
  - layer-count ablation

**Step 3: Commit**

```bash
git add docs/review/2026-04-round4-completion.md
git commit -m "docs(closeout): add round4 completion note"
```
