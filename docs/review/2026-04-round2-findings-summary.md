# Round-Two Findings Summary

Date: 2026-04-07

This note summarizes the current paper-facing findings from the two completed main tables:

- `docs/tables/round2/table1_pope_popular.md`
- `docs/tables/round2/table1_dash_b.md`

## 1. MIND Versus Simple Output Baselines

MIND beats the simple output baselines on both completed benchmarks for all four models.

On `POPE popular`, full MIND beats the best simple output baseline by:

- Qwen: `+0.0961` ROC-AUC and `+0.0279` PR-AUC
- InternVL: `+0.0938` ROC-AUC and `+0.2455` PR-AUC
- Molmo: `+0.2318` ROC-AUC and `+0.2305` PR-AUC

LLaVA is the one exception on `POPE popular` if we compare against chosen-answer confidence alone:

- LLaVA: full MIND is `-0.0191` ROC-AUC and `-0.0321` PR-AUC against chosen-answer confidence

On `DASH-B`, full MIND beats the best simple output baseline on all four models:

- Qwen: `+0.2338` ROC-AUC and `+0.3317` PR-AUC
- InternVL: `+0.2099` ROC-AUC and `+0.3224` PR-AUC
- LLaVA: `+0.1358` ROC-AUC and `+0.2296` PR-AUC
- Molmo: `+0.1426` ROC-AUC and `+0.1997` PR-AUC

So the main positive claim still holds: compact geometry clearly beats simple output confidence methods in the completed round-two runs.

## 2. Linear Probe Versus MIND

The linear probe beats full MIND everywhere.

On `POPE popular`, the linear-probe gains over full MIND are:

- Qwen: `+0.0253` ROC-AUC and `+0.2061` PR-AUC
- InternVL: `+0.0389` ROC-AUC and `+0.1458` PR-AUC
- LLaVA: `+0.0748` ROC-AUC and `+0.2364` PR-AUC
- Molmo: `+0.0370` ROC-AUC and `+0.2614` PR-AUC

On `DASH-B`, the gap gets even larger:

- Qwen: `+0.0715` ROC-AUC and `+0.2405` PR-AUC
- InternVL: `+0.1284` ROC-AUC and `+0.2615` PR-AUC
- LLaVA: `+0.1518` ROC-AUC and `+0.2649` PR-AUC
- Molmo: `+0.1980` ROC-AUC and `+0.4139` PR-AUC

That means the paper cannot sell MIND as the strongest internal-state detector. The honest claim is narrower: MIND is a compact, interpretable geometry signal that beats simple output baselines, but it does not close the gap to a richer probe.

## 3. `no_manifold` Versus Full MIND On DASH-B

`no_manifold` beats full MIND on every completed `DASH-B` row.

- Qwen: `+0.0097` ROC-AUC and `+0.0410` PR-AUC
- InternVL: `+0.0195` ROC-AUC and `+0.0204` PR-AUC
- LLaVA: `+0.0591` ROC-AUC and `+0.0649` PR-AUC
- Molmo: `+0.0860` ROC-AUC and `+0.1439` PR-AUC

This does not mean the geometry signal disappears on `DASH-B`. It means the current manifold step is not helping on the harder benchmark. The useful part of the method on `DASH-B` is still the drift signal itself, but the local manifold normalization is not paying for its complexity in the saved runs.

## 4. Model Differences

The four models do not behave the same way.

- On `POPE popular`, InternVL and Molmo are the strongest full-MIND rows by PR-AUC.
- On `DASH-B`, Qwen stays strong under full MIND, while Molmo drops the most.
- LLaVA is the only popular row where chosen-answer confidence edges out full MIND.
- The linear-probe gap is largest on `DASH-B` for Molmo and LLaVA, which suggests that those models still expose strong internal evidence, but the compact drift summary is leaving a lot of it on the table.

## 5. DASH-B Versus POPE Popular

The move from `POPE popular` to `DASH-B` changes the story in two ways.

First, the simple output baselines do not become enough. Full MIND still beats them across the board on `DASH-B`.

Second, the harder benchmark hurts the manifold version more than the simpler geometry variants. That is why `no_manifold` overtakes full MIND on every completed `DASH-B` row. So the negative result is not “geometry fails on DASH-B.” The sharper result is “the manifold step does not transfer as well as the rest of the geometry pipeline.”

## Bottom Line

The current round-two evidence supports three plain conclusions:

- MIND is a real improvement over simple output confidence baselines.
- Linear probing is still much stronger than MIND as a detector.
- The manifold step is the weak point on `DASH-B`, not the whole geometry signal.
