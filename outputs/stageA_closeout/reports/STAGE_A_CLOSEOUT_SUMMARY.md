# Stage A Closeout Summary

This report closes the representation pretest phase. It does not validate the final MIND detector.

## Status

- stage_a_closed: true
- stage_b_started: false
- panel_models: 16
- sphere_verdict: beneficial

## Canonical Tables

- metrics_long: outputs/stageA_closeout/reports/closeout_metrics_long.csv
- repope_classifier: outputs/stageA_closeout/reports/repope_main_table_classifier.csv
- repope_knn: outputs/stageA_closeout/reports/repope_main_table_knn.csv
- pope_secondary: outputs/stageA_closeout/reports/pope_secondary_table.csv
- dash_b_secondary: outputs/stageA_closeout/reports/dash_b_secondary_table.csv
- per_model_summary: outputs/stageA_closeout/reports/per_model_summary.csv

## Model Status

| model | status | reason |
| --- | --- | --- |
| llava-onevision-qwen2-7b-ov-hf | evaluated |  |
| glm-4.6v-flash | failed | glm-4.6v-flash/repope has no primary closeout population; glm-4.6v-flash/pope has no primary closeout population; glm-4.6v-flash/dash-b has no primary closeout population |
| qwen3-vl-8b | evaluated |  |
| internvl3.5-8b | evaluated |  |
| minicpm-v-2_6 | evaluated |  |
| gemma-3-12b-it | evaluated |  |
| gemma-4-12b-it | evaluated |  |
| qwen3.5-4b | evaluated |  |
| qwen3.5-9b | evaluated |  |
| phi-4-multimodal-instruct | evaluated |  |
| phi-3.5-vision-instruct | evaluated |  |
| gemma-3-4b-it | evaluated |  |
| molmo-7b-d-0924 | evaluated |  |
| minicpm-v-4_5 | evaluated |  |
| llava-v1.5-7b | evaluated |  |
| qwen2.5-vl-7b | evaluated |  |

## Verdict Inputs

- supporting_models: gemma-3-12b-it, gemma-3-4b-it, gemma-4-12b-it, llava-onevision-qwen2-7b-ov-hf, minicpm-v-2_6, minicpm-v-4_5, qwen3-vl-8b, qwen3.5-4b
- contradicting_models: internvl3.5-8b, llava-v1.5-7b, molmo-7b-d-0924, phi-3.5-vision-instruct, phi-4-multimodal-instruct, qwen2.5-vl-7b, qwen3.5-9b

Stage A is closed after Raw-Traj-LSTM is added and the closeout summary is written.
Later stages must not reopen Stage A except if the frozen theory note is explicitly revised.
