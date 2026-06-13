from __future__ import annotations

from .conftest import stage_d_attr


def test_stage_d_family_mapping_covers_all_panel_families() -> None:
    family_contract = stage_d_attr("stage_d_status", "stage_d_family_contract")()

    assert family_contract["families"] == [
        "qwen",
        "internvl",
        "llava",
        "gemma",
        "phi",
        "minicpm",
        "glm",
        "molmo",
    ]
    assert family_contract["model_to_family"]("qwen3-vl-8b") == "qwen"
    assert family_contract["model_to_family"]("internvl3.5-8b") == "internvl"
    assert family_contract["model_to_family"]("llava-v1.5-7b") == "llava"
    assert family_contract["model_to_family"]("gemma-4-12b-it") == "gemma"
    assert family_contract["model_to_family"]("phi-4-multimodal-instruct") == "phi"
    assert family_contract["model_to_family"]("minicpm-v-4_5") == "minicpm"
    assert family_contract["model_to_family"]("glm-4.6v-flash") == "glm"
    assert family_contract["model_to_family"]("molmo-7b-d-0924") == "molmo"


def test_stage_d_family_summary_keeps_glm_and_molmo_status() -> None:
    build_summary = stage_d_attr("stage_d_status", "build_stage_d_family_summary")

    rows = build_summary(
        panel_models=["glm-4.6v-flash", "molmo-7b-d-0924", "qwen3-vl-8b"],
        metric_rows=[
            _metric("molmo-7b-d-0924", "MIND-main", 0.4),
            _metric("qwen3-vl-8b", "MIND-main", 0.5),
        ],
        excluded_models={"glm-4.6v-flash": "answer format incompatible with frozen yes/no population rule"},
        separate_env_models={"molmo-7b-d-0924"},
    )

    by_family = {row["family"]: row for row in rows}
    assert by_family["glm"]["num_panel_models"] == 1
    assert by_family["glm"]["num_evaluable_models"] == 0
    assert "answer format incompatible" in by_family["glm"]["family_specific_notes"]
    assert by_family["molmo"]["main_env_vs_separate_env_note"] == "contains verified_separate_env model"
    assert by_family["qwen"]["num_evaluable_models"] == 1


def _metric(model: str, method: str, pr_auc: float) -> dict[str, object]:
    return {
        "model_alias": model,
        "method": method,
        "pr_auc": pr_auc,
        "metric_status": "passed",
        "target_dataset": "repope",
        "protocol": "repope_to_repope",
    }
