from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("tmp/asset_repair/final_model_panel_decisions.py")
    spec = importlib.util.spec_from_file_location("final_model_panel_decisions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_final_decisions_classify_all_remaining_models(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "repair"
    output_root.mkdir(parents=True)
    molmo_manifest = output_root / "molmo_separate_env_acceptance.json"
    molmo_manifest.write_text(
        json.dumps({"model_alias": "molmo-7b-d-0924", "status": "verified_separate_env"}),
        encoding="utf-8",
    )

    report = module.write_final_model_panel_decisions(
        output_root=output_root,
        molmo_acceptance_path=molmo_manifest,
        execute=True,
    )

    decisions = report["decisions"]
    assert set(decisions) == {
        "gemma-4-12b-it",
        "phi-4-multimodal-instruct",
        "molmo-7b-d-0924",
        "llava-v1.5-7b",
    }
    assert decisions["molmo-7b-d-0924"]["classification"] == "verified_separate_env"
    assert decisions["gemma-4-12b-it"]["classification"] == "blocked_manual_future_work"
    assert decisions["phi-4-multimodal-instruct"]["classification"] == "blocked_remove_from_panel"
    assert decisions["llava-v1.5-7b"]["classification"] == "blocked_remove_from_panel"
    assert (output_root / "final_model_panel_decisions.json").is_file()
    assert (output_root / "FINAL_MODEL_PANEL_DECISIONS.md").is_file()


def test_missing_molmo_acceptance_blocks_molmo_removal_decision(tmp_path: Path) -> None:
    module = _load_module()

    report = module.write_final_model_panel_decisions(output_root=tmp_path / "repair", execute=True)

    assert report["decisions"]["molmo-7b-d-0924"]["classification"] == "blocked_remove_from_panel"
    assert "separate-env acceptance manifest is missing" in report["decisions"]["molmo-7b-d-0924"]["reason"]


def test_decision_classes_are_distinct(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "repair"
    output_root.mkdir(parents=True)
    molmo_manifest = output_root / "molmo_separate_env_acceptance.json"
    molmo_manifest.write_text(json.dumps({"status": "verified_separate_env"}), encoding="utf-8")

    report = module.write_final_model_panel_decisions(
        output_root=output_root,
        molmo_acceptance_path=molmo_manifest,
        execute=True,
    )

    classes = {item["classification"] for item in report["decisions"].values()}
    assert {"verified_separate_env", "blocked_remove_from_panel", "blocked_manual_future_work"} <= classes
