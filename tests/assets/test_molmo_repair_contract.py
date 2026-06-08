from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("tmp/asset_repair/repair_molmo_asset.py")
    spec = importlib.util.spec_from_file_location("repair_molmo_asset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_molmo_diagnostics_stay_under_tmp_and_repair_outputs(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "LOCAL_PATH", tmp_path / "Molmo-7B-D-0924")

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert str(result["diagnostic_script"]).startswith("tmp/asset_repair/")
    assert str(result["report_json"]).startswith(str(tmp_path / "repair"))


def test_molmo_does_not_patch_production_wrapper_source() -> None:
    module = _load_script()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "src/mind/models/wrappers.py" not in source
    assert "PreTrainedModel.all_tied_weights_keys" not in source


def test_molmo_complete_but_incompatible_reports_remote_code_incompatible(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    model_dir = tmp_path / "Molmo-7B-D-0924"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"molmo","architectures":["MolmoForCausalLM"]}', encoding="utf-8")
    (model_dir / "modeling_molmo.py").write_text(
        "class MolmoForCausalLM:\n"
        "    def forward(self): pass\n"
        "    def prepare_inputs_for_generation(self): pass\n",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "preprocessing_molmo.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "LOCAL_PATH", model_dir)

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert result["status"] == "remote_code_incompatible"
    assert "_extract_generation_mode_kwargs" in result["reason"]


def test_normal_pipeline_remains_verification_authority(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "LOCAL_PATH", tmp_path / "missing")

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert result["normal_pipeline_required"] is True
    assert "normal asset audit/smoke/validation" in result["verification_authority"]
