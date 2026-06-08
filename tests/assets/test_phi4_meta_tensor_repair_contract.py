from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("tmp/asset_repair/repair_phi4_meta_tensor.py")
    spec = importlib.util.spec_from_file_location("repair_phi4_meta_tensor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_phi4_layout(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"phi4mm","architectures":["Phi4MMForCausalLM"]}', encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"synthetic")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )


def test_peft_install_requires_explicit_flag(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _write_phi4_layout(tmp_path)
    monkeypatch.setattr(module, "package_version", lambda name: None if name == "peft" else "1.0")

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=True, allow_install_peft=False)

    assert report["status"] == "blocked"
    assert "peft" in report["reason"]
    assert report["peft_install_attempted"] is False


def test_dry_run_reports_safe_loading_strategies_without_loading(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _write_phi4_layout(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(module, "run_load_strategy", lambda **kwargs: calls.append(kwargs) or {"status": "ok"})

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=False)

    assert calls == []
    assert report["load_diagnostics_ran"] is False
    assert {item["name"] for item in report["safe_loading_strategies"]} == {
        "low_cpu_mem_usage_false_device_map_none",
        "device_map_none",
        "device_map_auto",
    }
    assert (tmp_path / "repair" / "phi4_meta_tensor_repair_report.json").is_file()


def test_execute_detects_meta_tensor_parameter_names(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _write_phi4_layout(tmp_path)
    monkeypatch.setattr(module, "package_version", lambda name: "1.0")

    def fake_load_strategy(**kwargs):
        if kwargs["strategy"]["name"] == "low_cpu_mem_usage_false_device_map_none":
            return {"status": "meta_tensors_remaining", "meta_parameter_names": ["model.layers.0.weight"]}
        return {"status": "ok", "meta_parameter_names": []}

    monkeypatch.setattr(module, "run_load_strategy", fake_load_strategy)

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=True)

    assert report["status"] == "blocked"
    assert "meta tensors" in report["reason"]
    assert report["load_diagnostics"][0]["meta_parameter_names"] == ["model.layers.0.weight"]


def test_execute_reports_repaired_when_strategy_has_no_meta_tensors(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _write_phi4_layout(tmp_path)
    monkeypatch.setattr(module, "package_version", lambda name: "1.0")
    monkeypatch.setattr(module, "run_load_strategy", lambda **kwargs: {"status": "ok", "meta_parameter_names": []})

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=True)

    assert report["status"] == "load_strategy_available"
    assert "low_cpu_mem_usage_false_device_map_none" in report["reason"]


def test_production_wrapper_files_are_not_named_in_script() -> None:
    source = Path("tmp/asset_repair/repair_phi4_meta_tensor.py").read_text(encoding="utf-8")

    assert "src/mind/models/wrappers.py" not in source
    assert "src/mind/models/factory.py" not in source
