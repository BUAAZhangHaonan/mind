from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("tmp/asset_repair/repair_phi4_peft.py")
    spec = importlib.util.spec_from_file_location("repair_phi4_peft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_peft_reports_planned_install(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "package_version", lambda name: None if name == "peft" else "1.0")
    monkeypatch.setattr(module, "run_command", lambda command: (_ for _ in ()).throw(AssertionError("dry-run must not call pip")))

    result = module.run_repair(execute=False, allow_install_peft=False, output_root=tmp_path / "repair")

    assert result["status"] == "install_required"
    assert result["planned_command"] == [module.python_executable(), "-m", "pip", "install", "peft"]
    assert result["preflight_plan"]["subprocess_executed"] is False
    assert result["normal_pipeline_required"] is True
    assert "normal asset audit/smoke/validation" in result["verification_authority"]
    assert (tmp_path / "repair" / "phi-4-multimodal-instruct_repair_report.json").is_file()


def test_installation_is_not_automatic_without_allow_flag(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "package_version", lambda name: None if name == "peft" else "1.0")
    monkeypatch.setattr(module, "run_command", lambda command: calls.append(command))

    result = module.run_repair(execute=True, allow_install_peft=False, output_root=tmp_path / "repair")

    assert calls == []
    assert result["status"] == "blocked_missing_allow_install_peft"


def test_allow_install_peft_is_required_for_install_attempt(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "package_version", lambda name: None if name == "peft" else "1.0")
    monkeypatch.setattr(module, "run_command", lambda command: calls.append(command) or module.CommandResult(0, "ok", ""))

    result = module.run_repair(execute=True, allow_install_peft=True, output_root=tmp_path / "repair")

    assert calls == [[module.python_executable(), "-m", "pip", "install", "peft"]]
    assert result["status"] in {"installed", "install_completed_but_peft_not_detected"}
