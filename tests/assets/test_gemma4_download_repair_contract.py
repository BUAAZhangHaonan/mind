from __future__ import annotations

import importlib.util
from pathlib import Path
import json


def _load_script():
    path = Path("tmp/asset_repair/repair_gemma4_download.py")
    spec = importlib.util.spec_from_file_location("repair_gemma4_download", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_local_path_reports_download_required(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "LOCAL_PATH", tmp_path / "gemma-4-12B-it")

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert result["status"] == "download_required"
    assert result["execute"] is False
    assert result["normal_pipeline_required"] is True
    assert "normal asset audit/smoke/validation" in result["verification_authority"]
    assert (tmp_path / "repair" / "gemma-4-12b-it_repair_report.json").is_file()


def test_execute_download_uses_only_allowed_model_and_destination(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(module, "LOCAL_PATH", tmp_path / "gemma-4-12B-it")
    monkeypatch.setattr(module, "is_complete_asset", lambda path: False)

    def fake_run(command, *, env):
        calls.append((command, env))
        return module.CommandResult(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module, "run_command", fake_run)

    module.run_repair(execute=True, output_root=tmp_path / "repair")

    command, env = calls[0]
    assert "google/gemma-4-12B-it" in command
    assert str(tmp_path / "gemma-4-12B-it") in command
    assert "llava" not in " ".join(command).lower()
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_hf_endpoint_mirror_is_command_scoped(monkeypatch) -> None:
    module = _load_script()
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    env = module.command_environment({})

    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_uploaded_gemma4_unified_layout_reports_already_present(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    local_path = tmp_path / "gemma-4-12B-it"
    local_path.mkdir()
    (local_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "image_token_id": 258880,
            }
        ),
        encoding="utf-8",
    )
    (local_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (local_path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4UnifiedProcessor",
                "image_processor": {"image_processor_type": "Gemma4UnifiedImageProcessor"},
            }
        ),
        encoding="utf-8",
    )
    (local_path / "model.safetensors").write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "LOCAL_PATH", local_path)

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert result["status"] == "already_present"
    assert result["inspection"]["complete"] is True
    assert "processor_config.json:image_processor" in result["inspection"]["image_processor_files"]
