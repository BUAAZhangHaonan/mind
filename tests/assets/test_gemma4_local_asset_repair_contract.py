from __future__ import annotations

import importlib.util
import json
from pathlib import Path


EXPECTED_SHA = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"


def _load_module():
    path = Path("tmp/asset_repair/repair_gemma4_local_asset.py")
    spec = importlib.util.spec_from_file_location("repair_gemma4_local_asset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_complete_gemma4(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "image_token_id": 258880,
                "text_config": {
                    "num_hidden_layers": 48,
                    "hidden_size": 3840,
                    "enable_moe_block": False,
                    "num_experts": None,
                    "top_k_experts": None,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4UnifiedProcessor",
                "image_processor": {"image_processor_type": "Gemma4UnifiedImageProcessor"},
            }
        ),
        encoding="utf-8",
    )
    (path / "chat_template.jinja").write_text("{{ enable_thinking }}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"synthetic")


def test_existing_model_safetensors_with_expected_sha_is_accepted(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _write_complete_gemma4(tmp_path)
    monkeypatch.setattr(module, "sha256_file", lambda path: EXPECTED_SHA)
    monkeypatch.setattr(module, "inspect_safetensors", lambda path: {"readable": True, "tensor_key_count": 3})

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=False)

    assert report["status"] == "already_present"
    assert report["inspection"]["sha256_matches"] is True
    assert report["inspection"]["tensor_key_count"] == 3
    assert report["inspection"]["thinking_disable_evidence"] == "enable_thinking=False"
    assert (tmp_path / "repair" / "gemma4_local_asset_repair_report.json").is_file()


def test_missing_metadata_triggers_metadata_repair_plan_without_weight_redownload(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"model_type":"gemma4_unified","image_token_id":1}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic")
    monkeypatch.setattr(module, "sha256_file", lambda path: EXPECTED_SHA)
    monkeypatch.setattr(module, "inspect_safetensors", lambda path: {"readable": True, "tensor_key_count": 1})

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=False)

    assert report["status"] == "metadata_repair_required"
    assert "metadata" in report["reason"]
    assert report["download_plan"]["model_id"] == "google/gemma-4-12B-it"
    assert report["download_plan"]["exclude"] == ["model.safetensors"]


def test_active_moe_indicators_block_gemma4(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _write_complete_gemma4(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    payload["text_config"]["enable_moe_block"] = True
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "sha256_file", lambda path: EXPECTED_SHA)
    monkeypatch.setattr(module, "inspect_safetensors", lambda path: {"readable": True, "tensor_key_count": 1})

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=False)

    assert report["status"] == "blocked"
    assert "MoE indicators" in report["reason"]


def test_execute_metadata_download_does_not_redownload_matching_weights(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"model_type":"gemma4_unified","image_token_id":1}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic")
    monkeypatch.setattr(module, "sha256_file", lambda path: EXPECTED_SHA)
    monkeypatch.setattr(module, "inspect_safetensors", lambda path: {"readable": True, "tensor_key_count": 1})
    calls: list[list[str]] = []

    def fake_run(command, *, env):
        calls.append(command)
        return module.CommandResult(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module, "run_command", fake_run)

    report = module.run_repair(local_path=tmp_path, output_root=tmp_path / "repair", execute=True)

    assert calls
    command_text = " ".join(calls[0])
    assert "google/gemma-4-12B-it" in command_text
    assert "model.safetensors" in command_text
    assert report["status"] in {"metadata_repair_required", "already_present", "blocked"}
