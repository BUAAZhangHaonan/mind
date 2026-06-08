from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.registry import AssetModel


def _asset(path: Path) -> AssetModel:
    return AssetModel.model_validate(
        {
            "alias": "phi-4-multimodal-instruct",
            "local_path": str(path),
            "model_config_path": "configs/models/phi_4_multimodal_instruct.yaml",
            "model_id_or_family_name": "phi4mm",
            "family": "phi4mm",
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
            "thinking": {"supported": False, "disabled_by_default": True, "disable_argument": None},
            "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
            "prompt_template_id": "phi4mm_single_image_raw_question_v1",
            "prompt_template_text": "Phi-4 multimodal single-image prompt.",
            "hidden_state_index_offset": 1,
        }
    )


def _write_phi4_layout(path: Path) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "phi4mm",
                "architectures": ["Phi4MMForCausalLM"],
                "img_processor": {"image_dim_out": 1024},
                "embd_layer": {"image_embd_layer": {"embedding_cls": "Phi4ImageEmbedding"}},
                "num_hidden_layers": 2,
                "hidden_size": 4,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "processor_config.json").write_text(
        json.dumps({"processor_class": "Phi4MMProcessor"}),
        encoding="utf-8",
    )
    (path / "preprocessor_config.json").write_text(
        json.dumps({"processor_class": "Phi4MMProcessor", "image_processor_type": "Phi4MMImageProcessor"}),
        encoding="utf-8",
    )


def test_missing_peft_blocks_phi4_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_phi4_layout(tmp_path)

    monkeypatch.setattr(
        "mind.models.asset_validation.importlib.util.find_spec",
        lambda name: None if name == "peft" else importlib.util.find_spec(name),
    )

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "missing dependency" in result.reason
    assert "peft" in result.reason


def test_asset_audit_does_not_install_peft_without_explicit_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_path = Path("scripts/asset_audit.py")
    spec = importlib.util.spec_from_file_location("asset_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = False

    def fail_install() -> None:
        nonlocal called
        called = True
        raise AssertionError("peft install must not run without --allow-install-peft")

    monkeypatch.setattr(module, "install_peft_dependency", fail_install)
    module.handle_optional_peft_install(allow_install_peft=False)

    assert called is False


def test_allow_install_peft_flag_is_required_for_install_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = Path("scripts/asset_audit.py")
    spec = importlib.util.spec_from_file_location("asset_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[str] = []

    monkeypatch.setattr(module, "peft_is_installed", lambda: False)
    monkeypatch.setattr(module, "install_peft_dependency", lambda: calls.append("install"))

    module.handle_optional_peft_install(allow_install_peft=False)
    assert calls == []

    module.handle_optional_peft_install(allow_install_peft=True)
    assert calls == ["install"]


def test_asset_audit_parser_defaults_to_no_peft_install() -> None:
    module_path = Path("scripts/asset_audit.py")
    spec = importlib.util.spec_from_file_location("asset_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(["--registry", "configs/assets/model_assets.yaml", "--output-root", "outputs/assets"])
    explicit = module.build_parser().parse_args(
        ["--registry", "configs/assets/model_assets.yaml", "--output-root", "outputs/assets", "--allow-install-peft"]
    )

    assert args.allow_install_peft is False
    assert explicit.allow_install_peft is True
