from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_module():
    path = Path("tmp/asset_repair/run_gemma4_unified_tmp_smoke.py")
    spec = importlib.util.spec_from_file_location("run_gemma4_unified_tmp_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_gemma4_asset(path: Path) -> None:
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
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4UnifiedProcessor",
                "image_seq_length": 280,
                "feature_extractor": {"feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor"},
                "image_processor": {"image_processor_type": "Gemma4UnifiedImageProcessor"},
                "video_processor": {"video_processor_type": "Gemma4UnifiedVideoProcessor"},
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"processor_class": "Gemma4UnifiedProcessor", "image_token": "<|image|>"}),
        encoding="utf-8",
    )
    (path / "chat_template.jinja").write_text("{{ enable_thinking }} <|image|>", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"synthetic")


def test_dry_run_records_gemma4_unified_policy_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    local_path = tmp_path / "gemma4"
    _write_gemma4_asset(local_path)

    monkeypatch.setattr(module, "inspect_safetensors_keys", lambda path: ["model.language_model.embed_tokens.weight", "model.vision_embedder.patch_dense.weight"])
    monkeypatch.setattr(
        module,
        "inspect_transformers_runtime",
        lambda local_path: {
            "transformers_version": "test",
            "has_gemma4_unified_module": False,
            "has_gemma4_unified_model_class": False,
            "auto_config_status": "failed",
            "auto_config_error": "model type `gemma4_unified` not recognized",
        },
    )
    monkeypatch.setattr(
        module,
        "inspect_processor_wiring",
        lambda *args, **kwargs: {
            "status": "processor_wired",
            "prompt_template_id": module.PROMPT_TEMPLATE_ID,
            "enable_thinking": False,
            "contains_image_token": True,
            "contains_think_token": False,
            "image_token_count": 260,
            "pixel_values_shape": [1, 2520, 768],
        },
    )

    report = module.run_tmp_smoke(local_path=local_path, output_root=tmp_path / "repair", execute=False)

    assert report["alias"] == "gemma-4-12b-it"
    assert report["family"] == "gemma4_unified"
    assert report["thinking_disabled"] is True
    assert report["vision_tower_check_used"] is False
    assert report["status"] == "blocked_missing_transformers_gemma4_unified_support"
    assert "gemma4_unified" in report["reason"]
    assert report["non_unified_class_incompatibility"]["checkpoint_uses_unified_prefixes"] is True
    assert (tmp_path / "repair" / "gemma4_unified_tmp_smoke_report.json").is_file()


def test_processor_wiring_failure_blocks_before_model_loading(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    local_path = tmp_path / "gemma4"
    _write_gemma4_asset(local_path)

    monkeypatch.setattr(module, "inspect_safetensors_keys", lambda path: ["model.language_model.embed_tokens.weight"])
    monkeypatch.setattr(
        module,
        "inspect_transformers_runtime",
        lambda local_path: {"has_gemma4_unified_module": True, "has_gemma4_unified_model_class": True, "auto_config_status": "loaded"},
    )
    monkeypatch.setattr(
        module,
        "inspect_processor_wiring",
        lambda *args, **kwargs: {
            "status": "blocked",
            "reason": "image token was not inserted",
            "enable_thinking": False,
            "contains_image_token": False,
            "image_token_count": 0,
        },
    )

    report = module.run_tmp_smoke(local_path=local_path, output_root=tmp_path / "repair", execute=False)

    assert report["status"] == "blocked_processor_wiring"
    assert "image token" in report["reason"]


def test_execute_does_not_load_model_when_runtime_lacks_unified_support(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    local_path = tmp_path / "gemma4"
    _write_gemma4_asset(local_path)

    monkeypatch.setattr(module, "inspect_safetensors_keys", lambda path: ["model.language_model.embed_tokens.weight", "model.vision_embedder.patch_dense.weight"])
    monkeypatch.setattr(
        module,
        "inspect_transformers_runtime",
        lambda local_path: {"has_gemma4_unified_module": False, "has_gemma4_unified_model_class": False, "auto_config_status": "failed"},
    )
    monkeypatch.setattr(module, "inspect_processor_wiring", lambda *args, **kwargs: {"status": "processor_wired", "enable_thinking": False, "contains_image_token": True, "image_token_count": 260})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model load should not be attempted without gemma4_unified runtime support")

    monkeypatch.setattr(module, "run_supported_smoke", fail_if_called)

    report = module.run_tmp_smoke(local_path=local_path, output_root=tmp_path / "repair", execute=True)

    assert report["status"] == "blocked_missing_transformers_gemma4_unified_support"


def test_loader_requires_actual_gemma4_unified_processor(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    class FakeUnifiedProcessor:
        def __init__(self):
            self.image_processor = Gemma4UnifiedImageProcessor()

    class Gemma4UnifiedImageProcessor:
        pass

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(local_path, *, local_files_only):
            assert local_files_only is True
            return FakeUnifiedProcessor()

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=FakeAutoProcessor,
        Gemma4UnifiedProcessor=FakeUnifiedProcessor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    processor = module._load_unified_processor(tmp_path)

    assert isinstance(processor, FakeUnifiedProcessor)


def test_unified_batch_validation_rejects_old_processor_pixel_shape() -> None:
    module = _load_module()

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    result = module.validate_unified_processor_batch(
        {
            "input_ids": FakeTensor((1, 283)),
            "pixel_values": FakeTensor((1, 2520, 768)),
            "image_position_ids": FakeTensor((1, 2520, 2)),
            "mm_token_type_ids": FakeTensor((1, 283)),
        }
    )

    assert result["status"] == "blocked"
    assert "6912" in result["reason"]
