from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("tmp/asset_repair/repair_llava_v15_asset.py")
    spec = importlib.util.spec_from_file_location("repair_llava_v15_asset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llava_does_not_copy_onevision_metadata() -> None:
    module = _load_script()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "llava-onevision-qwen2-7b-ov-hf" not in source
    assert "copy" not in source.lower()


def test_llava_redownload_requires_exact_model_id(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    model_dir = tmp_path / "llava-v1.5-7b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llava"}', encoding="utf-8")
    monkeypatch.setattr(module, "LOCAL_PATH", model_dir)

    result = module.run_repair(
        execute=True,
        allow_install_tokenizer_deps=False,
        output_root=tmp_path / "repair",
    )

    assert result["status"] == "exact_model_id_required"
    assert "exact model id" in result["reason"]


def test_llava_tokenizer_deps_require_explicit_flag(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "LOCAL_PATH", tmp_path / "missing")
    monkeypatch.setattr(module, "missing_tokenizer_dependencies", lambda: ["protobuf", "tiktoken"])
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "run_command", lambda command, *, env: calls.append(command))

    result = module.run_repair(
        execute=True,
        allow_install_tokenizer_deps=False,
        output_root=tmp_path / "repair",
    )

    assert calls == []
    assert result["tokenizer_dependency_status"] == "missing_requires_explicit_flag"


def test_llava_incomplete_asset_reports_precise_missing_fields(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    model_dir = tmp_path / "llava-v1.5-7b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"llava","mm_vision_tower":"openai/clip-vit-large-patch14-336"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "LOCAL_PATH", model_dir)

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert result["status"] == "incomplete"
    assert "processor/image metadata" in result["reason"]
    assert "vision tower" in result["reason"]


def test_llava_hf_complete_layout_reports_complete(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    model_dir = tmp_path / "llava-1.5-7b-hf"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llava","architectures":["LlavaForConditionalGeneration"]}', encoding="utf-8")
    (model_dir / "processor_config.json").write_text('{"processor_class":"LlavaProcessor"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text('{"image_processor_type":"CLIPImageProcessor"}', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-00001.safetensors").write_text("", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"vision_tower.vision_model.embeddings.class_embedding":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "LOCAL_PATH", model_dir)
    monkeypatch.setattr(module, "missing_tokenizer_dependencies", lambda: [])

    result = module.run_repair(execute=False, output_root=tmp_path / "repair")

    assert result["status"] == "complete"
    assert result["inspection"]["local_path"].endswith("llava-1.5-7b-hf")
