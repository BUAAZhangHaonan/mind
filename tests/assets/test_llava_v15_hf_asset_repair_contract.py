from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("tmp/asset_repair/repair_llava_v15_hf_asset.py")
    spec = importlib.util.spec_from_file_location("repair_llava_v15_hf_asset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path, local_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "models:",
                "  - alias: llava-v1.5-7b",
                f"    local_path: {local_path}",
                "    family: llava_v15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_complete_llava_hf(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "llava", "architectures": ["LlavaForConditionalGeneration"]}),
        encoding="utf-8",
    )
    (path / "processor_config.json").write_text(json.dumps({"processor_class": "LlavaProcessor"}), encoding="utf-8")
    (path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "CLIPImageProcessor"}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"synthetic")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 9},
                "weight_map": {
                    "vision_tower.vision_model.embeddings.patch_embedding.weight": "model-00001-of-00001.safetensors",
                    "language_model.model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )


def test_complete_hf_7b_path_is_ready_for_standard_pipeline(tmp_path: Path) -> None:
    module = _load_module()
    local_path = tmp_path / "llava-1.5-7b-hf"
    registry = tmp_path / "model_assets.yaml"
    output_root = tmp_path / "repair"
    _write_complete_llava_hf(local_path)
    _write_registry(registry, local_path)

    report = module.repair_llava_v15_hf_asset(
        registry_path=registry,
        output_root=output_root,
        execute=True,
        expected_local_path=local_path,
    )

    assert report["status"] == "ready_for_standard_pipeline"
    assert report["metadata_copied_from_onevision"] is False
    assert report["local_processor_load_possible"] is True
    assert report["local_model_load_possible"] is True
    assert (output_root / "llava_v15_hf_asset_repair_report.json").is_file()


def test_registry_must_point_to_hf_7b_path(tmp_path: Path) -> None:
    module = _load_module()
    local_path = tmp_path / "llava-1.5-7b-hf"
    wrong_path = tmp_path / "llava-v1.5-7b"
    registry = tmp_path / "model_assets.yaml"
    _write_complete_llava_hf(local_path)
    _write_registry(registry, wrong_path)

    report = module.repair_llava_v15_hf_asset(
        registry_path=registry,
        output_root=tmp_path / "repair",
        execute=True,
        expected_local_path=local_path,
    )

    assert report["status"] == "blocked_remove_from_panel"
    assert "registry path mismatch" in report["reason"]


def test_incomplete_hf_path_fails_without_copying_onevision_metadata(tmp_path: Path) -> None:
    module = _load_module()
    local_path = tmp_path / "llava-1.5-7b-hf"
    registry = tmp_path / "model_assets.yaml"
    local_path.mkdir(parents=True)
    (local_path / "config.json").write_text(json.dumps({"model_type": "llava"}), encoding="utf-8")
    _write_registry(registry, local_path)

    report = module.repair_llava_v15_hf_asset(
        registry_path=registry,
        output_root=tmp_path / "repair",
        execute=True,
        expected_local_path=local_path,
    )

    assert report["status"] == "blocked_remove_from_panel"
    assert report["metadata_copied_from_onevision"] is False
    assert "missing processor metadata" in report["reason"]
