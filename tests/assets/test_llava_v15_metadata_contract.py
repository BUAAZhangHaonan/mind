from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mind.models.asset_validation import AssetStatus, audit_asset_metadata
from mind.models.registry import AssetModel


def _asset(path: Path, **overrides: object) -> AssetModel:
    payload = {
        "alias": "llava-v1.5-7b",
        "local_path": str(path),
        "model_config_path": "configs/models/llava_v1_5_7b.yaml",
        "model_id_or_family_name": "llava_v15",
        "family": "llava_v15",
        "dtype": "float16",
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "deterministic_generation": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking": {"supported": False, "disabled_by_default": True, "disable_argument": None},
        "policy": {"allow_moe": False, "allow_thinking": False, "allow_video_only": False, "allow_audio_only": False},
        "prompt_template_id": "llava_v15_single_image_raw_question_v1",
        "prompt_template_text": "LLaVA v1.5 single-image prompt.",
        "hidden_state_index_offset": "unknown",
    }
    payload.update(overrides)
    return AssetModel.model_validate(payload)


def _write_incomplete_llava_layout(path: Path) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llava",
                "architectures": ["LlavaLlamaForCausalLM"],
                "mm_vision_tower": "openai/clip-vit-large-patch14-336",
                "num_hidden_layers": 2,
                "hidden_size": 4,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_missing_processor_tokenizer_metadata_blocks_llava_v15(tmp_path: Path) -> None:
    _write_incomplete_llava_layout(tmp_path)

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "processor/image processor metadata is missing" in result.reason


def test_llava_v15_metadata_is_not_copied_from_onevision() -> None:
    module_path = Path("scripts/asset_audit.py")
    source = module_path.read_text(encoding="utf-8")

    assert "llava-onevision-qwen2-7b-ov-hf" not in source
    assert "copy" not in source.lower() or "llava_v15" not in source


def test_llava_v15_network_repair_requires_explicit_flag() -> None:
    module_path = Path("scripts/asset_audit.py")
    spec = importlib.util.spec_from_file_location("asset_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(["--registry", "configs/assets/model_assets.yaml", "--output-root", "outputs/assets"])
    explicit = module.build_parser().parse_args(
        [
            "--registry",
            "configs/assets/model_assets.yaml",
            "--output-root",
            "outputs/assets",
            "--repair-llava-v1-5-metadata",
        ]
    )

    assert args.repair_llava_v1_5_metadata is False
    assert explicit.repair_llava_v1_5_metadata is True


def test_incomplete_llava_v15_local_asset_remains_blocked(tmp_path: Path) -> None:
    _write_incomplete_llava_layout(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.mm_projector.weight": "model.safetensors"}}),
        encoding="utf-8",
    )

    result = audit_asset_metadata(_asset(tmp_path))

    assert result.status == AssetStatus.BLOCKED
    assert "vision tower weights are not included" in result.reason


def test_final_closure_inspection_preserves_llava_v15_blocker(tmp_path: Path) -> None:
    _write_incomplete_llava_layout(tmp_path)
    module_path = Path("scripts/asset_audit.py")
    spec = importlib.util.spec_from_file_location("asset_audit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = module.build_final_asset_closure_inspection([_asset(tmp_path)], gemma4_preflight={})

    assert rows[0]["status"] == AssetStatus.BLOCKED.value
    assert "processor/image processor metadata is missing" in rows[0]["reason"]
