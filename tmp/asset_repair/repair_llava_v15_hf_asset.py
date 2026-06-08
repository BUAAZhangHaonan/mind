#!/usr/bin/env python3
"""Check the LLaVA-v1.5 HF 7B local asset path.

This temporary repair script does not copy metadata from LLaVA-OneVision and
does not implement wrapper support. It only checks the registered HF 7B asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, normalize_mode


ALIAS = "llava-v1.5-7b"
EXPECTED_LOCAL_PATH = Path("/home/team/lvshuyang/Models/llava-1.5-7b-hf")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _registry_path(registry_path: Path) -> Path | None:
    if not registry_path.is_file():
        return None
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    for item in payload.get("models", []):
        if isinstance(item, dict) and item.get("alias") == ALIAS:
            value = item.get("local_path")
            return Path(value) if value else None
    return None


def _write_report(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "llava_v15_hf_asset_repair_report.json"
    md_path = output_root / "llava_v15_hf_asset_repair_report.md"
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = [
        "# LLaVA-v1.5 HF Asset Repair Report",
        "",
        f"- status: {report['status']}",
        f"- reason: {report['reason']}",
        f"- local_path: {report.get('local_path', '')}",
        f"- metadata_copied_from_onevision: {report['metadata_copied_from_onevision']}",
        "",
        "```json",
        json.dumps(report, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _has_tokenizer(path: Path) -> bool:
    return any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")) or (
        (path / "vocab.json").is_file() and (path / "merges.txt").is_file()
    )


def _weight_index(path: Path) -> dict[str, Any]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        payload = _read_json(path / name)
        if payload:
            return payload
    return {}


def _referenced_shards_exist(path: Path, index: dict[str, Any]) -> bool:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return bool(list(path.glob("*.safetensors")) or list(path.glob("*.bin")))
    return all((path / str(filename)).is_file() for filename in set(weight_map.values()))


def _has_vision_tower_weights(index: dict[str, Any]) -> bool:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return False
    return any("vision_tower" in str(key) or "vision_model" in str(key) for key in weight_map)


def _contains_onevision_marker(path: Path) -> bool:
    for name in ("config.json", "processor_config.json", "preprocessor_config.json"):
        file_path = path / name
        if file_path.is_file() and "onevision" in file_path.read_text(encoding="utf-8", errors="ignore").lower():
            return True
    return False


def repair_llava_v15_hf_asset(
    *,
    registry_path: Path = Path("configs/assets/model_assets.yaml"),
    output_root: Path = Path("outputs/assets/repair"),
    execute: bool = False,
    expected_local_path: Path = EXPECTED_LOCAL_PATH,
) -> dict[str, Any]:
    registered_path = _registry_path(registry_path)
    report: dict[str, Any] = {
        "model_alias": ALIAS,
        "status": "blocked_remove_from_panel",
        "reason": "",
        "mode": "execute" if execute else "dry_run",
        "registry_path": str(registry_path),
        "registered_local_path": str(registered_path) if registered_path else None,
        "expected_local_path": str(expected_local_path),
        "local_path": str(registered_path) if registered_path else None,
        "metadata_copied_from_onevision": False,
        "local_processor_load_possible": False,
        "local_model_load_possible": False,
        "standard_pipeline_is_authority": True,
    }
    if registered_path != expected_local_path:
        report["reason"] = f"registry path mismatch: expected {expected_local_path}, got {registered_path}"
        _write_report(output_root, report)
        return report

    local_path = registered_path
    assert local_path is not None
    missing: list[str] = []
    config = _read_json(local_path / "config.json")
    processor = _read_json(local_path / "processor_config.json")
    preprocessor = _read_json(local_path / "preprocessor_config.json")
    index = _weight_index(local_path)

    if not local_path.is_dir():
        missing.append("local path")
    if not config:
        missing.append("config.json")
    if not processor and not preprocessor:
        missing.append("processor metadata")
    if not _has_tokenizer(local_path):
        missing.append("tokenizer files")
    if not index:
        missing.append("safetensors index")
    if index and not _referenced_shards_exist(local_path, index):
        missing.append("model shards")
    if index and not _has_vision_tower_weights(index):
        missing.append("vision tower weights")
    report["metadata_copied_from_onevision"] = _contains_onevision_marker(local_path)
    if report["metadata_copied_from_onevision"]:
        missing.append("non-OneVision metadata")

    if missing:
        report["reason"] = "missing " + ", ".join(missing)
        _write_report(output_root, report)
        return report

    architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
    model_type = str(config.get("model_type", ""))
    processor_class = str(processor.get("processor_class") or "")
    image_processor_type = str(preprocessor.get("image_processor_type") or "")
    report.update(
        {
            "config_model_type": model_type,
            "architectures": architectures,
            "processor_class": processor_class,
            "image_processor_type": image_processor_type,
            "local_processor_load_possible": processor_class == "LlavaProcessor" or image_processor_type == "CLIPImageProcessor",
            "local_model_load_possible": "LlavaForConditionalGeneration" in architectures or model_type == "llava",
        }
    )
    if not report["local_processor_load_possible"]:
        report["reason"] = "local processor cannot be resolved from metadata"
        _write_report(output_root, report)
        return report
    if not report["local_model_load_possible"]:
        report["reason"] = "local model class cannot be resolved from metadata"
        _write_report(output_root, report)
        return report

    report["status"] = "ready_for_standard_pipeline"
    report["reason"] = "HF 7B asset metadata and local shard references are complete; standard smoke/validation remains the authority"
    _write_report(output_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--registry", type=Path, default=Path("configs/assets/model_assets.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = normalize_mode(build_parser().parse_args(argv))
    repair_llava_v15_hf_asset(registry_path=args.registry, output_root=args.output_root, execute=bool(args.execute))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
