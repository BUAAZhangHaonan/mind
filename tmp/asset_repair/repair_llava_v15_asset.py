#!/usr/bin/env python3
"""Temporary LLaVA-v1.5 asset repair diagnostics for Experiment 1.7."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CommandResult, add_common_args, command_environment, file_list, read_json, run_command, write_model_report


ALIAS = "llava-v1.5-7b"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/llava-v1.5-7b")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
TOKENIZER_DEPS = ("protobuf", "tiktoken")


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--allow-install-tokenizer-deps", action="store_true", default=False)
    return parser


def inspect_local_asset(local_path: Path = LOCAL_PATH) -> dict[str, object]:
    config = read_json(local_path / "config.json")
    processor_files = file_list(local_path, ("processor_config.json", "preprocessor_config.json", "image_processor_config.json", "chat_template*"))
    weight_files = file_list(local_path, ("*.safetensors", "*.bin", "*.index.json"))
    return {
        "local_path": str(local_path),
        "path_exists": local_path.exists(),
        "path_is_directory": local_path.is_dir(),
        "config_exists": (local_path / "config.json").is_file(),
        "model_type": config.get("model_type", ""),
        "architectures": config.get("architectures", []),
        "tokenizer_files": file_list(local_path, ("tokenizer*", "*.model", "vocab.json", "merges.txt")),
        "processor_files": processor_files,
        "image_processor_files": file_list(local_path, ("preprocessor_config.json", "image_processor_config.json")),
        "vision_tower_reference": config.get("mm_vision_tower", ""),
        "vision_tower_files": vision_tower_files(local_path),
        "weight_files": weight_files,
        "missing_metadata": missing_metadata(local_path),
        "missing_tokenizer_dependencies": missing_tokenizer_dependencies(),
        "exact_model_id": exact_model_id(config),
    }


def vision_tower_files(local_path: Path) -> list[str]:
    index = read_json(local_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return []
    return sorted(name for name in weight_map if str(name).startswith("model.vision_tower"))


def missing_metadata(local_path: Path) -> list[str]:
    missing: list[str] = []
    if not any((local_path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        missing.append("processor/image metadata")
    if not any((local_path / name).is_file() for name in ("preprocessor_config.json", "image_processor_config.json")):
        missing.append("image processor metadata")
    if not vision_tower_files(local_path):
        missing.append("vision tower")
    return missing


def missing_tokenizer_dependencies() -> list[str]:
    return [name for name in TOKENIZER_DEPS if importlib.util.find_spec(name) is None]


def exact_model_id(config: dict[str, object]) -> str:
    candidate = str(config.get("_name_or_path") or config.get("model_id") or "")
    if candidate and candidate.count("/") == 1 and "llava" in candidate.lower() and "v1.5" in candidate.lower():
        return candidate
    return ""


def classify(inspection: dict[str, object]) -> tuple[str, str]:
    if not inspection["path_exists"]:
        return "exact_model_id_required", "local path is missing and exact model id cannot be determined from local config"
    missing = inspection.get("missing_metadata", [])
    if missing:
        return "incomplete", "local LLaVA-v1.5 asset is incomplete: missing " + ", ".join(str(item) for item in missing)
    return "complete", "local LLaVA-v1.5 asset appears complete"


def download_command(model_id: str, local_path: Path = LOCAL_PATH) -> list[str]:
    return ["hf", "download", model_id, "--local-dir", str(local_path)]


def tokenizer_install_command(deps: list[str]) -> list[str]:
    return [sys.executable, "-m", "pip", "install", *deps]


def run_repair(
    *,
    execute: bool = False,
    allow_install_tokenizer_deps: bool = False,
    output_root: Path = Path("outputs/assets/repair"),
) -> dict[str, object]:
    inspection = inspect_local_asset(LOCAL_PATH)
    status, reason = classify(inspection)
    tokenizer_deps = list(inspection.get("missing_tokenizer_dependencies", []))
    tokenizer_dependency_status = "present"
    command_results: list[dict[str, object]] = []

    if tokenizer_deps:
        tokenizer_dependency_status = "missing_requires_explicit_flag"
        if execute and allow_install_tokenizer_deps:
            result = run_command(tokenizer_install_command(tokenizer_deps), env=command_environment(default_hf_endpoint=DEFAULT_HF_ENDPOINT))
            command_results.append({"command": tokenizer_install_command(tokenizer_deps), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            tokenizer_dependency_status = "install_attempted"

    model_id = str(inspection.get("exact_model_id", ""))
    if execute and status in {"incomplete", "exact_model_id_required"}:
        if not model_id:
            status = "exact_model_id_required"
            reason = "exact model id is required before redownloading LLaVA-v1.5; local config does not prove it"
        else:
            result = run_command(download_command(model_id), env=command_environment(default_hf_endpoint=DEFAULT_HF_ENDPOINT))
            command_results.append({"command": download_command(model_id), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            inspection = inspect_local_asset(LOCAL_PATH)
            status, reason = classify(inspection)

    report: dict[str, object] = {
        "alias": ALIAS,
        "status": status,
        "mode": "execute" if execute else "dry_run",
        "dry_run": not execute,
        "execute": execute,
        "allow_install_tokenizer_deps": allow_install_tokenizer_deps,
        "reason": reason,
        "local_path": str(LOCAL_PATH),
        "inspection": inspection,
        "tokenizer_dependency_status": tokenizer_dependency_status,
        "normal_pipeline_required": True,
        "verification_authority": "normal asset audit/smoke/validation pipeline must pass before this model can be marked verified",
        "command_results": command_results,
    }
    json_path, md_path = write_model_report(output_root=output_root, alias=ALIAS, report=report)
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_repair(
        execute=bool(args.execute),
        allow_install_tokenizer_deps=bool(args.allow_install_tokenizer_deps),
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
