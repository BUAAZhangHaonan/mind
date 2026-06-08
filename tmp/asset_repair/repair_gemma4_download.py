#!/usr/bin/env python3
"""Temporary Gemma 4 asset localization repair for Experiment 1.7."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CommandResult, add_common_args, command_environment, file_list, read_json, run_command, write_model_report


ALIAS = "gemma-4-12b-it"
MODEL_ID = "google/gemma-4-12B-it"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/gemma-4-12B-it")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    return parser


def inspect_local_asset(local_path: Path = LOCAL_PATH) -> dict[str, object]:
    config = read_json(local_path / "config.json")
    return {
        "local_path": str(local_path),
        "path_exists": local_path.exists(),
        "path_is_directory": local_path.is_dir(),
        "config_exists": (local_path / "config.json").is_file(),
        "model_type": config.get("model_type", ""),
        "architectures": config.get("architectures", []),
        "tokenizer_files": file_list(local_path, ("tokenizer*", "*.model", "vocab.json", "merges.txt", "special_tokens_map.json")),
        "processor_files": file_list(local_path, ("processor*", "preprocessor*", "image_processor*", "chat_template*")),
        "image_processor_files": file_list(local_path, ("preprocessor_config.json", "image_processor_config.json")),
        "safetensors_files": file_list(local_path, ("*.safetensors", "model.safetensors.index.json")),
        "generation_config_exists": (local_path / "generation_config.json").is_file(),
        "chat_template_exists": any((local_path / name).is_file() for name in ("chat_template.json", "chat_template.jinja")),
        "model_size_files": file_list(local_path, ("*.safetensors", "*.bin", "*.index.json")),
        "complete": is_complete_asset(local_path),
    }


def is_complete_asset(local_path: Path) -> bool:
    if not local_path.is_dir():
        return False
    config = read_json(local_path / "config.json")
    if config.get("model_type") != "gemma4":
        return False
    has_image = "image_token_id" in config or "image_token_index" in config or "image" in {
        str(modality).lower() for modality in config.get("supported_modalities", [])
    }
    has_tokenizer_or_processor = bool(file_list(local_path, ("tokenizer*", "*.model", "processor_config.json")))
    has_image_processor = bool(file_list(local_path, ("preprocessor_config.json", "image_processor_config.json")))
    return bool(has_image and has_tokenizer_or_processor and has_image_processor and safetensors_complete(local_path))


def safetensors_complete(local_path: Path) -> bool:
    index = read_json(local_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        return all((local_path / str(filename)).is_file() for filename in set(weight_map.values()))
    return bool(file_list(local_path, ("*.safetensors",)))


def download_command(local_path: Path = LOCAL_PATH) -> list[str]:
    executable = "hf" if shutil.which("hf") else "huggingface-cli"
    command = [
        executable,
        "download",
        MODEL_ID,
        "--local-dir",
        str(local_path),
    ]
    if executable == "huggingface-cli":
        command.extend(["--local-dir-use-symlinks", "False"])
    return command


def command_environment(base_env: Mapping[str, str] | None = None) -> dict[str, str]:  # type: ignore[override]
    from common import command_environment as _command_environment

    return _command_environment(base_env if base_env is not None else os.environ, default_hf_endpoint=DEFAULT_HF_ENDPOINT)


def run_repair(*, execute: bool = False, output_root: Path = Path("outputs/assets/repair")) -> dict[str, object]:
    mode = "execute" if execute else "dry_run"
    inspection = inspect_local_asset(LOCAL_PATH)
    status = "already_present" if inspection["complete"] else "download_required"
    reason = "local Gemma 4 asset appears complete" if inspection["complete"] else "local Gemma 4 asset is missing or incomplete"
    result: CommandResult | None = None
    command = download_command(LOCAL_PATH)
    env = command_environment()

    if execute and status == "download_required":
        result = run_command(command, env=env)
        inspection = inspect_local_asset(LOCAL_PATH)
        status = "downloaded" if inspection["complete"] else "blocked_download_failed"
        reason = "download completed and local asset appears complete" if inspection["complete"] else "download did not produce a complete Gemma 4 asset"
        if result.returncode != 0:
            status = "blocked_download_failed"
            reason = f"download command failed with exit code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"

    report: dict[str, object] = {
        "alias": ALIAS,
        "status": status,
        "mode": mode,
        "dry_run": not execute,
        "execute": execute,
        "reason": reason,
        "model_id": MODEL_ID,
        "local_path": str(LOCAL_PATH),
        "download_command": command,
        "command_env_overrides": {"HF_ENDPOINT": env.get("HF_ENDPOINT", "")},
        "inspection": inspection,
        "normal_pipeline_required": True,
        "verification_authority": "normal asset audit/smoke/validation pipeline must pass before this model can be marked verified",
    }
    if result is not None:
        report["download_result"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    json_path, md_path = write_model_report(output_root=output_root, alias=ALIAS, report=report)
    if execute:
        manifest = output_root / f"{ALIAS}_download_manifest.json"
        manifest.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        report["download_manifest"] = str(manifest)
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execute = bool(args.execute)
    run_repair(execute=execute, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
