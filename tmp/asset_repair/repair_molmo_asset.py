#!/usr/bin/env python3
"""Temporary Molmo asset repair diagnostics for Experiment 1.7."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import re
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CommandResult, add_common_args, command_environment, file_list, read_json, run_command, write_model_report


ALIAS = "molmo-7b-d-0924"
MODEL_ID = "allenai/Molmo-7B-D-0924"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/Molmo-7B-D-0924")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DIAGNOSTIC_SCRIPT = Path("tmp/asset_repair/repair_molmo_asset.py")


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    return parser


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_local_asset(local_path: Path = LOCAL_PATH) -> dict[str, object]:
    model_code = ""
    code_path = local_path / "modeling_molmo.py"
    if code_path.is_file():
        model_code = code_path.read_text(encoding="utf-8", errors="ignore")
    config = read_json(local_path / "config.json")
    return {
        "local_path": str(local_path),
        "path_exists": local_path.exists(),
        "path_is_directory": local_path.is_dir(),
        "config_exists": (local_path / "config.json").is_file(),
        "model_type": config.get("model_type", ""),
        "architectures": config.get("architectures", []),
        "auto_map": config.get("auto_map", {}),
        "remote_code_files": file_list(local_path, ("modeling_molmo.py", "configuration_molmo.py", "preprocessing_molmo.py", "image_preprocessing_molmo.py")),
        "safetensors_files": file_list(local_path, ("*.safetensors", "model.safetensors.index.json")),
        "tokenizer_files": file_list(local_path, ("tokenizer*", "*.model", "vocab.json", "merges.txt")),
        "processor_files": file_list(local_path, ("processor*", "preprocessor*", "image_preprocessing_molmo.py", "preprocessing_molmo.py")),
        "transformers_version": package_version("transformers"),
        "defines_all_tied_weights_keys": "all_tied_weights_keys" in model_code,
        "defines_extract_generation_mode_kwargs": "_extract_generation_mode_kwargs" in model_code,
        "defines_generate": bool(re.search(r"def\s+generate\s*\(", model_code)),
        "defines_generate_from_batch": "def generate_from_batch" in model_code,
        "defines_forward": bool(re.search(r"def\s+forward\s*\(", model_code)),
        "defines_prepare_inputs_for_generation": "def prepare_inputs_for_generation" in model_code,
        "complete": is_complete_asset(local_path),
    }


def is_complete_asset(local_path: Path) -> bool:
    return bool(
        local_path.is_dir()
        and (local_path / "config.json").is_file()
        and (local_path / "modeling_molmo.py").is_file()
        and file_list(local_path, ("*.safetensors", "model.safetensors.index.json"))
        and file_list(local_path, ("tokenizer*",))
        and (local_path / "preprocessing_molmo.py").is_file()
    )


def download_command(local_path: Path = LOCAL_PATH) -> list[str]:
    return ["hf", "download", MODEL_ID, "--local-dir", str(local_path)]


def classify(inspection: dict[str, object]) -> tuple[str, str]:
    if not inspection["path_exists"]:
        return "download_required", "local Molmo path does not exist"
    if not inspection["complete"]:
        return "incomplete", "local Molmo asset appears incomplete"
    if not inspection["defines_extract_generation_mode_kwargs"]:
        return "remote_code_incompatible", "Molmo remote code is complete but lacks _extract_generation_mode_kwargs required by the installed Transformers generation path"
    return "complete", "local Molmo asset appears complete"


def run_repair(*, execute: bool = False, output_root: Path = Path("outputs/assets/repair")) -> dict[str, object]:
    inspection = inspect_local_asset(LOCAL_PATH)
    status, reason = classify(inspection)
    command = download_command(LOCAL_PATH)
    result: CommandResult | None = None
    if execute and status in {"download_required", "incomplete"}:
        env = command_environment(default_hf_endpoint=DEFAULT_HF_ENDPOINT)
        result = run_command(command, env=env)
        inspection = inspect_local_asset(LOCAL_PATH)
        status, reason = classify(inspection)
        if result.returncode != 0:
            status = "blocked_download_failed"
            reason = f"Molmo download command failed with exit code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
    report: dict[str, object] = {
        "alias": ALIAS,
        "status": status,
        "mode": "execute" if execute else "dry_run",
        "dry_run": not execute,
        "execute": execute,
        "reason": reason,
        "model_id": MODEL_ID,
        "local_path": str(LOCAL_PATH),
        "download_command": command,
        "inspection": inspection,
        "diagnostic_script": str(DIAGNOSTIC_SCRIPT),
        "normal_pipeline_required": True,
        "verification_authority": "normal asset audit/smoke/validation pipeline must pass before this model can be marked verified",
    }
    if result is not None:
        report["download_result"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    json_path, md_path = write_model_report(output_root=output_root, alias=ALIAS, report=report)
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_repair(execute=bool(args.execute), output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
