#!/usr/bin/env python3
"""Inspect and repair the local Gemma 4 12B Unified asset metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CommandResult, add_common_args, command_environment, read_json, run_command


ALIAS = "gemma-4-12b-it"
MODEL_ID = "google/gemma-4-12B-it"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/gemma-4-12B-it")
EXPECTED_SHA256 = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
REPORT_JSON = "gemma4_local_asset_repair_report.json"
REPORT_MD = "gemma4_local_asset_repair_report.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> dict[str, object]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        return {"readable": True, "tensor_key_count": len(keys), "sample_keys": keys[:10]}
    except Exception as error:
        return {"readable": False, "tensor_key_count": 0, "error": f"{type(error).__name__}: {error}"}


def _is_active_moe_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    return True


def _active_moe_indicators(payload: object, prefix: str = "") -> list[str]:
    keys = {
        "enable_moe_block",
        "num_experts",
        "n_routed_experts",
        "experts_per_tok",
        "num_local_experts",
        "router_aux_loss_coef",
        "moe_intermediate_size",
        "top_k_experts",
    }
    found: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            label = key if not prefix else f"{prefix}.{key}"
            normalized = key.lower()
            if (normalized in keys or normalized == "moe") and _is_active_moe_value(value):
                found.append(label)
            found.extend(_active_moe_indicators(value, label))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_active_moe_indicators(value, f"{prefix}[{index}]"))
    return found


def _processor_config_has_image_processor(path: Path) -> bool:
    config = read_json(path / "processor_config.json")
    image_processor = config.get("image_processor")
    return isinstance(image_processor, Mapping) and bool(image_processor.get("image_processor_type"))


def _has_tokenizer(path: Path) -> bool:
    return any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"))


def _has_processor(path: Path) -> bool:
    return any((path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json"))


def _has_image_processor(path: Path) -> bool:
    return any((path / name).is_file() for name in ("image_processor_config.json", "preprocessor_config.json")) or _processor_config_has_image_processor(path)


def _has_thinking_disable(path: Path) -> bool:
    for name in ("chat_template.jinja", "chat_template.json", "tokenizer_config.json"):
        file_path = path / name
        if file_path.is_file() and "enable_thinking" in file_path.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def _resolve_layers(config: Mapping[str, object]) -> int | None:
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping) and isinstance(text_config.get("num_hidden_layers"), int):
        return int(text_config["num_hidden_layers"])
    if isinstance(config.get("num_hidden_layers"), int):
        return int(config["num_hidden_layers"])
    return None


def _resolve_hidden_size(config: Mapping[str, object]) -> int | None:
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping) and isinstance(text_config.get("hidden_size"), int):
        return int(text_config["hidden_size"])
    if isinstance(config.get("hidden_size"), int):
        return int(config["hidden_size"])
    return None


def inspect_local_asset(local_path: Path = LOCAL_PATH) -> dict[str, object]:
    config = read_json(local_path / "config.json")
    weight_path = local_path / "model.safetensors"
    sha = sha256_file(weight_path) if weight_path.is_file() else ""
    safetensors = inspect_safetensors(weight_path) if weight_path.is_file() else {"readable": False, "tensor_key_count": 0}
    layers = _resolve_layers(config)
    hidden_size = _resolve_hidden_size(config)
    return {
        "local_path": str(local_path),
        "path_exists": local_path.exists(),
        "path_is_directory": local_path.is_dir(),
        "config_exists": (local_path / "config.json").is_file(),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "tokenizer_files_exist": _has_tokenizer(local_path),
        "processor_files_exist": _has_processor(local_path),
        "image_processor_exists": _has_image_processor(local_path),
        "generation_config_exists": (local_path / "generation_config.json").is_file(),
        "chat_template_exists": any((local_path / name).is_file() for name in ("chat_template.jinja", "chat_template.json")),
        "thinking_disable_evidence": "enable_thinking=False" if _has_thinking_disable(local_path) else "",
        "model_safetensors_exists": weight_path.is_file(),
        "model_safetensors_sha256": sha,
        "sha256_matches": sha == EXPECTED_SHA256,
        "safetensors_readable": bool(safetensors.get("readable")),
        "tensor_key_count": int(safetensors.get("tensor_key_count", 0)),
        "sample_tensor_keys": safetensors.get("sample_keys", []),
        "active_moe_indicators": _active_moe_indicators(config),
        "total_layers": layers,
        "expected_total_layers": 48,
        "total_layers_matches_expected": layers == 48,
        "hidden_size": hidden_size,
    }


def _metadata_missing(inspection: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    if not inspection.get("config_exists"):
        missing.append("config.json")
    if not inspection.get("tokenizer_files_exist"):
        missing.append("tokenizer files")
    if not inspection.get("processor_files_exist"):
        missing.append("processor files")
    if not inspection.get("image_processor_exists"):
        missing.append("image processor metadata")
    if not inspection.get("chat_template_exists"):
        missing.append("chat template")
    return missing


def _metadata_download_command(local_path: Path) -> list[str]:
    executable = "hf" if shutil.which("hf") else "huggingface-cli"
    command = [
        executable,
        "download",
        MODEL_ID,
        "--local-dir",
        str(local_path),
        "--exclude",
        "model.safetensors",
        "--exclude",
        "*.safetensors",
    ]
    if executable == "huggingface-cli":
        command.extend(["--local-dir-use-symlinks", "False"])
    return command


def _write_report(output_root: Path, report: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / REPORT_JSON
    md_path = output_root / REPORT_MD
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gemma4 Local Asset Repair Report",
                "",
                f"- status: {report['status']}",
                f"- reason: {report['reason']}",
                f"- local_path: {report['local_path']}",
                "",
                "```json",
                json.dumps(report, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_repair(*, local_path: Path = LOCAL_PATH, output_root: Path = Path("outputs/assets/repair"), execute: bool = False) -> dict[str, object]:
    inspection = inspect_local_asset(local_path)
    status = "already_present"
    reason = "local Gemma4 Unified asset appears complete"
    command_result: CommandResult | None = None
    metadata_missing = _metadata_missing(inspection)
    download_plan = {
        "model_id": MODEL_ID,
        "local_dir": str(local_path),
        "exclude": ["model.safetensors"],
        "command": _metadata_download_command(local_path),
    }

    if inspection["active_moe_indicators"]:
        status = "blocked"
        reason = "active MoE indicators detected: " + ", ".join(str(value) for value in inspection["active_moe_indicators"])
    elif inspection["model_type"] not in {"gemma4_unified", "gemma4"}:
        status = "blocked"
        reason = f"config does not identify gemma4_unified: {inspection['model_type']}"
    elif not inspection["model_safetensors_exists"]:
        status = "blocked"
        reason = "model.safetensors is missing"
    elif not inspection["sha256_matches"]:
        status = "blocked"
        reason = "model.safetensors sha256 does not match expected uploaded file"
    elif not inspection["safetensors_readable"] or int(inspection["tensor_key_count"]) <= 0:
        status = "blocked"
        reason = "model.safetensors is not readable or has no tensor keys"
    elif metadata_missing:
        status = "metadata_repair_required"
        reason = "metadata missing: " + ", ".join(metadata_missing)
    elif not inspection["total_layers_matches_expected"]:
        status = "blocked"
        reason = f"total_layers is not 48: {inspection['total_layers']}"
    elif not inspection["hidden_size"]:
        status = "blocked"
        reason = "hidden size is unresolved"
    elif not inspection["thinking_disable_evidence"]:
        status = "blocked"
        reason = "thinking disable mechanism enable_thinking=False is not recorded"

    if execute and status == "metadata_repair_required" and inspection["sha256_matches"]:
        command_result = run_command(
            _metadata_download_command(local_path),
            env=command_environment(default_hf_endpoint=DEFAULT_HF_ENDPOINT),
        )
        inspection = inspect_local_asset(local_path)
        metadata_missing = _metadata_missing(inspection)
        status = "already_present" if not metadata_missing else "metadata_repair_required"
        reason = "metadata repair command completed" if not metadata_missing else "metadata still missing: " + ", ".join(metadata_missing)
        if command_result.returncode != 0:
            status = "blocked"
            reason = f"metadata download failed: {command_result.stderr.strip() or command_result.stdout.strip()}"

    report: dict[str, object] = {
        "alias": ALIAS,
        "status": status,
        "mode": "execute" if execute else "dry_run",
        "reason": reason,
        "local_path": str(local_path),
        "inspection": inspection,
        "download_plan": download_plan,
        "thinking_disabled": bool(inspection.get("thinking_disable_evidence")),
        "normal_pipeline_required": True,
        "verification_authority": "standard asset smoke/validation pipeline must pass before main-env verification",
    }
    if command_result is not None:
        report["command_result"] = {
            "returncode": command_result.returncode,
            "stdout": command_result.stdout,
            "stderr": command_result.stderr,
        }
    _write_report(output_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    return add_common_args(argparse.ArgumentParser(description=__doc__))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_repair(output_root=args.output_root, execute=bool(args.execute))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
