#!/usr/bin/env python3
"""Diagnose Phi-4 multimodal meta-tensor loading issues without wrapper edits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CommandResult, add_common_args, read_json, run_command


ALIAS = "phi-4-multimodal-instruct"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/Phi-4-multimodal-instruct")
REPORT_JSON = "phi4_meta_tensor_repair_report.json"
REPORT_MD = "phi4_meta_tensor_repair_report.md"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report() -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_version": sys.version.split()[0],
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "accelerate_version": package_version("accelerate"),
        "peft_version": package_version("peft"),
    }


def inspect_local_asset(local_path: Path = LOCAL_PATH) -> dict[str, object]:
    index_files = sorted(path.name for path in local_path.glob("*.index.json")) if local_path.is_dir() else []
    safetensors = sorted(path.name for path in local_path.glob("*.safetensors")) if local_path.is_dir() else []
    return {
        "local_path": str(local_path),
        "path_exists": local_path.exists(),
        "path_is_directory": local_path.is_dir(),
        "config_exists": (local_path / "config.json").is_file(),
        "config": read_json(local_path / "config.json"),
        "index_files": index_files,
        "safetensors_shards": safetensors,
    }


def safe_loading_strategies() -> list[dict[str, object]]:
    return [
        {
            "name": "low_cpu_mem_usage_false_device_map_none",
            "kwargs": {"low_cpu_mem_usage": False, "device_map": None},
        },
        {"name": "device_map_none", "kwargs": {"device_map": None}},
        {"name": "device_map_auto", "kwargs": {"device_map": "auto"}},
    ]


def install_peft() -> CommandResult:
    return run_command([sys.executable, "-m", "pip", "install", "peft"])


def run_load_strategy(*, local_path: Path, strategy: dict[str, object]) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM

        kwargs = dict(strategy.get("kwargs", {}))
        model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        meta_names = [
            name
            for name, parameter in model.named_parameters()
            if getattr(parameter, "device", None) is not None and str(parameter.device) == "meta"
        ]
        del model
        return {
            "status": "meta_tensors_remaining" if meta_names else "ok",
            "meta_parameter_names": meta_names[:50],
            "num_meta_parameters": len(meta_names),
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "meta_parameter_names": [],
            "num_meta_parameters": 0,
        }


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
                "# Phi4 Meta Tensor Repair Report",
                "",
                f"- status: {report['status']}",
                f"- reason: {report['reason']}",
                "",
                "```json",
                json.dumps(report, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_repair(
    *,
    local_path: Path = LOCAL_PATH,
    output_root: Path = Path("outputs/assets/repair"),
    execute: bool = False,
    allow_install_peft: bool = False,
) -> dict[str, object]:
    env = environment_report()
    inspection = inspect_local_asset(local_path)
    strategies = safe_loading_strategies()
    peft_install_attempted = False
    peft_install_result: dict[str, object] | None = None
    status = "diagnostic_planned"
    reason = "dry-run only; load diagnostics were not run"
    diagnostics: list[dict[str, object]] = []

    if env["peft_version"] is None:
        if not allow_install_peft:
            status = "blocked"
            reason = "missing dependency: peft; rerun with --allow-install-peft to install only peft"
            report = _report(env, inspection, strategies, diagnostics, status, reason, execute, False, peft_install_attempted, peft_install_result)
            _write_report(output_root, report)
            return report
        if execute:
            peft_install_attempted = True
            result = install_peft()
            peft_install_result = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            if result.returncode != 0:
                status = "blocked"
                reason = "peft install failed"
                report = _report(env, inspection, strategies, diagnostics, status, reason, execute, False, peft_install_attempted, peft_install_result)
                _write_report(output_root, report)
                return report

    if execute:
        for strategy in strategies:
            result = run_load_strategy(local_path=local_path, strategy=strategy)
            diagnostics.append({"strategy": strategy, **result})
            if result.get("status") == "ok":
                status = "load_strategy_available"
                reason = f"safe loading strategy avoids meta tensors: {strategy['name']}"
                break
            if result.get("status") == "meta_tensors_remaining":
                status = "blocked"
                reason = "meta tensors remain after safe loading strategy"
                break
        if not diagnostics:
            status = "blocked"
            reason = "no load diagnostics ran"
        elif status == "diagnostic_planned":
            status = "blocked"
            first_error = next((str(item.get("error")) for item in diagnostics if item.get("error")), "")
            reason = "all safe loading strategies failed before meta tensor inspection"
            if first_error:
                reason = f"{reason}: {first_error}"

    report = _report(env, inspection, strategies, diagnostics, status, reason, execute, bool(diagnostics), peft_install_attempted, peft_install_result)
    _write_report(output_root, report)
    return report


def _report(
    env: dict[str, object],
    inspection: dict[str, object],
    strategies: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    status: str,
    reason: str,
    execute: bool,
    load_diagnostics_ran: bool,
    peft_install_attempted: bool,
    peft_install_result: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "alias": ALIAS,
        "status": status,
        "mode": "execute" if execute else "dry_run",
        "reason": reason,
        "environment": env,
        "inspection": inspection,
        "safe_loading_strategies": strategies,
        "load_diagnostics": diagnostics,
        "load_diagnostics_ran": load_diagnostics_ran,
        "peft_install_attempted": peft_install_attempted,
        "peft_install_result": peft_install_result,
        "normal_pipeline_required": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--allow-install-peft", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_repair(output_root=args.output_root, execute=bool(args.execute), allow_install_peft=bool(args.allow_install_peft))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
