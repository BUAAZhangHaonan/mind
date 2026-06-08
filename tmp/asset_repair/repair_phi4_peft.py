#!/usr/bin/env python3
"""Temporary Phi-4 peft dependency repair for Experiment 1.7."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CommandResult, add_common_args, run_command, write_model_report


ALIAS = "phi-4-multimodal-instruct"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/Phi-4-multimodal-instruct")


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--allow-install-peft", action="store_true", default=False)
    return parser


def python_executable() -> str:
    return sys.executable


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report() -> dict[str, object]:
    return {
        "python_executable": python_executable(),
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "accelerate_version": package_version("accelerate"),
        "peft_version": package_version("peft"),
    }


def planned_install_command() -> list[str]:
    return [python_executable(), "-m", "pip", "install", "peft"]


def preflight_plan(command: list[str]) -> dict[str, object]:
    return {
        "command": command,
        "planned_install": [{"name": "peft"}],
        "subprocess_executed": False,
    }


def run_repair(
    *,
    execute: bool = False,
    allow_install_peft: bool = False,
    output_root: Path = Path("outputs/assets/repair"),
) -> dict[str, object]:
    before = environment_report()
    command = planned_install_command()
    status = "already_installed" if before.get("peft_version") else "install_required"
    reason = "peft is already installed" if status == "already_installed" else "missing dependency: peft"
    install_result: CommandResult | None = None
    plan: dict[str, object] = {}

    if not before.get("peft_version"):
        if execute and not allow_install_peft:
            status = "blocked_missing_allow_install_peft"
            reason = "--allow-install-peft is required before installing peft"
        else:
            plan = preflight_plan(command)
        if execute and allow_install_peft:
            install_result = run_command(command)
            after_install = package_version("peft")
            status = "installed" if install_result.returncode == 0 and after_install else "install_completed_but_peft_not_detected"
            reason = "peft install command completed" if status == "installed" else "peft install command did not leave peft import metadata available"

    after = environment_report()
    report: dict[str, object] = {
        "alias": ALIAS,
        "status": status,
        "mode": "execute" if execute else "dry_run",
        "dry_run": not execute,
        "execute": execute,
        "allow_install_peft": allow_install_peft,
        "reason": reason,
        "local_path": str(LOCAL_PATH),
        "planned_command": command,
        "preflight_plan": plan,
        "environment_before": before,
        "environment_after": after,
        "normal_pipeline_required": True,
        "verification_authority": "normal asset audit/smoke/validation pipeline must pass before this model can be marked verified",
    }
    if install_result is not None:
        report["install_result"] = {
            "returncode": install_result.returncode,
            "stdout": install_result.stdout,
            "stderr": install_result.stderr,
        }
    json_path, md_path = write_model_report(output_root=output_root, alias=ALIAS, report=report)
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_repair(execute=bool(args.execute), allow_install_peft=bool(args.allow_install_peft), output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
