#!/usr/bin/env python3
"""Shared helpers for temporary Experiment 1.7 asset repair scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import subprocess
from typing import Mapping, Sequence


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class ExecuteAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        setattr(namespace, self.dest, True)
        setattr(namespace, "dry_run", False)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action=ExecuteAction, nargs=0, default=False)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/assets/repair"))
    return parser


def normalize_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.execute:
        args.dry_run = False
    else:
        args.dry_run = True
    return args


def command_environment(
    base_env: Mapping[str, str] | None = None,
    *,
    default_hf_endpoint: str | None = None,
) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    if default_hf_endpoint and not env.get("HF_ENDPOINT"):
        env["HF_ENDPOINT"] = default_hf_endpoint
    return env


def run_command(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env) if env is not None else None,
        )
    except FileNotFoundError as error:
        return CommandResult(127, "", str(error))
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def write_model_report(
    *,
    output_root: Path,
    alias: str,
    report: Mapping[str, object],
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{alias}_repair_report.json"
    md_path = output_root / f"{alias}_repair_report.md"
    payload = dict(report)
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(payload), encoding="utf-8")
    return json_path, md_path


def markdown_report(report: Mapping[str, object]) -> str:
    alias = str(report.get("alias", "unknown"))
    lines = [
        f"# Repair Report: {alias}",
        "",
        f"- status: {report.get('status', '')}",
        f"- mode: {report.get('mode', '')}",
        f"- reason: {report.get('reason', '')}",
        "",
        "## Details",
        "```json",
        json.dumps(report, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_summary_report(*, output_root: Path, report: Mapping[str, object]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "remaining_asset_repair_summary.json"
    md_path = output_root / "REMAINING_ASSET_REPAIR_SUMMARY.md"
    payload = dict(report)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report({"alias": "remaining-assets", **payload}), encoding="utf-8")
    return json_path, md_path


def file_list(path: Path, patterns: Sequence[str]) -> list[str]:
    if not path.is_dir():
        return []
    names: set[str] = set()
    for pattern in patterns:
        names.update(candidate.name for candidate in path.glob(pattern))
    return sorted(names)


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
