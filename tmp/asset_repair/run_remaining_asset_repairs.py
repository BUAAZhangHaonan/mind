#!/usr/bin/env python3
"""Run temporary repair checks or explicit repairs for remaining Experiment 1 assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import repair_gemma4_download
import repair_llava_v15_asset
import repair_molmo_asset
import repair_phi4_peft
from common import add_common_args, write_summary_report


GEMMA4_LOCAL_PATH = repair_gemma4_download.LOCAL_PATH
PHI4_LOCAL_PATH = repair_phi4_peft.LOCAL_PATH
MOLMO_LOCAL_PATH = repair_molmo_asset.LOCAL_PATH
LLAVA_V15_LOCAL_PATH = repair_llava_v15_asset.LOCAL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--repair-gemma4", action="store_true", default=False)
    parser.add_argument("--repair-phi4", action="store_true", default=False)
    parser.add_argument("--repair-molmo", action="store_true", default=False)
    parser.add_argument("--repair-llava-v15", action="store_true", default=False)
    parser.add_argument("--allow-install-peft", action="store_true", default=False)
    parser.add_argument("--allow-install-tokenizer-deps", action="store_true", default=False)
    return parser


def run_repairs(
    *,
    execute: bool = False,
    repair_gemma4: bool = True,
    repair_phi4: bool = True,
    repair_molmo: bool = True,
    repair_llava_v15: bool = True,
    allow_install_peft: bool = False,
    allow_install_tokenizer_deps: bool = False,
    output_root: Path = Path("outputs/assets/repair"),
) -> dict[str, object]:
    repair_gemma4_download.LOCAL_PATH = GEMMA4_LOCAL_PATH
    repair_phi4_peft.LOCAL_PATH = PHI4_LOCAL_PATH
    repair_molmo_asset.LOCAL_PATH = MOLMO_LOCAL_PATH
    repair_llava_v15_asset.LOCAL_PATH = LLAVA_V15_LOCAL_PATH
    reports: list[dict[str, object]] = []
    if repair_gemma4:
        reports.append(repair_gemma4_download.run_repair(execute=execute, output_root=output_root))
    if repair_phi4:
        reports.append(
            repair_phi4_peft.run_repair(
                execute=execute,
                allow_install_peft=allow_install_peft,
                output_root=output_root,
            )
        )
    if repair_molmo:
        reports.append(repair_molmo_asset.run_repair(execute=execute, output_root=output_root))
    if repair_llava_v15:
        reports.append(
            repair_llava_v15_asset.run_repair(
                execute=execute,
                allow_install_tokenizer_deps=allow_install_tokenizer_deps,
                output_root=output_root,
            )
        )
    summary: dict[str, object] = {
        "mode": "execute" if execute else "dry_run",
        "dry_run": not execute,
        "execute": execute,
        "allow_install_peft": allow_install_peft,
        "allow_install_tokenizer_deps": allow_install_tokenizer_deps,
        "reports": reports,
        "normal_pipeline_required": True,
        "verification_authority": "repair reports are diagnostic only; normal asset audit/smoke/validation remains authoritative",
    }
    json_path, md_path = write_summary_report(output_root=output_root, report=summary)
    summary["summary_json"] = str(json_path)
    summary["summary_markdown"] = str(md_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    explicit_selection = args.repair_gemma4 or args.repair_phi4 or args.repair_molmo or args.repair_llava_v15
    run_repairs(
        execute=bool(args.execute),
        repair_gemma4=args.repair_gemma4 or not explicit_selection,
        repair_phi4=args.repair_phi4 or not explicit_selection,
        repair_molmo=args.repair_molmo or not explicit_selection,
        repair_llava_v15=args.repair_llava_v15 or not explicit_selection,
        allow_install_peft=bool(args.allow_install_peft),
        allow_install_tokenizer_deps=bool(args.allow_install_tokenizer_deps),
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
