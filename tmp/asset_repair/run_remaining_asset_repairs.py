#!/usr/bin/env python3
"""Run temporary repair checks or explicit repairs for remaining Experiment 1 assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import accept_molmo_separate_env
import final_model_panel_decisions
import repair_gemma4_local_asset
import repair_llava_v15_hf_asset
import repair_phi4_meta_tensor
from common import add_common_args, write_summary_report


GEMMA4_LOCAL_PATH = repair_gemma4_local_asset.LOCAL_PATH
PHI4_LOCAL_PATH = repair_phi4_meta_tensor.LOCAL_PATH
MOLMO_SOURCE_ROOT = Path("outputs/assets_molmo_tf457")
LLAVA_REGISTRY_PATH = Path("configs/assets/model_assets.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--repair-gemma4", action="store_true", default=False)
    parser.add_argument("--repair-phi4", action="store_true", default=False)
    parser.add_argument("--accept-molmo-separate-env", action="store_true", default=False)
    parser.add_argument("--repair-llava-v15", action="store_true", default=False)
    parser.add_argument("--allow-install-peft", action="store_true", default=False)
    return parser


def run_repairs(
    *,
    execute: bool = False,
    repair_gemma4: bool = True,
    repair_phi4: bool = True,
    accept_molmo: bool = True,
    repair_llava_v15: bool = True,
    allow_install_peft: bool = False,
    output_root: Path = Path("outputs/assets/repair"),
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    if repair_gemma4:
        reports.append(repair_gemma4_local_asset.run_repair(local_path=GEMMA4_LOCAL_PATH, execute=execute, output_root=output_root))
    if repair_phi4:
        reports.append(
            repair_phi4_meta_tensor.run_repair(
                local_path=PHI4_LOCAL_PATH,
                execute=execute,
                allow_install_peft=allow_install_peft,
                output_root=output_root,
            )
        )
    molmo_acceptance_path: Path | None = None
    if accept_molmo:
        molmo_report = accept_molmo_separate_env.accept_molmo_separate_env(
            source_root=MOLMO_SOURCE_ROOT,
            output_root=output_root,
            execute=execute,
        )
        reports.append(molmo_report)
        molmo_acceptance_path = output_root / "molmo_separate_env_acceptance.json"
    if repair_llava_v15:
        reports.append(
            repair_llava_v15_hf_asset.repair_llava_v15_hf_asset(
                registry_path=LLAVA_REGISTRY_PATH,
                output_root=output_root,
                execute=execute,
            )
        )
    decisions = final_model_panel_decisions.write_final_model_panel_decisions(
        output_root=output_root,
        molmo_acceptance_path=molmo_acceptance_path,
        phi4_report_path=output_root / "phi4_meta_tensor_repair_report.json",
        execute=execute,
    )
    summary: dict[str, object] = {
        "mode": "execute" if execute else "dry_run",
        "dry_run": not execute,
        "execute": execute,
        "allow_install_peft": allow_install_peft,
        "reports": reports,
        "final_model_panel_decisions": decisions,
        "normal_pipeline_required": True,
        "verification_authority": "repair reports are diagnostic only; normal asset audit/smoke/validation remains authoritative",
    }
    json_path, md_path = write_summary_report(output_root=output_root, report=summary)
    summary["summary_json"] = str(json_path)
    summary["summary_markdown"] = str(md_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    explicit_selection = args.repair_gemma4 or args.repair_phi4 or args.accept_molmo_separate_env or args.repair_llava_v15
    run_repairs(
        execute=bool(args.execute),
        repair_gemma4=args.repair_gemma4 or not explicit_selection,
        repair_phi4=args.repair_phi4 or not explicit_selection,
        accept_molmo=args.accept_molmo_separate_env or not explicit_selection,
        repair_llava_v15=args.repair_llava_v15 or not explicit_selection,
        allow_install_peft=bool(args.allow_install_peft),
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
