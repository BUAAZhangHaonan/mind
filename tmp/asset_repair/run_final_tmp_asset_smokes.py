#!/usr/bin/env python3
"""Summarize tmp-only smoke status for the four remaining Experiment 1 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, normalize_mode, read_json


TARGETS = (
    "gemma-4-12b-it",
    "phi-4-multimodal-instruct",
    "molmo-7b-d-0924",
    "llava-v1.5-7b",
)
REPORT_JSON = "final_tmp_asset_smoke_summary.json"
REPORT_MD = "FINAL_TMP_ASSET_SMOKE_SUMMARY.md"


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument(
        "--run-smokes",
        action="store_true",
        default=False,
        help="Actually invoke tmp smoke runners. Without this flag, only existing reports are summarized.",
    )
    return parser


def _report_paths(output_root: Path) -> dict[str, Path]:
    return {
        "gemma-4-12b-it": output_root / "gemma4_unified_tmp_smoke_report.json",
        "phi-4-multimodal-instruct": output_root / "phi4_tmp_smoke_report.json",
        "molmo-7b-d-0924": output_root / "molmo_separate_env_acceptance.json",
        "llava-v1.5-7b": output_root / "llava_v15_tmp_smoke_report.json",
    }


def _classification(alias: str, payload: dict[str, Any]) -> tuple[str, str]:
    if not payload:
        return "missing_report", "tmp report is missing"
    status = str(payload.get("status") or "")
    reason = str(payload.get("reason") or "")
    if alias == "molmo-7b-d-0924":
        if status == "verified_separate_env":
            return "verified_separate_env", reason or "separate environment accepted"
        return "blocked_tmp", reason or status
    if alias == "llava-v1.5-7b":
        if status == "verified":
            return "verified_tmp", reason or "tmp LLaVA-v1.5 smoke verified"
        return "blocked_tmp", reason or status
    if alias == "phi-4-multimodal-instruct":
        if status == "verified_tmp":
            return "verified_tmp", reason or "tmp Phi4 smoke verified"
        return "blocked_tmp", reason or status
    if alias == "gemma-4-12b-it":
        if status in {"verified_tmp", "verified_tmp_smoke"} or payload.get("runnable") is True:
            return "verified_tmp", reason or "tmp Gemma4 smoke verified"
        return "blocked_needs_separate_runtime", reason or status
    return "unknown", reason or status


def _run_tmp_smokes(*, output_root: Path, stage0_root: Path, device: str, smoke_limit: int) -> None:
    import run_gemma4_unified_tmp_smoke
    import run_phi4_tmp_smoke
    import run_llava_v15_tmp_smoke
    import accept_molmo_separate_env

    run_gemma4_unified_tmp_smoke.run_tmp_smoke(
        output_root=output_root,
        stage0_root=stage0_root,
        device=device,
        smoke_limit=smoke_limit,
        execute=True,
    )
    run_phi4_tmp_smoke.run(
        output_root=output_root,
        stage0_root=stage0_root,
        device=device,
        execute=True,
    )
    accept_molmo_separate_env.accept_molmo_separate_env(
        source_root=Path("outputs/assets_molmo_tf457"),
        output_root=output_root,
        execute=True,
    )
    run_llava_v15_tmp_smoke.run_tmp_smoke(
        output_root=output_root,
        stage0_root=stage0_root,
        datasets=("pope", "repope", "dash-b"),
        smoke_limit=smoke_limit,
        device=device,
        execute=True,
        allow_cpu=False,
    )


def build_summary(*, output_root: Path, execute: bool, run_smokes: bool, stage0_root: Path, device: str, smoke_limit: int) -> dict[str, Any]:
    if execute and run_smokes:
        _run_tmp_smokes(output_root=output_root, stage0_root=stage0_root, device=device, smoke_limit=smoke_limit)

    paths = _report_paths(output_root)
    models: dict[str, dict[str, Any]] = {}
    for alias in TARGETS:
        payload = read_json(paths[alias])
        classification, reason = _classification(alias, payload)
        models[alias] = {
            "classification": classification,
            "reason": reason,
            "report_path": str(paths[alias]),
            "raw_status": payload.get("status") if payload else None,
        }
        if alias == "gemma-4-12b-it" and payload:
            models[alias]["runtime"] = payload.get("runtime")
            models[alias]["processor_wiring"] = payload.get("processor_wiring")
            models[alias]["non_unified_class_incompatibility"] = payload.get("non_unified_class_incompatibility")
        if alias == "phi-4-multimodal-instruct" and payload:
            models[alias]["loading_plan"] = payload.get("loading_plan")
            models[alias]["smoke_result"] = payload.get("smoke_result")
        if alias == "llava-v1.5-7b" and payload:
            models[alias]["rows"] = payload.get("rows")
            models[alias]["total_layers"] = payload.get("total_layers")
            models[alias]["hidden_dim"] = payload.get("hidden_dim")
        if alias == "molmo-7b-d-0924" and payload:
            models[alias]["checked_shards"] = payload.get("checked_shards")

    runnable_classes = {"verified_tmp", "verified_separate_env"}
    blocked = [alias for alias, item in models.items() if item["classification"] not in runnable_classes]
    report: dict[str, Any] = {
        "mode": "execute" if execute else "dry_run",
        "run_smokes": run_smokes,
        "status": "blocked" if blocked else "verified_tmp_all",
        "reason": "one or more tmp assets are not runnable" if blocked else "all tmp assets are runnable",
        "target_models": list(TARGETS),
        "verified_tmp": sorted(alias for alias, item in models.items() if item["classification"] == "verified_tmp"),
        "verified_separate_env": sorted(
            alias for alias, item in models.items() if item["classification"] == "verified_separate_env"
        ),
        "blocked": sorted(blocked),
        "models": models,
        "production_wrapper_modified": False,
        "stageA_started": False,
        "full_cache_extraction_started": False,
        "training_started": False,
    }
    write_summary(output_root, report)
    return report


def write_summary(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / REPORT_JSON
    md_path = output_root / REPORT_MD
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = ["# Final Tmp Asset Smoke Summary", "", f"- status: {report['status']}", f"- reason: {report['reason']}", ""]
    for alias in TARGETS:
        item = report["models"][alias]
        lines.extend(
            [
                f"## {alias}",
                "",
                f"- classification: {item['classification']}",
                f"- reason: {item['reason']}",
                f"- report: {item['report_path']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = normalize_mode(build_parser().parse_args(argv))
    build_summary(
        output_root=args.output_root,
        execute=bool(args.execute),
        run_smokes=bool(args.run_smokes),
        stage0_root=args.stage0_root,
        device=args.device,
        smoke_limit=args.smoke_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
