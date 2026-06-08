#!/usr/bin/env python3
"""Write final panel decisions for the four remaining Experiment 1.7 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, normalize_mode


TARGETS = (
    "gemma-4-12b-it",
    "phi-4-multimodal-instruct",
    "molmo-7b-d-0924",
    "llava-v1.5-7b",
)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "final_model_panel_decisions.json"
    md_path = output_root / "FINAL_MODEL_PANEL_DECISIONS.md"
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = ["# Final Model Panel Decisions", ""]
    for alias, item in report["decisions"].items():
        lines.extend(
            [
                f"## {alias}",
                "",
                f"- classification: {item['classification']}",
                f"- reason: {item['reason']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _molmo_decision(molmo_acceptance: dict[str, Any]) -> dict[str, str]:
    if molmo_acceptance.get("status") == "verified_separate_env":
        return {
            "classification": "verified_separate_env",
            "reason": "separate-env acceptance manifest verified Molmo smoke and hidden-state validation artifacts",
        }
    if not molmo_acceptance:
        return {
            "classification": "blocked_remove_from_panel",
            "reason": "separate-env acceptance manifest is missing",
        }
    return {
        "classification": "blocked_remove_from_panel",
        "reason": str(molmo_acceptance.get("reason") or "separate-env acceptance did not pass"),
    }


def build_decisions(
    *,
    molmo_acceptance_path: Path | None = None,
    phi4_report_path: Path | None = None,
) -> dict[str, dict[str, str]]:
    molmo_acceptance = _read_json(molmo_acceptance_path)
    phi4_report = _read_json(phi4_report_path)
    phi4_reason = str(
        phi4_report.get("reason")
        or "Phi4 still has a loading blocker after safe temporary repair attempts"
    )
    return {
        "gemma-4-12b-it": {
            "classification": "blocked_manual_future_work",
            "reason": (
                "Gemma4 Unified should remain a candidate, but current main pipeline still rejects it as "
                "unsupported_by_policy through text_config.num_experts and lacks exact gemma4_unified "
                "processor/model support"
            ),
        },
        "phi-4-multimodal-instruct": {
            "classification": "blocked_remove_from_panel",
            "reason": (
                "normal smoke still reports RuntimeError: Tensor.item() cannot be called on meta tensors; "
                f"temporary safe-load diagnostic reports {phi4_reason}"
            ),
        },
        "molmo-7b-d-0924": _molmo_decision(molmo_acceptance),
        "llava-v1.5-7b": {
            "classification": "blocked_remove_from_panel",
            "reason": "LLaVA-v1.5 HF 7B asset is complete, but no production Experiment 1 wrapper is implemented",
        },
    }


def write_final_model_panel_decisions(
    *,
    output_root: Path = Path("outputs/assets/repair"),
    molmo_acceptance_path: Path | None = None,
    phi4_report_path: Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    if molmo_acceptance_path is None:
        molmo_acceptance_path = output_root / "molmo_separate_env_acceptance.json"
    if phi4_report_path is None:
        phi4_report_path = output_root / "phi4_meta_tensor_repair_report.json"
    decisions = build_decisions(molmo_acceptance_path=molmo_acceptance_path, phi4_report_path=phi4_report_path)
    report: dict[str, Any] = {
        "mode": "execute" if execute else "dry_run",
        "status": "written",
        "target_models": list(TARGETS),
        "decisions": decisions,
        "verified": sorted(alias for alias, item in decisions.items() if item["classification"] == "verified"),
        "verified_separate_env": sorted(
            alias for alias, item in decisions.items() if item["classification"] == "verified_separate_env"
        ),
        "blocked_remove_from_panel": sorted(
            alias for alias, item in decisions.items() if item["classification"] == "blocked_remove_from_panel"
        ),
        "blocked_manual_future_work": sorted(
            alias for alias, item in decisions.items() if item["classification"] == "blocked_manual_future_work"
        ),
    }
    _write(output_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--molmo-acceptance", type=Path, default=None)
    parser.add_argument("--phi4-report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = normalize_mode(build_parser().parse_args(argv))
    write_final_model_panel_decisions(
        output_root=args.output_root,
        molmo_acceptance_path=args.molmo_acceptance,
        phi4_report_path=args.phi4_report,
        execute=bool(args.execute),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
