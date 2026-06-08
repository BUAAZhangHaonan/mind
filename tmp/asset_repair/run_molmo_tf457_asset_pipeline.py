#!/usr/bin/env python3
"""Run Molmo smoke and validation in the separate Transformers 4.57.1 env.

This is an Experiment 1.7 temporary runner. It does not change production
wrappers. It only adds a process-local Transformers import alias needed before
the existing asset scripts import the production wrapper module.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import site
import sys
from typing import Sequence

USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
    sys.path.remove(USER_SITE)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, normalize_mode, write_model_report


ALIAS = "molmo-7b-d-0924"
DEFAULT_OUTPUT_ROOT = Path("outputs/assets_molmo_tf457")
DEFAULT_REPAIR_OUTPUT_ROOT = Path("outputs/assets/repair")
DEFAULT_MODELS = (ALIAS,)
DEFAULT_DATASETS = ("pope", "repope", "dash-b")


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--registry", type=Path, default=Path("configs/assets/model_assets.yaml"))
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--pipeline-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--steps", choices=("smoke", "validate", "both"), default="both")
    return parser


def _environment_report() -> dict[str, object]:
    report: dict[str, object] = {
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_version": sys.version.split()[0],
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
    except Exception as error:  # pragma: no cover - diagnostic only
        report["torch_version_error"] = repr(error)
    try:
        import transformers

        report["transformers_version"] = transformers.__version__
        report["has_auto_model_for_multimodal_lm"] = hasattr(transformers, "AutoModelForMultimodalLM")
        report["has_auto_model_for_image_text_to_text"] = hasattr(transformers, "AutoModelForImageTextToText")
    except Exception as error:  # pragma: no cover - diagnostic only
        report["transformers_error"] = repr(error)
    return report


def _install_process_local_transformers_alias() -> dict[str, object]:
    import transformers

    before = hasattr(transformers, "AutoModelForMultimodalLM")
    if before:
        return {
            "shim_applied": False,
            "reason": "AutoModelForMultimodalLM already exists",
            "global_transformers_files_modified": False,
        }
    if not hasattr(transformers, "AutoModelForImageTextToText"):
        raise RuntimeError("transformers has neither AutoModelForMultimodalLM nor AutoModelForImageTextToText")
    transformers.AutoModelForMultimodalLM = transformers.AutoModelForImageTextToText
    return {
        "shim_applied": True,
        "reason": "process-local alias AutoModelForMultimodalLM=AutoModelForImageTextToText",
        "global_transformers_files_modified": False,
    }


def _run_script(script_path: Path, argv: Sequence[str]) -> int:
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *argv]
        result = runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as error:
        code = error.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = previous_argv
    main_result = result.get("__return_code__")
    return int(main_result) if isinstance(main_result, int) else 0


def run_pipeline(
    *,
    execute: bool,
    output_root: Path,
    registry: Path,
    stage0_root: Path,
    pipeline_output_root: Path,
    device: str,
    smoke_limit: int,
    datasets: Sequence[str],
    models: Sequence[str],
    steps: str,
) -> dict[str, object]:
    report: dict[str, object] = {
        "alias": ALIAS,
        "mode": "execute" if execute else "dry_run",
        "status": "planned" if not execute else "running",
        "reason": "dry-run only; no smoke or validation command executed",
        "pipeline_output_root": str(pipeline_output_root),
        "models": list(models),
        "datasets": list(datasets),
        "smoke_limit": smoke_limit,
        "verification_authority": "existing asset smoke and validation scripts",
        "environment": _environment_report(),
    }
    if not execute:
        write_model_report(output_root=output_root, alias=f"{ALIAS}_tf457_pipeline", report=report)
        return report

    pipeline_output_root.mkdir(parents=True, exist_ok=True)
    shim_report = _install_process_local_transformers_alias()
    report["transformers_process_local_shim"] = shim_report
    repo_root = Path(__file__).resolve().parents[2]
    smoke_script = repo_root / "scripts" / "asset_smoke_extract.py"
    validate_script = repo_root / "scripts" / "asset_validate_hidden_states.py"
    commands: list[dict[str, object]] = []

    if steps in {"smoke", "both"}:
        smoke_argv = [
            "--registry",
            str(registry),
            "--output-root",
            str(pipeline_output_root),
            "--stage0-root",
            str(stage0_root),
            "--datasets",
            *datasets,
            "--smoke-limit",
            str(smoke_limit),
            "--device",
            device,
            "--models",
            *models,
        ]
        smoke_code = _run_script(smoke_script, smoke_argv)
        commands.append({"script": str(smoke_script), "argv": smoke_argv, "returncode": smoke_code})
        if smoke_code != 0:
            report.update({"status": "failed", "reason": f"smoke extraction returned {smoke_code}", "commands": commands})
            write_model_report(output_root=output_root, alias=f"{ALIAS}_tf457_pipeline", report=report)
            return report

    if steps in {"validate", "both"}:
        validate_argv = [
            "--output-root",
            str(pipeline_output_root),
            "--smoke-cache-root",
            str(pipeline_output_root / "smoke_cache"),
            "--models",
            *models,
        ]
        validate_code = _run_script(validate_script, validate_argv)
        commands.append({"script": str(validate_script), "argv": validate_argv, "returncode": validate_code})
        if validate_code != 0:
            report.update({"status": "failed", "reason": f"hidden-state validation returned {validate_code}", "commands": commands})
            write_model_report(output_root=output_root, alias=f"{ALIAS}_tf457_pipeline", report=report)
            return report

    summary_path = pipeline_output_root / "asset_completion_summary.json"
    model_status = None
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        model_status = payload.get("model_statuses", {}).get(ALIAS)
    report.update(
        {
            "status": str(model_status or "completed"),
            "reason": "Molmo separate-env pipeline completed; inspect asset_completion_summary.json for final status",
            "commands": commands,
            "summary_path": str(summary_path),
        }
    )
    write_model_report(output_root=output_root, alias=f"{ALIAS}_tf457_pipeline", report=report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = normalize_mode(build_parser().parse_args(argv))
    run_pipeline(
        execute=bool(args.execute),
        output_root=args.output_root,
        registry=args.registry,
        stage0_root=args.stage0_root,
        pipeline_output_root=args.pipeline_output_root,
        device=args.device,
        smoke_limit=args.smoke_limit,
        datasets=args.datasets,
        models=args.models,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
