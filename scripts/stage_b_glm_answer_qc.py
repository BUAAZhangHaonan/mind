#!/usr/bin/env python3
"""Run offline Stage B GLM answer quality control."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) in sys.path:
    sys.path.remove(str(REPO_SRC))
sys.path.insert(0, str(REPO_SRC))

from mind.trajectory.stage_b_glm_qc import (  # noqa: E402
    DEFAULT_GLM_QC_DATASETS,
    GLM_MODEL_ALIAS,
    scan_glm_cache_rows,
    write_glm_qc_reports,
)
from mind.trajectory.stage_b_manifest import load_stage_b_panel_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageB"))
    parser.add_argument("--model-alias", default=GLM_MODEL_ALIAS)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_GLM_QC_DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preflight_dir = args.output_root / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_stage_b_panel_manifest(args.full_cache_root)
    rows = scan_glm_cache_rows(
        manifest,
        args.full_cache_root,
        model_alias=args.model_alias,
        dataset_families=args.datasets,
        limit=args.limit,
    )
    payload = write_glm_qc_reports(
        rows,
        json_path=preflight_dir / "glm_answer_qc.json",
        markdown_path=preflight_dir / "glm_answer_qc.md",
    )
    summary = payload["summary"]
    print(
        "Stage B GLM QC complete "
        f"rows={summary['num_rows']} nonparseable={summary['num_nonparseable']} "
        f"json={preflight_dir / 'glm_answer_qc.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
